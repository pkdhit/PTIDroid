from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import time
from typing import Any

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional dependency
    OpenAI = None


@dataclass(frozen=True)
class LLMSettings:
    base_url: str | None
    api_key: str | None
    model: str


@dataclass(frozen=True)
class LLMConfig:
    assertion: LLMSettings
    baseline: LLMSettings


def _env(name: str, fallback: str | None = None) -> str | None:
    value = os.environ.get(name)
    if value is None or not value.strip():
        return fallback
    return value.strip()


def _normalize_optional(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


@lru_cache(maxsize=1)
def _load_llm_config_file() -> dict[str, Any]:
    config_path = _env("LLM_CONFIG_FILE")
    if config_path:
        candidate = Path(config_path)
    else:
        candidate = Path(__file__).resolve().parent / "llm_config.json"

    if not candidate.exists():
        return {}

    try:
        return json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _settings_from_section(section: dict[str, Any] | None, default_model: str) -> LLMSettings:
    section = section or {}
    return LLMSettings(
        base_url=_normalize_optional(section.get("base_url")),
        api_key=_normalize_optional(section.get("api_key")),
        model=_normalize_optional(section.get("model")) or default_model,
    )


def _apply_env_overrides(settings: LLMSettings, prefix: str) -> LLMSettings:
    base_url = _env(f"{prefix}_LLM_BASE_URL") or _env(f"{prefix}_BASE_URL") or settings.base_url
    api_key = _env(f"{prefix}_LLM_API_KEY") or _env(f"{prefix}_API_KEY") or settings.api_key
    model = _env(f"{prefix}_LLM_MODEL") or _env(f"{prefix}_MODEL") or settings.model
    return LLMSettings(base_url=base_url, api_key=api_key, model=model)


def load_llm_config() -> LLMConfig:
    """Load assertion/baseline LLM settings from config file, with env overrides.

    The default config file is ``testcases/llm_config.json``. You may also set
    ``LLM_CONFIG_FILE`` to point to another JSON file.
    """

    raw_config = _load_llm_config_file()
    default_model = "Qwen/Qwen3-VL-32B-Instruct"

    assertion = _settings_from_section(raw_config.get("ASSERTION"), default_model)
    baseline = _settings_from_section(raw_config.get("BASELINE"), default_model)

    assertion = _apply_env_overrides(assertion, "ASSERTION")
    baseline = _apply_env_overrides(baseline, "BASELINE")

    return LLMConfig(assertion=assertion, baseline=baseline)


def load_llm_settings(prefix: str = "ASSERTION") -> LLMSettings:
    """Load OpenAI-compatible model settings from config file and env overrides.

    This helper supports both cloud API-key access and a local OpenAI-compatible
    server. For local Qwen3-VL deployment, point the base URL to the server side
    process launched from ``model/qwen3_vl_openai_server.py``.
    """

    config = load_llm_config()
    if prefix.upper() == "BASELINE":
        return config.baseline
    return config.assertion


@lru_cache(maxsize=16)
def _cached_openai_client(base_url: str | None, api_key: str | None):
    if OpenAI is None:
        return None
    if not base_url or not api_key:
        if base_url and not api_key:
            # Local OpenAI-compatible servers often ignore the API key, but the
            # OpenAI Python SDK still requires a non-empty value.
            api_key = "local-dummy-key"
        else:
            return None
    try:
        return OpenAI(base_url=base_url, api_key=api_key)
    except Exception:
        return None


def make_openai_client(base_url: str | None, api_key: str | None):
    return _cached_openai_client(base_url, api_key)


def extract_openai_token_usage(response: Any) -> int:
    """Best-effort extraction of total token usage from an OpenAI-compatible response."""

    usage = getattr(response, "usage", None)
    if usage is None and isinstance(response, dict):
        usage = response.get("usage")

    if usage is None:
        return 0

    if isinstance(usage, dict):
        for key in ("total_tokens", "total", "tokens"):
            value = usage.get(key)
            if isinstance(value, int) and value >= 0:
                return value
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
            return max(prompt_tokens, 0) + max(completion_tokens, 0)
        return 0

    for key in ("total_tokens", "total", "tokens"):
        value = getattr(usage, key, None)
        if isinstance(value, int) and value >= 0:
            return value

    prompt_tokens = getattr(usage, "prompt_tokens", None)
    completion_tokens = getattr(usage, "completion_tokens", None)
    if isinstance(prompt_tokens, int) and isinstance(completion_tokens, int):
        return max(prompt_tokens, 0) + max(completion_tokens, 0)

    return 0


def get_openai_chat_completion(
    client: Any,
    *,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.2,
    max_retries: int = 3,
    timeout: int = 120,
    max_output_tokens: int | None = None,
):
    """带重试的 LLM 调用。"""
    last_exception: Exception | None = None

    def _resolve_output_tokens() -> int:
        if isinstance(max_output_tokens, int) and max_output_tokens > 0:
            return max_output_tokens
        env_value = os.environ.get("LLM_MAX_OUTPUT_TOKENS") or os.environ.get("BASELINE_LLM_MAX_OUTPUT_TOKENS")
        if env_value:
            try:
                parsed = int(env_value)
                if parsed > 0:
                    return parsed
            except Exception:
                pass
        # Baseline/cloud prompts are fairly large; keep enough room for a complete JSON response.
        return 8192

    def _is_responses_model(model_name: str) -> bool:
        lowered = (model_name or "").lower()
        return lowered.startswith("Pro/moonshotai/Kimi-K2.6") or lowered.startswith("gemini-2.5-pro")

    output_tokens = _resolve_output_tokens()

    def _call_chat_completions_api() -> Any:
        request_kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "timeout": timeout,
        }
        if _is_responses_model(model):
            request_kwargs["max_completion_tokens"] = output_tokens
        else:
            request_kwargs["max_tokens"] = output_tokens

        response = client.chat.completions.create(**request_kwargs)
        if not extract_openai_chat_content(response):
            raise ValueError("chat.completions returned empty assistant content")
        return response

    def _call_responses_api() -> Any:
        response = client.responses.create(
            model=model,
            input=messages,
            temperature=temperature,
            max_output_tokens=output_tokens,
        )
        if not extract_openai_chat_content(response):
            raise ValueError("responses API returned empty assistant content")
        return response

    for attempt in range(max_retries):
        try:
            try:
                return _call_chat_completions_api()
            except Exception as chat_exc:
                if _is_responses_model(model):
                    try:
                        print(f"[LLM] chat.completions failed, falling back to responses API: {chat_exc}")
                        return _call_responses_api()
                    except Exception as responses_exc:
                        print(f"[LLM] responses API also failed: {responses_exc}")
                        raise responses_exc
                raise
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"[LLM] Request failed (attempt {attempt+1}/{max_retries}), retrying in {wait_time}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"[LLM] All retries failed: {e}")
                raise

    if last_exception is not None:
        raise last_exception
    raise RuntimeError("LLM request failed without exception")


def extract_openai_chat_content(response: Any) -> str:
    """Best-effort extraction of assistant text from OpenAI-compatible responses."""

    def _normalize_text(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, dict):
            # Prefer obvious textual fields first, then recurse only into known text-bearing shapes.
            for key in ("content", "text", "output_text", "output", "refusal"):
                if key in value:
                    text = _normalize_text(value.get(key))
                    if text:
                        return text
            if isinstance(value.get("message"), dict):
                return _normalize_text(value.get("message"))
            if isinstance(value.get("output"), list):
                return _normalize_text(value.get("output"))
            return ""
        if isinstance(value, list):
            parts: list[str] = []
            for item in value:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    if item.get("type") == "text" and isinstance(item.get("text"), str):
                        parts.append(item["text"])
                    elif isinstance(item.get("text"), str):
                        parts.append(item["text"])
                else:
                    text = _normalize_text(getattr(item, "content", None))
                    if not text:
                        text = _normalize_text(getattr(item, "text", None))
                    if not text:
                        text = _normalize_text(getattr(item, "output_text", None))
                    if text:
                        parts.append(text)
            return "".join(parts).strip()

        for attr in ("content", "text", "output_text", "output", "refusal"):
            if hasattr(value, attr):
                text = _normalize_text(getattr(value, attr))
                if text:
                    return text

        if hasattr(value, "model_dump"):
            try:
                dumped = value.model_dump()
                return _normalize_text(dumped)
            except Exception:
                pass

        if hasattr(value, "dict"):
            try:
                dumped = value.dict()
                return _normalize_text(dumped)
            except Exception:
                pass

        return ""

    def _walk(node: Any) -> list[Any]:
        found: list[Any] = []
        if node is None:
            return found

        if isinstance(node, dict):
            for key in ("content", "text", "output_text", "message", "output", "refusal"):
                if key in node:
                    found.append(node.get(key))
            for value in node.values():
                if isinstance(value, (dict, list)):
                    found.extend(_walk(value))
        elif isinstance(node, list):
            for item in node:
                found.extend(_walk(item))
        else:
            for attr in ("content", "text", "output_text", "output", "refusal"):
                if hasattr(node, attr):
                    found.append(getattr(node, attr))
        # Common nested object shapes (choices/messages).
            for attr in ("choices", "message", "output", "messages", "content"):
                if hasattr(node, attr):
                    try:
                        found.extend(_walk(getattr(node, attr)))
                    except Exception:
                        pass
        return found

    candidates: list[Any] = []

    direct_output = getattr(response, "output_text", None)
    if direct_output:
        candidates.append(direct_output)

    try:
        response_dump = response.model_dump() if hasattr(response, "model_dump") else None
    except Exception:
        response_dump = None
    if isinstance(response_dump, dict):
        candidates.extend(_walk(response_dump))

    choices = getattr(response, "choices", None) or []
    if choices:
        first_choice = choices[0]
        message = getattr(first_choice, "message", None)
        if message is not None:
            candidates.append(getattr(message, "content", None))
            candidates.append(getattr(message, "output_text", None))
            try:
                message_dump = message.model_dump() if hasattr(message, "model_dump") else None
            except Exception:
                message_dump = None
            if isinstance(message_dump, dict):
                candidates.append(message_dump.get("content"))
                candidates.append(message_dump.get("output_text"))

        choice_dump = None
        try:
            choice_dump = first_choice.model_dump() if hasattr(first_choice, "model_dump") else None
        except Exception:
            choice_dump = None
        if isinstance(choice_dump, dict):
            candidates.append(choice_dump.get("message", {}).get("content") if isinstance(choice_dump.get("message"), dict) else None)
            candidates.append(choice_dump.get("text"))
            candidates.extend(_walk(choice_dump))

    for candidate in candidates:
        text = _normalize_text(candidate)
        if text:
            return text

    return ""
