from __future__ import annotations

import base64
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llm_client import extract_openai_token_usage, get_openai_chat_completion, load_llm_settings, make_openai_client

# 从 control_selector 导入所有控件筛选相关函数
from control_selector import (
    _normalize_text,
    _tokenize_loose,
    _soft_token_overlap_score,
    _collect_ui_summary_candidates,
    _extract_ui_summary,
    _extract_ui_summary_from_page_source,
    _build_xpath_hint_from_node,
    _snapshot_page_source,
    _extract_related_ui_summary,
    _format_ui_summary_for_prompt,
)

try:
    from androguard.core.bytecodes.apk import APK
except Exception:  # pragma: no cover - optional dependency
    APK = None


@dataclass(frozen=True)
class PhaseAssertionResult:
    assertions: list[str]
    reason: str
    raw_response: str
    selected_targets: list[dict[str, str]] | None = None
    token_usage: int = 0
    insert_after_step: int | None = None


@dataclass(frozen=True)
class AssertionGenerationResult:
    should_insert: bool
    assertion: str | None
    reason: str
    candidate: dict[str, Any] | None = None
    token_usage: int = 0


def _debug_log(message: str):
    print(f"[ASSERTION_DEBUG] {message}")


def _format_budget_usage(current_bytes: int, max_bytes: int) -> str:
    if max_bytes <= 0:
        return f"{current_bytes} bytes"
    percent = (current_bytes / max_bytes) * 100
    return f"{current_bytes} bytes / {max_bytes} bytes ({percent:.1f}%)"


NO_SELECTOR_UI_SUMMARY_BYTE_LIMIT = 10 * 1024
NO_SELECTOR_INPUT_BYTE_LIMIT = 7 * 1024
ASSERTION_PROMPT_INPUT_BYTE_LIMIT = 10 * 1024


def _truncate_ui_summary_to_byte_limit(ui_summary: list[dict[str, Any]], max_bytes: int) -> tuple[list[dict[str, Any]], bool]:
    """按 UTF-8 字节数截断 UI summary，优先保留前面的控件信息。"""
    if max_bytes <= 0:
        return [], True

    kept: list[dict[str, Any]] = []
    current_bytes = 2  # []
    truncated = False

    for item in ui_summary:
        item_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        item_bytes = len(item_json.encode("utf-8"))
        separator_bytes = 1 if kept else 0
        if current_bytes + separator_bytes + item_bytes > max_bytes:
            truncated = True
            break
        kept.append(item)
        current_bytes += separator_bytes + item_bytes

    return kept, truncated


def _is_container_like_control(control: dict[str, Any]) -> bool:
    class_name = _normalize_text(str(control.get("class", "")))
    if not class_name:
        return False
    container_keywords = (
        "layout",
        "viewgroup",
        "framelayout",
        "linearlayout",
        "relativelayout",
        "constraintlayout",
        "scrollview",
        "listview",
        "recyclerview",
        "drawerlayout",
        "coordinatorlayout",
        "nestedscrollview",
    )
    return any(keyword in class_name for keyword in container_keywords)


def _build_compact_ui_controls(page_source: str | None, drop_container_controls: bool = False) -> list[dict[str, str]]:
    """将 XML 压缩成只保留关键字段的控件列表。"""
    if not page_source:
        return []

    ui_summary = _extract_ui_summary_from_page_source(page_source, limit=None)
    compact_controls: list[dict[str, str]] = []
    seen_signatures: set[str] = set()

    for item in ui_summary:
        if not isinstance(item, dict):
            continue

        control: dict[str, str] = {}
        for key in ("xpath", "content-desc", "text", "resource-id", "class"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                control[key] = value.strip()

        if not control:
            continue
        if drop_container_controls and _is_container_like_control(control):
            continue

        signature = "|".join(control.get(key, "") for key in ("xpath", "content-desc", "text", "resource-id", "class"))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        compact_controls.append(control)

    return compact_controls


def _load_compact_controls_from_page_source(page_source: str | None) -> list[dict[str, str]]:
    if not page_source:
        return []

    try:
        payload = json.loads(page_source)
    except Exception:
        return _build_compact_ui_controls(page_source, drop_container_controls=False)

    if not isinstance(payload, list):
        return []

    controls: list[dict[str, str]] = []
    for item in payload:
        if not isinstance(item, dict):
            continue
        control: dict[str, str] = {}
        for key in ("xpath", "content-desc", "text", "resource-id", "class"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                control[key] = value.strip()
        if control:
            controls.append(control)
    return controls


def _serialize_compact_controls(controls: list[dict[str, str]]) -> str:
    return json.dumps(controls, ensure_ascii=False, separators=(",", ":"))


def _compact_controls_size_bytes(controls: list[dict[str, str]]) -> int:
    return len(_serialize_compact_controls(controls).encode("utf-8"))


def _shrink_compact_controls_to_byte_limit(
    controls: list[dict[str, str]],
    max_bytes: int,
) -> tuple[list[dict[str, str]], bool]:
    """按顺序保留控件，直到 compact JSON 达到字节上限。"""
    if max_bytes <= 0:
        return [], True

    kept: list[dict[str, str]] = []
    current_bytes = 2  # []
    truncated = False

    for item in controls:
        item_json = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        item_bytes = len(item_json.encode("utf-8"))
        separator_bytes = 1 if kept else 0
        if current_bytes + separator_bytes + item_bytes > max_bytes:
            truncated = True
            break
        kept.append(item)
        current_bytes += separator_bytes + item_bytes

    return kept, truncated


def _build_no_selector_inputs(
    phase_start_snapshot: dict[str, Any] | None,
    phase_end_snapshot: dict[str, Any] | None,
    max_bytes: int = NO_SELECTOR_INPUT_BYTE_LIMIT,
) -> tuple[str, str | None, str | None, int]:
    raw_page_source = _snapshot_page_source(phase_end_snapshot)
    image_settings = [
        (960, 55),
        (720, 45),
        (640, 40),
        (512, 35),
        (384, 30),
        (256, 25),
        (160, 20),
        (96, 10),
    ]

    start_path = (phase_start_snapshot or {}).get("screenshot_path")
    end_ui = (phase_end_snapshot or {}).get("ui") or {}
    end_scroll_paths = end_ui.get("scroll_screenshot_paths") if isinstance(end_ui, dict) else []
    end_path = None
    if isinstance(end_scroll_paths, list):
        for path in end_scroll_paths:
            if isinstance(path, str) and path.strip():
                end_path = path.strip()
                break
    if not end_path:
        end_path = (phase_end_snapshot or {}).get("screenshot_path")

    for max_dimension, jpeg_quality in image_settings:
        start_image = _encode_image_as_data_url(start_path, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
        end_image = _encode_image_as_data_url(end_path, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
        start_bytes = len(start_image.encode("utf-8")) if start_image else 0
        end_bytes = len(end_image.encode("utf-8")) if end_image else 0
        total_bytes = len(raw_page_source.encode("utf-8")) + start_bytes + end_bytes
        _debug_log(
            "No-selector budget probe (raw XML + screenshots): "
            f"xml={len(raw_page_source.encode('utf-8'))} bytes, "
            f"start_image={start_bytes} bytes, end_image={end_bytes} bytes, "
            f"total={_format_budget_usage(total_bytes, max_bytes)}"
        )
        if total_bytes <= max_bytes:
            return raw_page_source, start_image, end_image, total_bytes

    # 先尽可能压缩截图，若仍超限则压缩 XML 为控件摘要，优先保留关键字段。
    start_image = _encode_image_as_data_url(start_path, max_dimension=96, jpeg_quality=10)
    end_image = _encode_image_as_data_url(end_path, max_dimension=96, jpeg_quality=10)
    start_bytes = len(start_image.encode("utf-8")) if start_image else 0
    end_bytes = len(end_image.encode("utf-8")) if end_image else 0
    total_bytes = len(raw_page_source.encode("utf-8")) + start_bytes + end_bytes
    _debug_log(
        "No-selector minimum screenshot stage: "
        f"xml={len(raw_page_source.encode('utf-8'))} bytes, "
        f"start_image={start_bytes} bytes, end_image={end_bytes} bytes, "
        f"total={_format_budget_usage(total_bytes, max_bytes)}"
    )

    if total_bytes <= max_bytes:
        _debug_log(
            f"No-selector inputs prepared with screenshot compression only: start={bool(start_image)}, end={bool(end_image)}, total_bytes={total_bytes}."
        )
        return raw_page_source, start_image, end_image, total_bytes

    compact_controls = _build_compact_ui_controls(raw_page_source, drop_container_controls=False)
    compact_page_source = _serialize_compact_controls(compact_controls)
    compact_bytes = len(compact_page_source.encode("utf-8")) + start_bytes + end_bytes
    _debug_log(
        "No-selector compact XML stage: "
        f"controls={len(compact_controls)}, compact_xml={len(compact_page_source.encode('utf-8'))} bytes, "
        f"start_image={start_bytes} bytes, end_image={end_bytes} bytes, "
        f"total={_format_budget_usage(compact_bytes, max_bytes)}"
    )
    if compact_bytes <= max_bytes:
        _debug_log(
            f"No-selector inputs exceeded limit with raw XML; using compact control summary instead: controls={len(compact_controls)}, total_bytes={compact_bytes}."
        )
        return compact_page_source, start_image, end_image, compact_bytes

    compact_controls_no_container = _build_compact_ui_controls(raw_page_source, drop_container_controls=True)
    compact_page_source_no_container = _serialize_compact_controls(compact_controls_no_container)
    compact_no_container_bytes = len(compact_page_source_no_container.encode("utf-8")) + start_bytes + end_bytes
    _debug_log(
        "No-selector container-pruned stage: "
        f"controls={len(compact_controls_no_container)}, compact_xml={len(compact_page_source_no_container.encode('utf-8'))} bytes, "
        f"start_image={start_bytes} bytes, end_image={end_bytes} bytes, "
        f"total={_format_budget_usage(compact_no_container_bytes, max_bytes)}"
    )
    if compact_no_container_bytes <= max_bytes:
        _debug_log(
            f"No-selector inputs exceeded limit after compact summary; dropping container controls: controls={len(compact_controls_no_container)}, total_bytes={compact_no_container_bytes}."
        )
        return compact_page_source_no_container, start_image, end_image, compact_no_container_bytes

    _debug_log(
        f"[WARNING] No-selector inputs still exceed {max_bytes} bytes after compact XML and container pruning: {compact_no_container_bytes}."
    )
    return compact_page_source_no_container, start_image, end_image, compact_no_container_bytes


def _estimate_assertion_prompt_input_bytes(user_payload: dict[str, Any], image_urls: list[str | None]) -> int:
    payload_bytes = len(json.dumps(user_payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8"))
    image_bytes = sum(len(url.encode("utf-8")) for url in image_urls if url)
    return payload_bytes + image_bytes


def _compress_assertion_prompt_images(
    user_payload: dict[str, Any],
    start_image_url: str | None,
    end_image_url: str | None,
    max_bytes: int = ASSERTION_PROMPT_INPUT_BYTE_LIMIT,
) -> tuple[str | None, str | None, int]:
    """对断言生成阶段的截图做逐步压缩，尽量将总输入控制在字节上限内。"""
    compression_steps = [
        (960, 55),
        (720, 45),
        (640, 40),
        (512, 35),
        (384, 30),
        (256, 25),
        (160, 20),
        (96, 10),
    ]

    best_start = start_image_url
    best_end = end_image_url
    best_total = _estimate_assertion_prompt_input_bytes(user_payload, [best_start, best_end])

    for max_dimension, jpeg_quality in compression_steps:
        compressed_start = _compress_data_url_image(best_start, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
        compressed_end = _compress_data_url_image(best_end, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
        total_bytes = _estimate_assertion_prompt_input_bytes(user_payload, [compressed_start, compressed_end])
        _debug_log(
            "Assertion prompt budget probe: "
            f"dimension={max_dimension}, quality={jpeg_quality}, total={_format_budget_usage(total_bytes, max_bytes)}"
        )
        best_start = compressed_start
        best_end = compressed_end
        best_total = total_bytes
        if total_bytes <= max_bytes:
            return best_start, best_end, best_total

    _debug_log(
        f"[WARNING] Assertion prompt still exceeds {max_bytes} bytes after screenshot compression: {best_total}."
    )
    return best_start, best_end, best_total


def _default_model() -> str:
    return load_llm_settings("ASSERTION").model


def _load_client(base_url: str | None, api_key: str | None):
    client = make_openai_client(base_url, api_key)
    if client is None:
        _debug_log("Failed to initialize OpenAI-compatible client.")
    return client


def _normalize_name(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower().strip()
    value = re.sub(r"[._/\\-]+", " ", value)
    value = re.sub(r"[^a-z0-9\u4e00-\u9fff ]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _candidate_app_score(candidate_name: str, target_app_name: str, task_description: str) -> float:
    cand_norm = _normalize_name(candidate_name)
    target_norm = _normalize_name(target_app_name)
    desc_norm = _normalize_name(task_description)

    if not cand_norm:
        return 0.0

    score = 0.0
    if target_norm:
        if cand_norm == target_norm:
            score += 100.0
        elif target_norm in cand_norm and len(cand_norm) > len(target_norm):
            score += 35.0
        elif cand_norm in target_norm and len(target_norm) > len(cand_norm):
            score += 8.0

    if desc_norm:
        if cand_norm in desc_norm:
            score += 20.0

    cand_tokens = {token for token in cand_norm.split() if len(token) >= 3}
    desc_tokens = {token for token in desc_norm.split() if len(token) >= 3}
    score += len(cand_tokens & desc_tokens) * 12.0

    return score


def _iter_apk_sources(apk_root: Path) -> list[Path]:
    if not apk_root.exists():
        return []
    sources: list[Path] = []
    for path in apk_root.iterdir():
        if path.name.startswith('.'):
            continue
        if path.is_dir() or path.suffix.lower() == '.apk':
            sources.append(path)
    return sources


def _apk_root_dir() -> Path:
    repo_root = Path(__file__).resolve().parent
    preferred = repo_root / "apk"
    if preferred.exists():
        return preferred
    fallback = Path("G:/testcases/apk")
    if fallback.exists():
        return fallback
    return preferred


def _extract_target_app_name_from_task_description(task_description: str, client=None, model: str | None = None) -> str | None:
    text = task_description or ""
    normalized = _normalize_name(text)

    regex_patterns = [
        r"redirect(?:s|ed|ing)? to (?:the )?(?P<app>[a-z0-9 ./_\-]+?) app\b",
        r"navigate(?:s|d|ing)? to (?:the )?(?P<app>[a-z0-9 ./_\-]+?) app\b",
        r"open(?:s|ed|ing)? (?:the )?(?P<app>[a-z0-9 ./_\-]+?) app\b",
        r"check whether it redirects to (?:the )?(?P<app>[a-z0-9 ./_\-]+?) app\b",
        r"go to (?:the )?(?P<app>[a-z0-9 ./_\-]+?) app\b",
        r"open in (?:the )?(?P<app>[a-z0-9 ./_\-]+?)(?: app| browser)?\b",
    ]
    for pattern in regex_patterns:
        match = re.search(pattern, normalized)
        if match:
            candidate = match.group("app").strip()
            candidate = re.sub(r"\bapp\b$", "", candidate).strip()
            if candidate:
                return candidate

    heuristic_aliases = {
        "phone": ["phone", "dialer", "telephone"],
        "firefox": ["firefox", "mozilla"],
        "brave": ["brave", "browser"],
        "booking": ["booking", "booking.com"],
        "google tasks": ["google tasks", "googletask", "google task"],
        "k9mail": ["k9mail", "k-9 mail", "k9-mail", "k9 mail"],
        "deskclock": ["deskclock", "clock", "alarm clock"],
        "calendar": ["calendar"],
        "youtube": ["youtube"],
        "spotify": ["spotify"],
        "signal": ["signal"],
        "trello": ["trello"],
        "tumblr": ["tumblr"],
        "wikipedia": ["wikipedia"],
        "yelp": ["yelp"],
        "yahnac": ["yahnac"],
        "mail": ["mail"],
        "element": ["element"],
        "cnn": ["cnn"],
        "foxnews": ["foxnews", "fox news"],
        "geek": ["geek"],
        "calculator": ["calculator", "tip calculator", "tip caluculator"],
    }
    normalized_aliases: list[tuple[str, str]] = []
    for app_name, aliases in heuristic_aliases.items():
        for alias in aliases:
            alias_norm = _normalize_name(alias)
            if alias_norm:
                normalized_aliases.append((app_name, alias_norm))

    for app_name, alias_norm in sorted(normalized_aliases, key=lambda item: len(item[1]), reverse=True):
        if alias_norm in normalized:
            return app_name

    if client is not None:
        prompt = (
            "Extract the target Android app name mentioned in the following TaskDescription. "
            "Return only the app name, with no extra words. "
            "Examples: 'redirects to the Phone app' -> 'Phone'; 'open blog in Firefox' -> 'Firefox'.\n\n"
            f"TaskDescription: {task_description}"
        )
        try:
            response = client.chat.completions.create(
                model=model or _default_model(),
                messages=[
                    {"role": "system", "content": "You extract Android app names from task descriptions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
            )
            content = (response.choices[0].message.content or "").strip()
            content = re.sub(r"^[\"'`]+|[\"'`]+$", "", content).strip()
            content = re.sub(r"\bapp\b$", "", content, flags=re.IGNORECASE).strip()
            if content:
                return content
        except Exception as exc:
            _debug_log(f"Failed to extract app name via LLM: {exc}")

    return None


def _choose_best_apk_source(task_description: str, target_app_name: str | None) -> Path | None:
    apk_root = _apk_root_dir()
    sources = _iter_apk_sources(apk_root)
    if not sources:
        return None

    normalized_target = _normalize_name(target_app_name)
    if normalized_target:
        exact_matches = [
            source
            for source in sources
            if _normalize_name(source.stem if source.is_file() else source.name) == normalized_target
        ]
        if exact_matches:
            exact_matches.sort(key=lambda path: (0 if path.suffix.lower() == ".apk" else 1, len(path.name), path.name.lower()))
            return exact_matches[0]

    best: tuple[float, Path] | None = None
    for source in sources:
        candidate_name = source.stem if source.is_file() else source.name
        score = _candidate_app_score(candidate_name, target_app_name or "", task_description)
        if best is None or score > best[0]:
            best = (score, source)

    return best[1] if best else None


def _choose_apk_file_from_source(source: Path) -> Path | None:
    if source.is_file() and source.suffix.lower() == '.apk':
        return source

    if not source.is_dir():
        return None

    apk_files = sorted(
        [path for path in source.rglob('*.apk') if path.is_file()],
        key=lambda path: (
            0 if path.name.lower().startswith('base') else 1,
            len(path.name),
            path.name.lower(),
        ),
    )
    if not apk_files:
        return None
    return apk_files[0]


def _extract_package_name_from_apk(apk_file: Path | None) -> str | None:
    if apk_file is None or APK is None:
        return None

    try:
        apk = APK(str(apk_file))
        package_name = apk.get_package()
        if package_name:
            return package_name
    except Exception as exc:
        _debug_log(f"Failed to parse package name from APK {apk_file}: {exc}")
    return None


def _resolve_package_name_from_task_description(task_description: str, client=None, model: str | None = None) -> str | None:
    target_app_name = _extract_target_app_name_from_task_description(task_description, client=client, model=model)
    if not target_app_name:
        _debug_log("Could not extract target app name from task description.")
        return None

    source = _choose_best_apk_source(task_description, target_app_name)
    if source is None:
        _debug_log(f"Could not locate APK source for target app: {target_app_name}")
        return None

    apk_file = _choose_apk_file_from_source(source)
    if apk_file is None:
        _debug_log(f"Could not locate APK file under source: {source}")
        return None

    package_name = _extract_package_name_from_apk(apk_file)
    if not package_name:
        _debug_log(f"Could not extract package name from APK file: {apk_file}")
        return None

    _debug_log(
        f"Resolved target app name to package: target_app_name={target_app_name!r}, source={source}, apk_file={apk_file}, package={package_name!r}"
    )
    return package_name


def _build_assertion_argument_hints(context: dict[str, Any], allowed_assertions: list[str]) -> dict[str, str]:
    current_step = context.get("current_step") or {}
    candidate_elements = current_step.get("candidate_elements") or []

    xpath_candidates: list[str] = []
    text_candidates: list[str] = []
    for candidate in candidate_elements:
        if not isinstance(candidate, dict):
            continue
        xpath = candidate.get("xpath_by_resource_id") or candidate.get("xpath_by_class") or candidate.get("xpath")
        if xpath:
            xpath_candidates.append(str(xpath))
        text_value = candidate.get("text") or candidate.get("content_desc")
        if text_value:
            text_candidates.append(str(text_value))

    hints: dict[str, str] = {}
    if "check_dark_mode" in allowed_assertions:
        hints["check_dark_mode"] = (
            "Fill the parentheses with a container control identifier, not a leaf text control. "
            "Prefer a large parent/container node from current_step.candidate_elements, such as android.widget.FrameLayout."
        )
    if "check_light_mode" in allowed_assertions:
        hints["check_light_mode"] = (
            "Fill the parentheses with a container control identifier, not a leaf text control. "
            "Prefer a large parent/container node from current_step.candidate_elements, such as android.widget.FrameLayout."
        )
    if "check_elements_same_screen" in allowed_assertions:
        hints["check_elements_same_screen"] = (
            "Fill the second argument with a Python list of XPath strings for the elements that must appear on the same screen. "
            f"Candidate XPaths: {xpath_candidates[:6]}"
        )
    if "check_row_checked_by_text" in allowed_assertions:
        hints["check_row_checked_by_text"] = (
            "Fill the second argument with the exact row text string whose checked switch/checkbox should be verified. "
            f"Candidate texts: {text_candidates[:6]}"
        )
    return hints


def _normalize_generation_output(raw_text: str) -> AssertionGenerationResult:
    text = (raw_text or "").strip()
    if not text:
        return AssertionGenerationResult(False, None, "empty response")

    if text.lower() in {"no", "false", "skip", "none", "null", "do not insert"}:
        return AssertionGenerationResult(False, None, text)

    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        payload = json.loads(text)
    except Exception:
        return AssertionGenerationResult(True, text, "plain text assertion")

    should_insert = bool(payload.get("should_insert", True)) if isinstance(payload, dict) else True
    assertion = payload.get("assertion") if isinstance(payload, dict) else None
    reason = str(payload.get("reason", "ok")) if isinstance(payload, dict) else "ok"
    candidate = payload.get("candidate") if isinstance(payload, dict) else None
    return AssertionGenerationResult(should_insert, assertion if isinstance(assertion, str) else None, reason, candidate if isinstance(candidate, dict) else None)


def _build_prompt(context: dict[str, Any], allowed_assertions: list[str]) -> list[dict[str, str]]:
    assertion_argument_hints = _build_assertion_argument_hints(context, allowed_assertions)
    system = (
        "You are an assertion generator for Android UI test cases. Decide whether to insert an assertion after the current action step. "
        "If insertion is needed, generate ONE valid Python assertion line. Return JSON with keys should_insert, assertion, reason, candidate."
    )
    user = {
        "context": context,
        "allowed_assertions": allowed_assertions,
        "assertion_argument_hints": assertion_argument_hints,
        "output_rules": [
            "Return JSON with keys: should_insert, assertion, reason, candidate.",
            "If no assertion is needed, set should_insert to false and assertion to null.",
            "Do not include markdown fences.",
        ],
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, indent=2)},
    ]


def _extract_verification_focus(phase_intent: str) -> str:
    """只从阶段意图提取本阶段要验证的目标，不看整条任务描述。"""
    text = _normalize_text(phase_intent)
    if not text:
        return ""

    patterns = [
        r"(?:check|verify)(?: whether| if)?(?: there (?:is|are))?(?: a| an| the)? (?P<focus>.+?)(?: appears?| appear| is displayed| on the interface| on the page| as expected|\.|$)",
        r"(?:finally )?(?:check|verify)(?: whether| if)? (?P<focus>.+?)(?: appears?| appear| is displayed| on the interface| on the page| as expected|\.|$)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            focus = match.group("focus").strip()
            focus = re.sub(r"^(there is|there are|a|an|the)\s+", "", focus, flags=re.IGNORECASE).strip()
            focus = re.sub(r"\s+", " ", focus)
            if focus:
                return focus[:120]
    return phase_intent[:120]


def _encode_image_as_data_url(image_path: str | None, max_dimension: int = 960, jpeg_quality: int = 55) -> str | None:
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_file():
        return None

    try:
        image_bytes = path.read_bytes()
        mime_type = "image/png"
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(image_bytes))
            if max_dimension > 0 and (img.width > max_dimension or img.height > max_dimension):
                img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
            if img.mode in ("RGBA", "LA", "P"):
                img = img.convert("RGB")
            buffer = io.BytesIO()
            img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
            image_bytes = buffer.getvalue()
            mime_type = "image/jpeg"
        except Exception:
            pass

        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except Exception as exc:
        _debug_log(f"Failed to encode image {image_path!r}: {exc}")
        return None


def _encode_data_url_from_image_bytes(image_bytes: bytes, max_dimension: int = 960, jpeg_quality: int = 55) -> str | None:
    if not image_bytes:
        return None

    mime_type = "image/png"
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes))
        if max_dimension > 0 and (img.width > max_dimension or img.height > max_dimension):
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        image_bytes = buffer.getvalue()
        mime_type = "image/jpeg"
    except Exception:
        pass

    try:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"
    except Exception:
        return None


def _compress_data_url_image(
    data_url: str | None,
    max_dimension: int,
    jpeg_quality: int,
) -> str | None:
    if not data_url:
        return None
    if not data_url.startswith("data:"):
        return data_url

    try:
        _, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
    except Exception:
        return data_url

    compressed = _encode_data_url_from_image_bytes(image_bytes, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
    return compressed or data_url


def _snapshot_to_payload(snapshot: dict[str, Any] | None, label: str) -> dict[str, Any]:
    if not snapshot:
        return {"label": label, "available": False}

    payload = {
        "label": label,
        "available": True,
        "step_index": snapshot.get("step_index"),
        "step_label": snapshot.get("step_label"),
        "step_type": snapshot.get("step_type"),
        "current_package": snapshot.get("current_package"),
        "current_activity": snapshot.get("current_activity"),
        "ui_summary": _extract_ui_summary(snapshot),
    }
    image_url = _encode_image_as_data_url(snapshot.get("screenshot_path"))
    if image_url:
        payload["image_data_url"] = image_url
    return payload


# ================== 增强的 _parse_assertion_list ==================
def _parse_assertion_list(raw_text: str) -> PhaseAssertionResult:
    """解析LLM返回的文本，提取断言列表和selected_targets。
    增强点：
    - 优先解析JSON，提取assertions和selected_targets。
    - 若JSON解析失败，尝试用正则提取"assertions"数组。
    - 若仍失败，回退到按行拆分，但仅保留形如断言的行。
    """
    text = (raw_text or "").strip()
    if not text:
        return PhaseAssertionResult([], "empty response", raw_text)

    # 1. 去除markdown代码块
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    # 辅助：去掉首尾引号（如果整个文本被引号包裹）
    if len(text) >= 2 and text[0] == text[-1] and text[0] in {'"', "'"}:
        try:
            import ast
            unwrapped = ast.literal_eval(text)
            if isinstance(unwrapped, str) and unwrapped.strip():
                text = unwrapped.strip()
        except Exception:
            pass

    # 2. 尝试完整JSON解析
    try:
        payload = json.loads(text)
        if isinstance(payload, str):
            # 如果解析结果为字符串，尝试再次解析（可能嵌套）
            nested = payload.strip()
            if nested:
                try:
                    payload = json.loads(nested)
                except Exception:
                    pass

        # 提取assertions
        assertions = []
        selected_targets = []
        insertion_step = None
        if isinstance(payload, dict):
            assertions = payload.get("assertions", [])
            if isinstance(assertions, str):
                # 如果assertions是字符串，尝试解析为列表
                try:
                    parsed_assertions = json.loads(assertions)
                    if isinstance(parsed_assertions, list):
                        assertions = parsed_assertions
                except Exception:
                    pass
            elif not isinstance(assertions, list):
                assertions = []

            selected_targets = payload.get("selected_targets", [])
            if not isinstance(selected_targets, list):
                selected_targets = []
            insertion_step = payload.get("insert_after_step", payload.get("insertion_step"))
        elif isinstance(payload, list):
            # 如果整个响应就是一个列表，视为断言列表
            assertions = payload

        # 标准化断言列表
        normalized_assertions = [str(item).strip() for item in assertions if str(item).strip()]
        # 标准化selected_targets
        normalized_targets: list[dict[str, str]] = []
        for item in selected_targets:
            if not isinstance(item, dict):
                continue
            target_item: dict[str, str] = {}
            for key in ("resource-id", "text", "content-desc", "class", "xpath", "reason"):
                value = item.get(key)
                if isinstance(value, str) and value.strip():
                    target_item[key] = value.strip()
            if target_item:
                normalized_targets.append(target_item)

        if normalized_assertions:
            if isinstance(insertion_step, str) and insertion_step.strip().isdigit():
                insertion_step = int(insertion_step.strip())
            elif not isinstance(insertion_step, int):
                insertion_step = None
            return PhaseAssertionResult(
                normalized_assertions,
                "ok",
                raw_text,
                normalized_targets,
                0,
                insertion_step,
            )
    except Exception:
        pass

    # 3. 尝试用正则提取 "assertions": [...]
    # 匹配 "assertions": [ ... ] 中的数组内容
    array_match = re.search(r'"assertions"\s*:\s*(\[.*?\])', text, re.DOTALL)
    if array_match:
        array_text = array_match.group(1)
        try:
            array_payload = json.loads(array_text)
            if isinstance(array_payload, list):
                assertions = [str(item).strip() for item in array_payload if str(item).strip()]
                if assertions:
                    # 尝试提取selected_targets
                    targets_match = re.search(r'"selected_targets"\s*:\s*(\[.*?\])', text, re.DOTALL)
                    targets = []
                    if targets_match:
                        try:
                            targets_payload = json.loads(targets_match.group(1))
                            if isinstance(targets_payload, list):
                                targets = targets_payload
                        except Exception:
                            pass
                    normalized_targets = []
                    for item in targets:
                        if not isinstance(item, dict):
                            continue
                        target_item = {}
                        for key in ("resource-id", "text", "content-desc", "class", "xpath", "reason"):
                            value = item.get(key)
                            if isinstance(value, str) and value.strip():
                                target_item[key] = value.strip()
                        if target_item:
                            normalized_targets.append(target_item)
                    return PhaseAssertionResult(assertions, "ok", raw_text, normalized_targets)
        except Exception:
            pass

    # 4. 回退：按行拆分，但只保留像断言的行（包含driver.find_element或check_等）
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    # 先尝试移除列表符号
    lines = [line[1:].strip() if line.startswith("-") else line for line in lines]
    # 过滤：只保留看起来像断言的行
    assertion_pattern = re.compile(
        r'(driver\.find_element|driver\.find_elements|check_|assert\s+)\s*[\(=]',
        re.IGNORECASE
    )
    filtered = [line for line in lines if assertion_pattern.search(line)]
    if filtered:
        return PhaseAssertionResult(filtered, "recovered from filtered lines", raw_text, None)

    # 如果还是没有，返回空
    return PhaseAssertionResult([], "no assertions found", raw_text, None)
# ================== 增强的 _parse_assertion_list 结束 ==================


def _fill_current_app_package_assertion(assertion: str, current_package: str | None) -> str:
    normalized = assertion.strip()
    if not normalized.startswith("check_current_app_package"):
        return assertion
    if not current_package:
        return assertion
    return f"check_current_app_package('{current_package}')"


def _normalize_assertion_quotes(assertion: str) -> str:
    normalized = assertion.strip()
    if not normalized:
        return assertion

    def _escape_xpath_predicate_quotes(xpath: str) -> str:
        def _replace_predicate(match: re.Match[str]) -> str:
            attr_name = match.group("attr")
            quote = match.group("quote")
            value = match.group("value")

            if quote == "'" and "'" in value:
                escaped_value = value.replace("'", r"\'")
                return f"@{attr_name}='{escaped_value}'"
            if quote == '"' and "'" in value and '"' not in value:
                escaped_value = value.replace("'", r"\'")
                return f"@{attr_name}='{escaped_value}'"
            if quote == '"' and '"' in value:
                escaped_value = value.replace('"', r'\"')
                return f'@{attr_name}="{escaped_value}"'
            return match.group(0)

        return re.sub(
            r"@(?P<attr>text|content-desc|resource-id)\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)",
            _replace_predicate,
            xpath,
        )

    def _escape_xpath_in_assertion(text: str) -> str:
        xpath_match = re.search(r"(By\.XPATH\s*,\s*)(['\"])(.*?)(\2)", text)
        if not xpath_match:
            return text

        prefix = xpath_match.group(1)
        outer_quote = xpath_match.group(2)
        xpath_value = xpath_match.group(3)
        suffix_start = xpath_match.end(4)
        escaped_xpath = _escape_xpath_predicate_quotes(xpath_value)
        if escaped_xpath == xpath_value:
            return text
        return text[:xpath_match.start()] + f"{prefix}{outer_quote}{escaped_xpath}{outer_quote}" + text[suffix_start:]

    normalized = _escape_xpath_in_assertion(normalized)

    def _replace_text_comparison(match: re.Match[str]) -> str:
        prefix = match.group("prefix")
        value = match.group("value")
        if "'" not in value:
            return match.group(0)
        escaped = value.replace('"', r'\"')
        return f'{prefix}"{escaped}"'

    normalized = re.sub(
        r'(?P<prefix>\bassert\s+.+?==\s*)(?P<quote>["\'])(?P<value>.+)(?P=quote)(?P<suffix>\s*)$',
        _replace_text_comparison,
        normalized,
    )

    normalized = _escape_xpath_in_assertion(normalized)
    return normalized


def generate_phase_assertions(
    phase_info: dict[str, Any],
    phase_start_snapshot: dict[str, Any] | None,
    phase_end_snapshot: dict[str, Any] | None,
    allowed_assertions: list[str],
    task_description: str | None = None,
    use_control_selector: bool = True,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
    feedback: str | None = None,
) -> PhaseAssertionResult:
    client = _load_client(base_url, api_key)
    if client is None:
        return PhaseAssertionResult([], "LLM unavailable", "", None, 0)

    phase_intent = str(phase_info.get("intent", "")).strip()
    verification_focus = _extract_verification_focus(phase_intent)
    start_payload = _snapshot_to_payload(phase_start_snapshot, "phase_start")
    end_payload = _snapshot_to_payload(phase_end_snapshot, "phase_end")
    candidate_insertion_steps = phase_info.get("candidate_insertion_steps") or []
    candidate_insertion_steps = [item for item in candidate_insertion_steps if isinstance(item, dict)]

    if use_control_selector:
        # ========== 默认路径：保留控件筛选器，只取最相关的少量候选控件 ==========
        end_related_ui, control_token_usage = _extract_related_ui_summary(
            phase_end_snapshot, verification_focus or phase_intent, limit=5
        )
        candidate_source_note = "filtered by control selector"
        candidate_limit_note = "at most five candidate controls"
    else:
        # ========== 消融路径：不经过控件筛选器，直接使用当前阶段最后一步的原始 XML + 起始/结束截图 ==========
        end_page_source, start_image, end_image, total_bytes = _build_no_selector_inputs(
            phase_start_snapshot,
            phase_end_snapshot,
            NO_SELECTOR_INPUT_BYTE_LIMIT,
        )
        if total_bytes > NO_SELECTOR_INPUT_BYTE_LIMIT:
            _debug_log(
                f"No-selector inputs exceeded {NO_SELECTOR_INPUT_BYTE_LIMIT} bytes after compression attempts: {total_bytes}."
            )
        else:
            _debug_log(
                f"No-selector inputs compression completed successfully: {total_bytes} bytes <= {NO_SELECTOR_INPUT_BYTE_LIMIT} bytes."
            )
        end_related_ui = _load_compact_controls_from_page_source(end_page_source)
        start_payload["image_data_url"] = start_image
        end_payload["image_data_url"] = end_image
        end_payload["page_source"] = end_page_source
        end_payload["page_source_format"] = "compact_controls_json" if end_page_source.startswith("[") else "raw_xml"
        control_token_usage = 0
        candidate_source_note = "final-step snapshot with compact XML control summary and start/end screenshots"
        candidate_limit_note = "compact controls derived from the final step snapshot"
    # ====================================================

    start_payload["ui_summary"] = []
    end_payload["ui_summary"] = end_related_ui

    resolved_current_package = _resolve_package_name_from_task_description(task_description or "", client=client, model=model) if task_description else None
    if resolved_current_package:
        _debug_log(f"Resolved current package from task description: {resolved_current_package!r}")
    else:
        _debug_log("Could not resolve current package from task description.")

    system_prompt = (
        "You are an Android assertion generator. Use only the phase intent, start screenshot, end screenshot, and end-screen UI attributes. "
        "Generate ONLY assertions from the allowed list in the user message. Do not invent new functions or custom syntax. "
        "Use UI evidence for text/content-desc/resource-id/xpath whenever possible, and keep assertions tightly centered on the phase intent. "
        "For resource-id and any xpath @resource-id fragments, you must use the exact real UI value from the UI summary; never fabricate or create them from intent. For example, if the intent says 'check the search list', the real UI resource-id might be 'sear_rl_list', but you create a wrong assertion with resource-id 'search_list' or 'list_item_container'. This is not allowed. Always use the real UI evidence. "
        "Important: When you generate assertions, prefer using xpath to find the target control. If xpath is not available, use resource-id. "
        "Important quoting rule: if the assertion string value contains an apostrophe (for example Can't), you MUST use double quotes around that string literal, not single quotes. "
        "You will be given at most five candidate controls from the real UI. Choose target detection controls only from those five controls. "
        "Critical rule: if the intent is to check that a control exists, and you don't see that control in the UI summary, you STILL must verify its presence, for example `assert elem.is_displayed() or  assert elem.text == 'Bookmarks'`. Do NOT verify its absence. "
        "Critical rule: if the intent is to check that a control does not exist, you MUST verify absence with `assert len(elems) == 0`. Do NOT return `assert elem.is_displayed() is False`. "
        "Critical rule: After you generate assertions, first check the target control's resource-id in your assertion is exactly the same as the resource-id in the UI summary. If it is not, you must fix it to match the real UI evidence. "
        "If you can meet the intent with one assertion, you do not need to generate assertions for other controls that also meet the intent. "
        "Return ONLY JSON with keys `assertions` and `selected_targets`."
    )

    user_payload = {
        "phase_intent": phase_intent,
        "phase_info": {
            "phase_id": phase_info.get("phase_id"),
            "action_range": phase_info.get("action_range"),
            "intent": phase_intent,
        },
        "allowed_assertions": allowed_assertions,
        "allowed_assertion_types": allowed_assertions,
        "selected_target_requirement": (
            "Include selected_targets for the real UI controls used to derive each assertion. Use exact UI evidence when available; never fabricate resource-id or xpath values."
        ),
        "phase_start_snapshot": {
            "step_index": start_payload.get("step_index"),
            "step_label": start_payload.get("step_label"),
            "step_type": start_payload.get("step_type"),
            "current_package": start_payload.get("current_package"),
            "current_activity": start_payload.get("current_activity"),
        },
        "phase_end_snapshot": {
            "step_index": end_payload.get("step_index"),
            "step_label": end_payload.get("step_label"),
            "step_type": end_payload.get("step_type"),
            "current_package": end_payload.get("current_package"),
            "current_activity": end_payload.get("current_activity"),
            "page_source": end_payload.get("page_source"),
            "page_source_format": end_payload.get("page_source_format"),
            "ui_summary": end_related_ui,
            "ui_summary_pretty": _format_ui_summary_for_prompt(end_related_ui),
        },
        "candidate_source": candidate_source_note,
        "candidate_scope": candidate_limit_note,
        "candidate_insertion_steps": candidate_insertion_steps,
        "scope_rule": (
            "Keep assertions directly related to the phase intent. You may only choose target controls from the five candidate controls in phase_end_snapshot.ui_summary. "
            "Prefer exact text/content-desc from the UI summary when it matches the target. Use observed UI evidence for resource-id and xpath fragments. "
            "Important: If a target text contains an apostrophe such as Can't, wrap that string in double quotes in the assertion. Do not use single quotes for text values that contain apostrophes. "
            "Important: The text attribute of the target control in your generated assertion must be written as '@text = ....', not as 'text() = ....'.the same applies to content-desc: use '@content-desc = ...', not 'content-desc() = ...'. "
            "If multiple object detection controls meet the same inspection requirement and can generate assertions, selecting one of them to generate an assertion is sufficient."
            "If the intent and the chosen control conflict only a little, use the actual control information exactly as provided. "
            "If the intent and the chosen control conflict a lot, you may adjust text/content-desc to match the intent, but resource-id must always remain the exact real UI value from the candidate control. "
            "If the intent is about checking that a control exists, but you don't see the control in the UI summary, you still have to verify its presence, like `assert elem.is_displayed()`. Don't verify its absence!!! "
            "If the intent is about checking that a control not exists, you must insert an assertion like `assert len(elems) == 0`. "
            "If the intent is about checking that a control is checked or selected, you must insert an assertion like `check_row_checked_by_text('row text')`. "
            "check_elements_same_screen must be used with a list of xpath strings for the elements that must appear on the same screen. For example: check_elements_same_screen(driver, ['xpath1', 'xpath2']). "
            "check_dark_mode and check_light_mode must be used with a container control identifier, such as check_dark_mode(android.widget.FrameLayout). "
            "If candidate_insertion_steps is provided, choose the most suitable insert_after_step from those candidates and return it in JSON. "
            "If you think the best insertion point is the final action step, return that step number. "
            "If unsure, return selected_targets with the actual control evidence rather than hallucinating a locator. Never invent an assertion function outside the allowed_assertions list."
        ),
        "feedback": feedback or "",
    }

    start_image_url = start_payload.get("image_data_url")
    end_image_url = end_payload.get("image_data_url")
    if use_control_selector:
        start_image_url, end_image_url, total_input_bytes = _compress_assertion_prompt_images(
            user_payload,
            start_image_url,
            end_image_url,
            ASSERTION_PROMPT_INPUT_BYTE_LIMIT,
        )
        if total_input_bytes <= ASSERTION_PROMPT_INPUT_BYTE_LIMIT:
            _debug_log(
                f"Assertion prompt inputs compressed successfully for control-selector path: {total_input_bytes} bytes <= {ASSERTION_PROMPT_INPUT_BYTE_LIMIT} bytes."
            )
        else:
            _debug_log(
                f"[WARNING] Assertion prompt inputs for control-selector path still exceed {ASSERTION_PROMPT_INPUT_BYTE_LIMIT} bytes: {total_input_bytes}."
            )

    if feedback:
        user_payload["error_feedback"] = feedback

    user_text = [json.dumps(user_payload, ensure_ascii=False, indent=2)]
    if feedback:
        user_text.append("\n错误反馈（请避免重复这些问题）：\n" + feedback)

    user_content: list[dict[str, Any]] = [{"type": "text", "text": "\n".join(user_text)}]
    if start_image_url:
        user_content.append({"type": "text", "text": "Phase start screenshot:"})
        user_content.append({"type": "image_url", "image_url": {"url": start_image_url}})
    if end_image_url:
        user_content.append({"type": "text", "text": "Phase end screenshot:"})
        user_content.append({"type": "image_url", "image_url": {"url": end_image_url}})

    response = get_openai_chat_completion(
        client,
        model=model or _default_model(),
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
        temperature=0.2,
    )
    raw_content = response.choices[0].message.content or ""
    main_token_usage = extract_openai_token_usage(response)
    total_token_usage = main_token_usage + control_token_usage
    _debug_log(f"Phase assertion raw response: {raw_content!r}")
    _debug_log(f"Token usage: main={main_token_usage}, control={control_token_usage}, total={total_token_usage}")

    parsed = _parse_assertion_list(raw_content)
    assertions: list[str] = []
    for assertion in parsed.assertions:
        filled = _fill_current_app_package_assertion(assertion, resolved_current_package)
        filled = _normalize_assertion_quotes(filled)
        assertions.append(filled)

    return PhaseAssertionResult(
        assertions,
        parsed.reason,
        parsed.raw_response,
        parsed.selected_targets,
        total_token_usage,
        parsed.insert_after_step,
    )


def generate_assertion(
    task_description: str,
    category: int,
    all_action_descriptions: list[str],
    current_snapshot: dict[str, Any],
    previous_snapshot: dict[str, Any] | None,
    allowed_assertions: list[str],
    excluded_targets: list[str] | None = None,
    generated_assertions: list[str] | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> AssertionGenerationResult:
    """Deprecated compatibility wrapper for legacy single-assertion flow."""
    _debug_log(
        "generate_assertion is deprecated; prefer generate_phase_assertions. "
        f"category={category}, action_count={len(all_action_descriptions)}, excluded_targets={excluded_targets or []}"
    )

    context = {
        "task_description": task_description,
        "category": category,
        "all_action_descriptions": all_action_descriptions,
        "current_snapshot": current_snapshot,
        "previous_snapshot": previous_snapshot,
        "allowed_assertions": allowed_assertions,
        "excluded_targets": excluded_targets or [],
        "generated_assertions": generated_assertions or [],
    }
    prompt = _build_prompt(context, allowed_assertions)
    client = _load_client(base_url, api_key)
    if client is None:
        return AssertionGenerationResult(False, None, "LLM unavailable", None, 0)

    try:
        response = get_openai_chat_completion(
            client,
            model=model or _default_model(),
            messages=prompt,
            temperature=0.2,
        )
        content = response.choices[0].message.content or ""
        token_usage = extract_openai_token_usage(response)
        normalized = _normalize_generation_output(content)
        return AssertionGenerationResult(
            normalized.should_insert,
            normalized.assertion,
            normalized.reason,
            normalized.candidate,
            token_usage,
        )
    except Exception as exc:
        _debug_log(f"Deprecated generate_assertion failed: {exc}")
        return AssertionGenerationResult(False, None, f"failed: {exc}", None, 0)