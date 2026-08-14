from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from llm_client import extract_openai_token_usage, get_openai_chat_completion, load_llm_settings, make_openai_client


CATEGORY_LABELS = {
    1: "Simple Functional Check Assertion",
    2: "UI Hierarchy-based Assertions",
    3: "Interaction / Logic Assertions",
    4: "Nullity/Exception Assertions",
}

_VERIFICATION_INTENT_PREFIXES = ("check", "verify", "confirm")
_BANNED_INTENT_WORDS = {"click", "tap", "press", "clicking", "tapping", "pressing", "clicks", "taps", "presses","clicked", "tapped", "pressed"}
_BANNED_INTENT_PATTERNS = (
    r"\bclickable\b",
    r"\bis clickable\b",
    r"\bcan be clicked\b",
    r"\bto be clicked\b",
    r"\bwhether .* clickable\b",
)


@dataclass(frozen=True)
class PhasePlanResult:
    phases: list[dict[str, Any]]
    reason: str
    raw_response: str
    token_usage: int = 0


def _debug_log(message: str):
    print(f"[INTENT_PLANNER_DEBUG] {message}")


def _default_model() -> str:
    return load_llm_settings("ASSERTION").model


def _category_label(category: int | None) -> str:
    if category in CATEGORY_LABELS:
        return CATEGORY_LABELS[category]
    return "Unknown Category"


def _normalize_text(value: str | None) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _find_unquoted_period(text: str) -> int:
    """Find the first English period that is outside single/double quotes."""
    in_single_quote = False
    in_double_quote = False
    escape_next = False

    for index, char in enumerate(text):
        if escape_next:
            escape_next = False
            continue

        if char == "\\":
            escape_next = True
            continue

        if char == "'" and not in_double_quote:
            prev_char = text[index - 1] if index > 0 else ""
            next_char = text[index + 1] if index + 1 < len(text) else ""
            if prev_char.isalnum() and next_char.isalnum():
                continue
            in_single_quote = not in_single_quote
            continue

        if char == '"' and not in_single_quote:
            in_double_quote = not in_double_quote
            continue

        if char == "." and not in_single_quote and not in_double_quote:
            return index

    return -1


def _extract_verification_clauses(task_description: str) -> list[str]:
    text = _normalize_text(task_description)
    if not text:
        return []

    clauses: list[str] = []
    pattern = re.compile(r"\b(check|verify)\b", re.IGNORECASE)
    for match in pattern.finditer(text):
        remainder = text[match.end():]
        period_index = _find_unquoted_period(remainder)
        clause = remainder[:period_index] if period_index >= 0 else remainder
        clause = clause.strip()
        clause = re.sub(r"^[\s:\-–—,;]+", "", clause).strip()
        if clause:
            clauses.append(clause)
    return clauses


def _validate_phase_intent(intent: str) -> tuple[bool, list[str]]:
    text = _normalize_text(intent)
    lower = text.lower()
    errors: list[str] = []
    starts_with_verification_prefix = any(
        lower.startswith(prefix) for prefix in _VERIFICATION_INTENT_PREFIXES
    )

    if not text:
        return False, ["intent 为空"]

    if not starts_with_verification_prefix:
        errors.append("intent 必须以 check/verify/confirm 开头")
    if not re.search(r"\b(if|whether)\b", lower):
        errors.append("intent 需要包含 if 或 whether")

    words = set(re.findall(r"[a-z]+", lower))
    banned_words = sorted(words & _BANNED_INTENT_WORDS)
    if banned_words:
        errors.append(f"intent 不能包含点击动作词: {', '.join(banned_words)}")

    if any(re.search(pattern, lower) for pattern in _BANNED_INTENT_PATTERNS):
        errors.append("intent 不能描述控件是否可点击，只能描述存在/跳转/已点击/模式/多控件/不存在等检查结果")

    return len(errors) == 0, errors


def _build_fallback_phases(task_description: str, action_count: int) -> list[dict[str, Any]]:
    clauses = _extract_verification_clauses(task_description)
    if not clauses:
        return [{
            "phase_id": 1,
            "action_range": [1, max(action_count, 1)],
            "intent": _normalize_text(task_description)[:80],
        }]

    phase_count = len(clauses)
    phases: list[dict[str, Any]] = []
    base = action_count // phase_count
    remainder = action_count % phase_count
    start = 1
    for index, clause in enumerate(clauses, 1):
        length = base + (1 if index <= remainder else 0)
        end = start + max(length - 1, 0)
        if index == phase_count:
            end = action_count
        phases.append({
            "phase_id": index,
            "action_range": [start, max(start, min(end, action_count))],
            "intent": clause[:120],
        })
        start = end + 1
    return phases


def _fallback_from_task_description(task_description: str, action_count: int) -> list[dict[str, Any]]:
    text = _normalize_text(task_description)
    clauses = _extract_verification_clauses(text)
    if clauses:
        return _build_fallback_phases(task_description, action_count)
    return [{
        "phase_id": 1,
        "action_range": [1, max(action_count, 1)],
        "intent": text[:120],
    }]


def _normalize_phase_count(phases: list[dict[str, Any]], task_description: str, action_count: int) -> list[dict[str, Any]]:
    expected_count = len(_extract_verification_clauses(task_description))
    if expected_count <= 0:
        return phases if phases else _build_fallback_phases(task_description, action_count)

    if len(phases) != expected_count:
        _debug_log(
            f"Phase count mismatch: expected={expected_count}, got={len(phases)}; using deterministic fallback"
        )
        return _build_fallback_phases(task_description, action_count)

    normalized: list[dict[str, Any]] = []
    clauses = _extract_verification_clauses(task_description)
    for index, phase in enumerate(phases, 1):
        action_range = phase.get("action_range")
        if not isinstance(action_range, list) or len(action_range) != 2:
            return _build_fallback_phases(task_description, action_count)
        start, end = action_range
        if not isinstance(start, int) or not isinstance(end, int):
            return _build_fallback_phases(task_description, action_count)
        intent = clauses[index - 1] if index - 1 < len(clauses) else str(phase.get("intent", "")).strip()
        if not intent:
            return _build_fallback_phases(task_description, action_count)
        normalized.append({
            "phase_id": index,
            "action_range": [max(1, start), max(1, end)],
            "intent": intent,
        })

    return normalized


def _normalize_llm_phase_boundaries(phases: list[dict[str, Any]], action_count: int) -> list[dict[str, Any]]:
    if not phases:
        return []

    cleaned: list[dict[str, Any]] = []
    for index, phase in enumerate(phases, 1):
        if not isinstance(phase, dict):
            continue
        action_range = phase.get("action_range")
        if not isinstance(action_range, list) or len(action_range) != 2:
            continue
        start, end = action_range
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        cleaned.append({
            "phase_id": index,
            "action_range": [max(1, start), max(1, end)],
        })

    if not cleaned:
        return []

    if len(cleaned) == 1:
        cleaned[0]["action_range"] = [1, action_count]
        return cleaned

    first_end = cleaned[0]["action_range"][1]
    first_end = max(1, min(first_end, action_count - 1)) if action_count > 1 else 1
    cleaned[0]["action_range"] = [1, first_end]
    cleaned[1]["action_range"] = [min(first_end + 1, action_count), action_count]
    cleaned[1]["phase_id"] = 2
    return cleaned


def _build_llm_phase_prompt(
    task_description: str,
    action_descriptions: list[str],
    category_label: str,
    max_phases: int = 3,
) -> list[dict[str, str]]:
    system_prompt = (
        "You are an Android test intent planner. Split the action sequence into logical phases and generate a phase intent for each phase. "
        "You must use the provided test category, TaskDescription, and action descriptions. "
        f"Prefer only 1 to {max_phases} phases, and do not create too many phases. "
        "A small number of coarse-grained phases is preferred over fine-grained splitting. "
        "Return ONLY a JSON array. Each element must contain phase_id, action_range, and intent. "
        "action_range must be a 1-based inclusive [start, end] pair. "
        "Do not include markdown fences or extra commentary."
    )
    user_payload = {
        "category_name": category_label,
        "task_description": task_description,
        "action_descriptions": action_descriptions,
        "rules": [
            "Use the category_name and TaskDescription to infer the high-level verification goal.",
            "Split the action sequence into 1 or 2 phases when possible.",
            "Do not make phases too fine-grained.",
            "The intent must describe a verification result, not an action sequence.",
            "The intent should usually start with check/verify and use if/whether.",
            "Do not use click/press/tap in the intent, and do not generate 'whether the button is clickable' or similar clickable-ability assertions.",
            "Pay special attention to clue words that often introduce the real verification goal in TaskDescription, such as Finally, you will see, you can see, or similar trailing phrases; the intent should focus on the checking meaning that follows those clues.",
            "Critical: When choosing a target control for existence/absence checks, do not reuse the control that is explicitly involved in the pre-action sequence(like 'click', 'search', 'input', 'set', 'enter', 'open' and so on); for example, if the action says Click the 'My Account' button, do not select the 'My Account' button as the detection target. " ,
            "Important: when you see keyword like 'Finally' or 'you will/can see' in the task description, it usually indicates a verification goal, so you should generate an intent that checks whether the expected result is achieved.",
            "Prefer intents about: whether a target control exists, whether the app has navigated to another app, whether a checkbox/button has been clicked, whether the current screen is dark or light mode, whether multiple target controls exist, or whether a control does not exist.",
            "If the task naturally contains two different verification moments, keep them as two phases.",
            "Return only phase_id, action_range, and intent.",
        ],
    }
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
    ]


def _build_llm_phase_retry_prompt(
    task_description: str,
    action_descriptions: list[str],
    category_label: str,
    feedback: list[str],
    max_phases: int = 3,
) -> list[dict[str, str]]:
    messages = _build_llm_phase_prompt(task_description, action_descriptions, category_label, max_phases=max_phases)
    retry_payload = {
        "feedback": feedback,
        "retry_rules": [
            "Fix the intents only; keep the same phase count and action_range if possible.",
            "Each intent must start with check/verify.",
            "Each intent must contain if or whether.",
            "Pay special attention to clue words such as Finally, you will see, you can see, or similar trailing phrases in TaskDescription; the intent should follow the checking meaning after these clues.",
            "Do not include click, press, tap, clickable, or phrases like 'can be clicked'.",
            "For existence/absence checks, do not choose the controls in the pre-action sequence as the detection target.",
            "Do not generate clickable-ability assertions; generate existence, navigation, clicked-state, mode, multi-control, or absence checks instead.",
            "CRITICAL: The action_range of all phases must cover every action from 1 to the total number of actions. Do not omit any action. The union of all phases must be exactly [1, total_actions].",
        ],
    }
    messages.append({"role": "user", "content": json.dumps(retry_payload, ensure_ascii=False, indent=2)})
    return messages


# ----- 新增辅助函数：将各种 phase_id 表示转换为整数 -----
def _parse_phase_id(value: Any, fallback: int) -> int:
    """
    尝试将 phase_id 转换为整数。
    支持形式：
      - 整数直接返回
      - 字符串如 "1", "phase_1", "phase 1", "第1阶段" 等（提取第一个数字）
      - 若转换失败，返回 fallback
    """
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        # 尝试提取第一个数字序列
        match = re.search(r'\d+', value)
        if match:
            return int(match.group())
    # 其他情况返回 fallback
    return fallback


def _parse_phase_payload(
    raw_text: str,
    action_count: int,
    task_description: str,
    require_intent: bool = True,
    normalize_intents: bool = True,
) -> PhasePlanResult:
    text = (raw_text or "").strip()
    if not text:
        return PhasePlanResult(
            phases=_fallback_from_task_description(task_description, action_count),
            reason="empty response",
            raw_response=raw_text,
        )

    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        payload = json.loads(text)
        phases = payload if isinstance(payload, list) else payload.get("phases", [])
        normalized: list[dict[str, Any]] = []
        if isinstance(phases, list):
            for index, item in enumerate(phases, 1):
                if not isinstance(item, dict):
                    continue
                action_range = item.get("action_range")
                intent = str(item.get("intent", "")).strip()
                phase_id = item.get("phase_id", index)   # 可能是字符串
                if (
                    isinstance(action_range, list)
                    and len(action_range) == 2
                    and all(isinstance(v, int) for v in action_range)
                    and (intent if require_intent else True)
                ):
                    start, end = action_range
                    parsed_phase_id = _parse_phase_id(phase_id, index)   # 关键修改
                    item: dict[str, Any] = {
                        "phase_id": parsed_phase_id,
                        "action_range": [max(1, start), max(1, end)],
                    }
                    if require_intent:
                        item["intent"] = intent
                    elif intent:
                        item["intent"] = intent
                    normalized.append(item)
        if normalized:
            if normalize_intents:
                normalized = _normalize_phase_count(normalized, task_description, action_count)
            return PhasePlanResult(normalized, "ok", raw_text)
    except Exception:
        pass

    return PhasePlanResult(
        phases=_fallback_from_task_description(task_description, action_count),
        reason="parse failed",
        raw_response=raw_text,
    )


def _load_client(base_url: str | None = None, api_key: str | None = None):
    client = make_openai_client(base_url, api_key)
    if client is None:
        _debug_log("Failed to initialize OpenAI-compatible client.")
    return client


def _check_phase_coverage(phases: list[dict[str, Any]], action_count: int) -> tuple[bool, str]:
    """
    返回 (是否全覆盖, 缺失描述)
    """
    if not phases:
        return False, "没有定义任何阶段"

    covered = set()
    for phase in phases:
        action_range = phase.get("action_range")
        if not isinstance(action_range, list) or len(action_range) != 2:
            continue
        start, end = action_range
        if not isinstance(start, int) or not isinstance(end, int):
            continue
        if start < 1:
            start = 1
        if end > action_count:
            end = action_count
        if start <= end:
            covered.update(range(start, end + 1))

    expected = set(range(1, action_count + 1))
    missing = expected - covered
    if not missing:
        return True, ""
    else:
        return False, f"动作 {sorted(missing)} 未被任何阶段覆盖"


def _plan_phases_with_llm(
    task_description: str,
    action_descriptions: list[str],
    category: int | None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> PhasePlanResult:
    client = _load_client(base_url, api_key)
    if client is None:
        _debug_log("LLM unavailable, falling back to deterministic single-phase plan.")
        return PhasePlanResult(_fallback_from_task_description(task_description, len(action_descriptions)), "LLM unavailable", "", 0)

    category_label = _category_label(category)
    action_count = len(action_descriptions)
    max_retries = 2  # 最多尝试2次（初始+1次重试）
    token_usage_total = 0
    last_raw_content = ""
    last_phases = None

    for attempt in range(max_retries + 1):
        try:
            if attempt == 0:
                messages = _build_llm_phase_prompt(task_description, action_descriptions, category_label)
            else:
                # 构建重试 prompt，携带之前的反馈
                feedback_messages = []
                if last_phases is not None:
                    coverage_ok, missing_desc = _check_phase_coverage(last_phases, action_count)
                    if not coverage_ok:
                        feedback_messages.append(f"上一次规划未覆盖全部动作: {missing_desc}")
                if not feedback_messages:
                    feedback_messages.append("请确保所有动作都被分配到某个阶段，且覆盖全部动作。")
                messages = _build_llm_phase_retry_prompt(
                    task_description,
                    action_descriptions,
                    category_label,
                    feedback_messages,
                )

            response = get_openai_chat_completion(
                client,
                model=model or _default_model(),
                messages=messages,
                temperature=0,
            )
            content = getattr(response.choices[0].message, "content", "") or ""
            token_usage = extract_openai_token_usage(response)
            token_usage_total += token_usage
            _debug_log(f"Raw LLM phase response (attempt {attempt+1}): {content!r}")

            # 解析响应
            parsed = _parse_phase_payload(content, action_count, task_description, require_intent=True, normalize_intents=True)
            phases = parsed.phases
            last_raw_content = content
            last_phases = phases

            # 检查覆盖
            coverage_ok, missing_desc = _check_phase_coverage(phases, action_count)
            if coverage_ok:
                # 验证 intent 格式
                feedback: list[str] = []
                for index, phase in enumerate(phases, 1):
                    intent = str(phase.get("intent", ""))
                    ok, errors = _validate_phase_intent(intent)
                    if not ok:
                        feedback.append(f"phase {index}: {'; '.join(errors)}")
                if not feedback:
                    # 完全通过
                    return PhasePlanResult(phases, "ok", content, token_usage_total)
                else:
                    # intent 格式有误，继续重试
                    _debug_log(f"Intent validation failed on attempt {attempt+1}: {feedback}")
                    if attempt < max_retries:
                        # 更新 last_phases 用于反馈
                        last_phases = phases
                        continue
                    else:
                        # 最后一次尝试失败，返回已解析的 phases，但标记有警告
                        _debug_log(f"Intent validation failed after {max_retries+1} attempts, returning last parsed phases.")
                        return PhasePlanResult(phases, f"intent validation warnings: {feedback}", content, token_usage_total)
            else:
                # 覆盖不全，需要重试
                _debug_log(f"Coverage check failed on attempt {attempt+1}: {missing_desc}")
                if attempt < max_retries:
                    # 保留 last_phases 以供反馈
                    continue
                else:
                    # 最后一次尝试仍失败，使用 fallback
                    _debug_log(f"Coverage still incomplete after {max_retries+1} attempts; falling back to deterministic plan.")
                    return PhasePlanResult(_fallback_from_task_description(task_description, action_count),
                                           f"fallback due to coverage: {missing_desc}", content, token_usage_total)

        except Exception as exc:
            _debug_log(f"LLM phase planning attempt {attempt+1} failed: {exc}")
            if attempt < max_retries:
                continue
            else:
                _debug_log("All LLM attempts failed; falling back to deterministic plan.")
                return PhasePlanResult(_fallback_from_task_description(task_description, action_count),
                                       f"failed: {exc}", "", token_usage_total)

    # 理论上不会执行到这里
    return PhasePlanResult(_fallback_from_task_description(task_description, action_count),
                           "unexpected end", "", token_usage_total)


def plan_phases(
    task_description: str,
    action_descriptions: list[str],
    category: int | None = None,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> PhasePlanResult:
    action_descriptions = [str(item).strip() for item in action_descriptions if str(item).strip()]
    action_count = len(action_descriptions)
    if action_count == 0:
        return PhasePlanResult([], "no actions", "", 0)

    clauses = _extract_verification_clauses(task_description)
    if not clauses:
        return _plan_phases_with_llm(
            task_description=task_description,
            action_descriptions=action_descriptions,
            category=category,
            base_url=base_url,
            api_key=api_key,
            model=model,
        )

    client = _load_client(base_url, api_key)
    if client is None:
        _debug_log("LLM unavailable, falling back to deterministic phase split.")
        return PhasePlanResult(_build_fallback_phases(task_description, action_count), "LLM unavailable", "", 0)

    phase_count = len(clauses)
    system_prompt = (
        "You are an Android test intent planner. Split the action sequence into logical phases. "
        "Use ONLY the TaskDescription and the action descriptions to plan phase boundaries. "
        "The number of phases must exactly match the number of verification clauses provided by the user. "
        f"There are exactly {phase_count} verification clauses, so return exactly {phase_count} phases. "
        "Do not invent additional phases and do not merge phases. "
        "Return ONLY a JSON array. Each element must contain phase_id and action_range. "
        "action_range must be a 1-based inclusive [start, end] pair. "
        "Do not include intent text, UI information, extra text, markdown fences, or commentary."
    )
    user_payload = {
        "task_description": task_description,
        "action_descriptions": action_descriptions,
        "verification_clauses": clauses,
        "rules": [
            "Phase count must equal verification_clauses count.",
            "Use verification_clauses only as boundary hints; do not change them.",
            "Do not output intent text; only output phase_id and action_range.",
            "Prefer boundaries that align with the verification clauses and the action flow.",
        ],
    }

    try:
        response = get_openai_chat_completion(
            client,
            model=model or _default_model(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False, indent=2)},
            ],
            temperature=0,
        )
        content = getattr(response.choices[0].message, "content", "") or ""
        token_usage = extract_openai_token_usage(response)
        _debug_log(f"Raw response: {content!r}")

        parsed = _parse_phase_payload(content, action_count, task_description, require_intent=False)

        if parsed.phases:
            normalized_phases = _normalize_llm_phase_boundaries(parsed.phases, action_count)
            if not normalized_phases:
                _debug_log("Parsed phase payload could not be normalized; using fallback plan.")
                return PhasePlanResult(_build_fallback_phases(task_description, action_count), "count mismatch", content, token_usage)

            phased: list[dict[str, Any]] = []
            for index, phase in enumerate(normalized_phases, 1):
                phase_intent = clauses[index - 1] if index - 1 < len(clauses) else _normalize_text(task_description)[:120]
                phased.append({
                    "phase_id": index,
                    "action_range": phase.get("action_range", [1, action_count]),
                    "intent": phase_intent,
                })

            if len(clauses) != len(normalized_phases):
                _debug_log(
                    f"Keeping LLM phase count {len(normalized_phases)} despite verification clause count {len(clauses)}."
                )
            # 检查覆盖
            coverage_ok, missing_desc = _check_phase_coverage(phased, action_count)
            if not coverage_ok:
                _debug_log(f"Coverage check failed in clause-based flow: {missing_desc}; falling back.")
                return PhasePlanResult(_build_fallback_phases(task_description, action_count), "coverage fallback", content, token_usage)
            return PhasePlanResult(phased, parsed.reason, parsed.raw_response, token_usage)

        _debug_log("Parsed phase payload was empty; using fallback plan.")
        return PhasePlanResult(_build_fallback_phases(task_description, action_count), "count mismatch", content, token_usage)
    except Exception as exc:
        _debug_log(f"Phase planning failed: {exc}")
        return PhasePlanResult(_build_fallback_phases(task_description, action_count), f"failed: {exc}", "", 0)