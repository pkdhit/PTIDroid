from __future__ import annotations

import re
import xml.etree.ElementTree as ET
import ast
from dataclasses import dataclass
from typing import Any

from appium.webdriver.common.appiumby import AppiumBy

from action_handlers import find_element_with_scroll


@dataclass(frozen=True)
class AssertionCheckResult:
    passed: bool
    reason: str
    matched_by: str | None = None
    fatal: bool = False


_STRING_LITERAL = r"(['\"])(?:\\.|(?!\1).)*\1"


_ASSERTION_PATTERNS: dict[str, re.Pattern[str]] = {
    "check_dark_mode": re.compile(rf"^check_dark_mode\({ _STRING_LITERAL }\)$"),
    "check_light_mode": re.compile(rf"^check_light_mode\({ _STRING_LITERAL }\)$"),
    "check_current_app_package": re.compile(
        rf"^check_current_app_package\({ _STRING_LITERAL }(?:,\s*{ _STRING_LITERAL })?\)$"
    ),
    "check_elements_same_screen": re.compile(
        rf"^check_elements_same_screen\(\s*(?:driver,\s*)?(?:\[(?:\s*{ _STRING_LITERAL }\s*(?:,\s*{ _STRING_LITERAL }\s*)*)?\]|\((?:\s*{ _STRING_LITERAL }\s*(?:,\s*{ _STRING_LITERAL }\s*)*)?\)|{ _STRING_LITERAL }(?:\s*,\s*{ _STRING_LITERAL })*)\s*\)$"
    ),
    "check_row_checked_by_text": re.compile(rf"^check_row_checked_by_text\({ _STRING_LITERAL }\)$"),
    "find_element_xpath_is_displayed": re.compile(
        rf"^elem\s*=\s*driver\.find_element\(By\.(?:XPATH|ID),\s*{ _STRING_LITERAL }\)\s*;\s*assert\s+elem\.is_displayed\(\)\s*$"
    ),
    "find_element_xpath_text_equals": re.compile(
        rf"^elem\s*=\s*driver\.find_element\(By\.(?:XPATH|ID),\s*{ _STRING_LITERAL }\)\s*;\s*assert\s+elem\.text\s*==\s*{ _STRING_LITERAL }\s*$"
    ),
    "find_elements_len_zero": re.compile(
        rf"^elems\s*=\s*driver\.find_elements\(By\.(?:XPATH|ID),\s*{ _STRING_LITERAL }\)\s*;\s*assert\s+len\(elems\)\s*==\s*0\s*$"
    ),
}


def _has_malformed_same_screen_quotes(assertion: str) -> bool:
    """Detect unescaped quotes inside check_elements_same_screen arguments.

    We intentionally do not auto-fix these assertions. Instead, the caller
    should surface a clear format warning so the generator can regenerate the
    assertion with proper quoting/escaping.
    """
    normalized = assertion.strip()
    if not normalized.startswith("check_elements_same_screen("):
        return False

    if ", [" not in normalized and ", (" not in normalized:
        return False

    # Common malformed pattern:
    # check_elements_same_screen(driver, ['//...[@text='Breaking News']', ...])
    # The outer list item is single-quoted, and the XPath predicate also uses
    # single quotes, which makes the Python expression invalid.
    malformed_xpath_quote_pattern = re.compile(
        r"(?:\[\s*|\(\s*)'//.*?\[@[^\]]*?=\s*'",
        re.DOTALL,
    )
    if malformed_xpath_quote_pattern.search(normalized):
        return True

    malformed_xpath_double_quote_pattern = re.compile(
        r'(?:\[\s*|\(\s*)"//.*?\[@[^\]]*?=\s*"',
        re.DOTALL,
    )
    if malformed_xpath_double_quote_pattern.search(normalized):
        return True

    # Also catch already-split malformed fragments such as:
    # "['//...[@resource-id='android:id/title' and @text='Network']'"
    if normalized.startswith("check_elements_same_screen(") and "'//" in normalized and "=@" in normalized:
        if re.search(r"'//.*\[@[^\]]*='", normalized, re.DOTALL):
            return True

    try:
        ast.parse(normalized, mode="eval")
        return False
    except SyntaxError as exc:
        message = str(exc).lower()
        return (
            "unterminated string" in message
            or "eol while scanning string literal" in message
            or "invalid syntax" in message
        )
    except Exception:
        # When the expression cannot even be parsed, this is still very likely
        # to be a quoting problem for same-screen assertions.
        return True


def _is_known_assertion_like(assertion: str) -> bool:
    normalized = assertion.strip()
    if not normalized:
        return False

    return (
        normalized.startswith("check_current_app_package(")
        or normalized.startswith("check_dark_mode(")
        or normalized.startswith("check_light_mode(")
        or normalized.startswith("check_row_checked_by_text(")
        or normalized.startswith("check_elements_same_screen(")
        or normalized.startswith("elem = driver.find_element(")
        or normalized.startswith("elems = driver.find_elements(")
    )


def _has_quote_syntax_issue(assertion: str) -> bool:
    normalized = assertion.strip()
    if not normalized or not _is_known_assertion_like(normalized):
        return False

    try:
        ast.parse(normalized, mode="exec")
        return False
    except SyntaxError as exc:
        message = str(exc).lower()
        return (
            "unterminated string" in message
            or "eol while scanning string literal" in message
            or "invalid syntax" in message
            or "unterminated triple-quoted string" in message
        )
    except Exception:
        return True


def _find_malformed_xpath_predicate_quotes(text: str) -> list[str]:
    """Return malformed XPath predicate text snippets that contain unescaped same-quote characters.

    Examples that should be rejected:
    - @text='Passwords don't match'
    - @text="The user name "Aurora" is not available"

    The helper intentionally stays conservative and focuses on the XPath
    patterns generated by this project.
    """
    normalized = text.strip()
    if not normalized:
        return []

    malformed_snippets: list[str] = []

    def _attribute_value_segment(segment: str) -> str:
        """Trim trailing predicates so we only inspect the current attribute value.

        Example:
        "'org:id/title' and @text='Passwords don't match'"
        -> "'org:id/title'"
        "'Passwords don't match'"
        -> unchanged
        """
        delimiter = segment.find(" and @")
        return segment[:delimiter].strip() if delimiter != -1 else segment.strip()

    for match in re.finditer(r"@(?:text|content-desc|resource-id)\s*=\s*(['\"])", normalized):
        quote = match.group(1)
        start = match.end()
        end = normalized.find("]", start)
        if end == -1:
            end = len(normalized)

        segment = _attribute_value_segment(normalized[start:end])
        escaped = False
        quote_positions: list[int] = []
        for idx, ch in enumerate(segment):
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == quote:
                quote_positions.append(idx)

        # A valid XPath quoted literal should contain exactly one unescaped same-quote
        # inside the scanned segment (the closing quote of the current attribute value).
        # Any additional same-quote character means the text is malformed,
        # e.g. @text='Passwords don't match'.
        if len(quote_positions) != 1:
            malformed_snippets.append(f"{quote}{segment.strip()}")

    return malformed_snippets


def _python_double_quoted_string_literal(value: str) -> str:
    escaped = value.replace("\\", r"\\").replace('"', r'\"')
    return f'"{escaped}"'


def _xpath_string_literal(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'

    parts: list[str] = []
    segments = value.split("'")
    for index, segment in enumerate(segments):
        if segment:
            parts.append(f"'{segment}'")
        if index < len(segments) - 1:
            parts.append('"' + "'" + '"')
    return f"concat({', '.join(parts)})"


def _repair_xpath_predicate_quotes(assertion: str) -> str:
    normalized = assertion.strip()
    if not normalized or "By.XPATH" not in normalized:
        return assertion

    pattern = re.compile(
        r"(?P<prefix>.*?By\.XPATH,\s*)(?P<quote>['\"])(?P<xpath>.*)(?P=quote)(?P<suffix>\s*\)\s*;\s*assert.*)$",
        re.DOTALL,
    )
    match = pattern.match(normalized)
    if not match:
        return assertion

    prefix = match.group("prefix")
    xpath_text = match.group("xpath")
    suffix = match.group("suffix")

    def _find_value_end(text: str, start: int) -> int:
        candidates: list[int] = []
        for token in (
            " and @",
            " or @",
            "]",
        ):
            pos = text.find(token, start)
            if pos != -1:
                candidates.append(pos)
        return min(candidates) if candidates else len(text)

    rebuilt: list[str] = []
    cursor = 0
    attr_pattern = re.compile(r"@(?P<attr>text|content-desc|resource-id)\s*=\s*")

    for attr_match in attr_pattern.finditer(xpath_text):
        start = attr_match.start()
        value_start = attr_match.end()
        value_end = _find_value_end(xpath_text, value_start)

        rebuilt.append(xpath_text[cursor:start])
        raw_segment = xpath_text[value_start:value_end].strip()
        if len(raw_segment) >= 2 and raw_segment[0] == raw_segment[-1] and raw_segment[0] in {'"', "'"}:
            raw_value = raw_segment[1:-1]
        else:
            raw_value = raw_segment.strip('"\'')

        rebuilt.append(f"@{attr_match.group('attr')}={_xpath_string_literal(raw_value)}")
        cursor = value_end

    rebuilt.append(xpath_text[cursor:])
    repaired_xpath = ''.join(rebuilt)
    repaired_xpath = _python_double_quoted_string_literal(repaired_xpath)
    return f"{prefix}{repaired_xpath}{suffix}"


def _has_malformed_xpath_predicate_quotes(text: str) -> bool:
    return bool(_find_malformed_xpath_predicate_quotes(text))


def _assertion_is_allowed(assertion: str, allowed_assertions: list[str]) -> bool:
    normalized = assertion.strip()
    if not normalized:
        return False

    for name, pattern in _ASSERTION_PATTERNS.items():
        if pattern.fullmatch(normalized):
            if name.startswith("check_"):
                return name in allowed_assertions
            return True

    # 兜底：一些模型会生成等价但空格/分号略有变化的单行断言。
    # 这里不放宽到任意代码，只放宽到本项目已允许的断言结构。
    compact = re.sub(r"\s+", " ", normalized)
    if compact.startswith("check_current_app_package("):
        return "check_current_app_package" in allowed_assertions and compact.endswith(")")
    if compact.startswith("check_dark_mode("):
        return "check_dark_mode" in allowed_assertions and compact.endswith(")")
    if compact.startswith("check_light_mode("):
        return "check_light_mode" in allowed_assertions and compact.endswith(")")
    if compact.startswith("check_row_checked_by_text("):
        return "check_row_checked_by_text" in allowed_assertions and compact.endswith(")")
    if compact.startswith("check_elements_same_screen("):
        return "check_elements_same_screen" in allowed_assertions and compact.endswith(")")

    element_prefixes = (
        "elem = driver.find_element(By.XPATH,",
        "elem = driver.find_element(By.ID,",
        "elems = driver.find_elements(By.XPATH,",
        "elems = driver.find_elements(By.ID,",
    )
    if any(compact.startswith(prefix) for prefix in element_prefixes):
        if "; assert elem.is_displayed()" in compact:
            return True
        if "; assert elem.text == " in compact:
            return True
        if "; assert len(elems) == 0" in compact:
            return True

    return False


def _describe_format_violation(assertion: str) -> str:
    normalized = assertion.strip()
    if not normalized:
        return "空断言"

    if _has_malformed_same_screen_quotes(normalized):
        return (
            "引号格式错误：`check_elements_same_screen` 中的 XPath 字符串包含未转义的引号，"
            "请把内部文本引号改成双引号，或使用转义字符后重新生成"
        )

    if _has_quote_syntax_issue(normalized):
        return (
            "引号格式错误：断言中的字符串包含未转义的同类引号，"
            "请检查单引号/双引号嵌套并重新生成"
        )

    if _has_malformed_xpath_predicate_quotes(normalized):
        malformed_texts = _find_malformed_xpath_predicate_quotes(normalized)
        details = f"，具体问题文本：{'; '.join(malformed_texts)}" if malformed_texts else ""
        return (
            "引号格式错误：XPath 字符串中的文本内容包含未转义的同类引号，"
            "请改用另一种引号包裹文本或给引号添加转义字符"
            f"{details}"
        )

    tail_markers = {
        " is False": "多了尾随修饰 `is False`",
        " is True": "多了尾随修饰 `is True`",
        " == True": "多了尾随修饰 `== True`",
        " == False": "多了尾随修饰 `== False`",
        " is not None": "多了尾随修饰 `is not None`",
        " is None": "多了尾随修饰 `is None`",
    }
    for marker, message in tail_markers.items():
        if marker in normalized:
            return message

    if normalized.startswith("elem = driver.find_element") and "; assert" not in normalized:
        return "格式不完整：`find_element` 断言缺少分号或 `assert` 后半部分"
    if normalized.startswith("elems = driver.find_elements") and "; assert" not in normalized:
        return "格式不完整：`find_elements` 断言缺少分号或 `assert len(elems) == 0` 后半部分"

    return "未匹配允许的断言模板"


_XPATH_RE = re.compile(r"By\.XPATH\s*,\s*(['\"])(.*?)\1")
_ID_RE = re.compile(r"By\.ID\s*,\s*(['\"])(.*?)\1")
_LEN_ZERO_RE = re.compile(r"assert\s+len\s*\(\s*[^)]+\s*\)\s*==\s*0")
_XPATH_RESOURCE_ID_RE = re.compile(r"@resource-id\s*=\s*(['\"])(.*?)\1")
_XPATH_CLASS_RE = re.compile(r"^//([^\[]+)\s*$")
_XPATH_CLASS_PREFIX_RE = re.compile(r"^//([^\[]+)")


def _extract_locator(assertion_line: str) -> tuple[str | None, str | None]:
    xpath_match = _XPATH_RE.search(assertion_line)
    if xpath_match:
        return "xpath", xpath_match.group(2)

    id_match = _ID_RE.search(assertion_line)
    if id_match:
        return "id", id_match.group(2)

    return None, None


def _extract_assertion_target_key(assertion_line: str) -> str | None:
    """Extract a stable key for the target control referenced by an assertion.

    This is used to deduplicate multiple assertions generated for the same UI
    target before format/locator validation runs.
    """
    locator_type, locator_value = _extract_locator(assertion_line)
    if locator_type and locator_value:
        normalized_value = locator_value.strip()
        if locator_type == "xpath":
            normalized_value = _strip_text_predicates(normalized_value)
        if normalized_value:
            return f"{locator_type}:{normalized_value}"

    same_screen_targets = _parse_check_elements_same_screen_args(assertion_line)
    if same_screen_targets:
        normalized_targets = []
        for target in same_screen_targets:
            target_text = str(target).strip()
            if not target_text:
                continue
            if target_text.startswith("//"):
                target_text = _strip_text_predicates(target_text)
            normalized_targets.append(target_text)
        if normalized_targets:
            return "same_screen:" + "|".join(sorted(dict.fromkeys(normalized_targets)))

    return None


def _strip_text_predicates(xpath: str) -> str:
    xpath = re.sub(r"\[@text\s*=\s*(['\"]).*?\1\]", "]", xpath)
    xpath = re.sub(r"\[@content-desc\s*=\s*(['\"]).*?\1\]", "]", xpath)
    xpath = re.sub(r"\s+and\s+@text\s*=\s*(['\"]).*?\1", "", xpath)
    xpath = re.sub(r"\s+and\s+@content-desc\s*=\s*(['\"]).*?\1", "", xpath)
    xpath = xpath.replace("[]", "")
    xpath = re.sub(r"\s+and\s+\]", "]", xpath)
    return xpath


def _is_nonexistence_assertion(assertion_line: str) -> bool:
    return bool(_LEN_ZERO_RE.search(assertion_line or ""))


def _task_description_has_delete_intent(task_description: str | None) -> bool:
    text = str(task_description or "").lower()
    delete_keywords = (
        "remove",
        "delete",
        "delete task",
        "remove task",
        "delete item",
        "remove item",
        "delete mail",
        "remove mail",
        "delete message",
        "remove message",
    )
    return any(keyword in text for keyword in delete_keywords)


def _parse_page_source(page_source: str | None) -> ET.Element | None:
    if not page_source:
        return None
    try:
        return ET.fromstring(page_source)
    except Exception:
        return None


def _snapshot_page_source(snapshot: dict[str, Any] | None) -> str:
    if not snapshot:
        return ""
    ui = snapshot.get("ui") or {}
    return str(ui.get("raw_page_source") or "")


def _snapshot_page_sources(snapshot: dict[str, Any] | None) -> list[str]:
    if not snapshot:
        return []

    ui = snapshot.get("ui") or {}
    page_sources = ui.get("page_sources") or []
    if not isinstance(page_sources, list):
        return []

    normalized_sources: list[str] = []
    for page_source in page_sources:
        if isinstance(page_source, str) and page_source.strip():
            normalized_sources.append(page_source.strip())
    return normalized_sources


def _snapshot_ui_elements(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    if not snapshot:
        return []

    ui = snapshot.get("ui") or {}
    elements = ui.get("elements") or []
    if isinstance(elements, list):
        return [element for element in elements if isinstance(element, dict)]
    return []


def _element_attributes(element: dict[str, Any]) -> dict[str, Any]:
    attributes = element.get("attributes") or {}
    return attributes if isinstance(attributes, dict) else {}


def _snapshot_nodes(snapshot: dict[str, Any] | None) -> list[dict[str, Any]]:
    return _snapshot_ui_elements(snapshot)


def _snapshot_roots(snapshot: dict[str, Any] | None, include_page_sources: bool = False) -> list[ET.Element]:
    roots: list[ET.Element] = []
    page_sources = _snapshot_page_sources(snapshot) if include_page_sources else []
    sources = page_sources or [_snapshot_page_source(snapshot)]

    for page_source in sources:
        root = _parse_page_source(page_source)
        if root is not None:
            roots.append(root)
    return roots


def _node_matches_locator(node: dict[str, Any], locator_type: str | None, locator_value: str | None) -> bool:
    if not locator_type or not locator_value:
        return False

    attributes = _element_attributes(node)
    node_resource_id = str(attributes.get("resource-id", "") or "").strip()
    node_class = str(attributes.get("class", "") or "").strip()

    locator_value = locator_value.strip()
    if not locator_value:
        return False

    if locator_type == "id":
        return bool(node_resource_id) and node_resource_id == locator_value

    if locator_type != "xpath":
        return False

    hints = _extract_xpath_hints(_strip_text_predicates(locator_value))
    hint_class = hints.get("class", "").strip()
    hint_resource_id = hints.get("resource-id", "").strip()

    if hint_resource_id:
        return bool(node_resource_id) and node_resource_id == hint_resource_id

    if hint_class and node_class:
        if node_class == hint_class or node_class.startswith(hint_class) or hint_class.startswith(node_class):
            return True

    return False


def _snapshot_contains_xpath(snapshot: dict[str, Any] | None, xpath: str, include_page_sources: bool = False) -> bool:
    if not include_page_sources:
        for node in _snapshot_nodes(snapshot):
            if _node_matches_locator(node, "xpath", xpath):
                return True
        return False

    for root in _snapshot_roots(snapshot, include_page_sources=True):
        if _xml_contains_target(root, xpath):
            return True
    return False


def _snapshot_contains_locator(
    snapshot: dict[str, Any] | None,
    locator_type: str | None,
    locator_value: str | None,
    include_page_sources: bool = False,
) -> bool:
    if not include_page_sources:
        for node in _snapshot_nodes(snapshot):
            if _node_matches_locator(node, locator_type, locator_value):
                return True
        return False

    for root in _snapshot_roots(snapshot, include_page_sources=True):
        if _xml_contains_target_prefix(root, locator_type, locator_value):
            return True
    return False


def _snapshot_signature_set(snapshot: dict[str, Any] | None) -> set[str]:
    signatures: set[str] = set()
    nodes = _snapshot_nodes(snapshot)
    if nodes:
        for node in nodes:
            attributes = _element_attributes(node)
            resource_id = str(attributes.get("resource-id", "") or "").strip()
            text = str(attributes.get("text", "") or "").strip()
            content_desc = str(attributes.get("content-desc", "") or "").strip()
            class_name = str(attributes.get("class", "") or "").strip()
            if not any([resource_id, text, content_desc, class_name]):
                continue
            signatures.add(f"{class_name}|{resource_id}|{text}|{content_desc}")
        return signatures

    root = _parse_page_source(_snapshot_page_source(snapshot))
    if root is None:
        return set()

    for node in root.iter():
        resource_id = str(node.attrib.get("resource-id", "") or "").strip()
        text = str(node.attrib.get("text", "") or "").strip()
        content_desc = str(node.attrib.get("content-desc", "") or "").strip()
        class_name = str(node.attrib.get("class", "") or "").strip()
        if not any([resource_id, text, content_desc, class_name]):
            continue
        signatures.add(f"{class_name}|{resource_id}|{text}|{content_desc}")
    return signatures


def _snapshot_similarity(current_snapshot: dict[str, Any] | None, candidate_snapshot: dict[str, Any] | None) -> tuple[int, int, int, int]:
    current_package = str((current_snapshot or {}).get("current_package") or "").strip()
    current_activity = str((current_snapshot or {}).get("current_activity") or "").strip()
    candidate_package = str((candidate_snapshot or {}).get("current_package") or "").strip()
    candidate_activity = str((candidate_snapshot or {}).get("current_activity") or "").strip()

    score = 0
    if current_package and candidate_package and current_package == candidate_package:
        score += 100
    if current_activity and candidate_activity and current_activity == candidate_activity:
        score += 60

    current_signatures = _snapshot_signature_set(current_snapshot)
    candidate_signatures = _snapshot_signature_set(candidate_snapshot)
    if current_signatures and candidate_signatures:
        overlap = len(current_signatures & candidate_signatures)
        score += overlap * 3
        score -= abs(len(current_signatures) - len(candidate_signatures))

    if _snapshot_page_source(current_snapshot) and _snapshot_page_source(candidate_snapshot):
        score += 1 if _snapshot_page_source(current_snapshot) != _snapshot_page_source(candidate_snapshot) else 0

    step_index = int(candidate_snapshot.get("step_index") or 0) if candidate_snapshot else 0
    return score, step_index, len(candidate_signatures), len(_snapshot_page_source(candidate_snapshot))


def _extract_xpath_hints(xpath: str) -> dict[str, str]:
    hints: dict[str, str] = {}
    resource_id_match = _XPATH_RESOURCE_ID_RE.search(xpath)
    if resource_id_match:
        hints["resource-id"] = resource_id_match.group(2)

    class_match = _XPATH_CLASS_RE.match(xpath.strip())
    if class_match:
        hints["class"] = class_match.group(1).strip()
    else:
        class_prefix_match = _XPATH_CLASS_PREFIX_RE.match(xpath.strip())
        if class_prefix_match:
            hints["class"] = class_prefix_match.group(1).strip()
    return hints


def _locator_matches_prefix(node: ET.Element, locator_type: str | None, locator_value: str | None) -> bool:
    if not locator_type or not locator_value:
        return False

    locator_value = locator_value.strip()
    if not locator_value:
        return False

    node_resource_id = str(node.attrib.get("resource-id", "") or "").strip()
    node_class = str(node.attrib.get("class", "") or "").strip()

    if locator_type == "id":
        return bool(node_resource_id) and node_resource_id == locator_value

    if locator_type != "xpath":
        return False

    hints = _extract_xpath_hints(_strip_text_predicates(locator_value))
    hint_class = hints.get("class", "").strip()
    hint_resource_id = hints.get("resource-id", "").strip()

    if hint_resource_id:
        return bool(node_resource_id) and node_resource_id == hint_resource_id

    if hint_class and node_class:
        if node_class == hint_class or node_class.startswith(hint_class) or hint_class.startswith(node_class):
            return True

    return False


def _normalize_same_screen_target_value(value: Any) -> str | None:
    """Normalize targets for check_elements_same_screen.

    The generator may produce nested stringified list items such as:
    - "['//a', '//b']"
    - "'//a'"
    - '"//a"'
    This helper strips one or more layers of quoting/list wrappers so the
    checker can evaluate the same XPath values as normal assertions.
    """
    if value is None:
        return None

    current: Any = value
    for _ in range(3):
        if isinstance(current, (list, tuple)):
            # Caller handles lists/tuples separately; keep first scalar path here.
            return None

        if not isinstance(current, str):
            current = str(current)

        text = current.strip()
        if not text:
            return None

        if len(text) >= 2 and text[0] == text[-1] and text[0] in {"'", '"'}:
            current = text[1:-1].strip()
            continue

        try:
            parsed = ast.literal_eval(text)
        except Exception:
            return text

        if isinstance(parsed, str):
            current = parsed.strip()
            continue

        if isinstance(parsed, (list, tuple)):
            # Return the outer caller's list handling path.
            return None

        return str(parsed).strip() or None

    return str(current).strip() or None


def _coerce_same_screen_targets(value: Any) -> list[str]:
    if value is None:
        return []

    if isinstance(value, (list, tuple)):
        normalized_items: list[str] = []
        for item in value:
            normalized_items.extend(_coerce_same_screen_targets(item))
        return normalized_items

    normalized_item = _normalize_same_screen_target_value(value)
    return [normalized_item] if normalized_item else []


def _drop_leading_driver_target(targets: list[str]) -> list[str]:
    if not targets:
        return []

    cleaned_targets = [target for target in targets if str(target).strip()]
    if cleaned_targets and str(cleaned_targets[0]).strip() == "driver":
        return cleaned_targets[1:]
    return cleaned_targets


def _parse_check_elements_same_screen_args(assertion_line: str) -> list[str] | None:
    match = re.match(r"^check_elements_same_screen\s*\((.*)\)\s*$", assertion_line.strip())
    if not match:
        return None

    inner = match.group(1).strip()
    if not inner:
        return None

    # Support:
    # - check_elements_same_screen(driver, ['xpath1', 'xpath2'])
    # by dropping a leading driver argument when present.
    try:
        parsed_call = ast.parse(assertion_line.strip(), mode="eval")
        if isinstance(parsed_call.body, ast.Call):
            call = parsed_call.body
            if isinstance(call.func, ast.Name) and call.func.id == "check_elements_same_screen":
                args = list(call.args)
                if args and isinstance(args[0], ast.Name) and args[0].id == "driver":
                    args = args[1:]

                normalized_args: list[str] = []
                for arg in args:
                    try:
                        value = ast.literal_eval(arg)
                    except Exception:
                        continue
                    normalized_args.extend(_coerce_same_screen_targets(value))
                normalized_args = _drop_leading_driver_target(normalized_args)
                if normalized_args:
                    return normalized_args
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(inner)
    except Exception:
        if _has_malformed_same_screen_quotes(assertion_line):
            return None

        parts: list[str] = []
        current: list[str] = []
        quote: str | None = None
        escape = False

        for ch in inner:
            if escape:
                current.append(ch)
                escape = False
                continue
            if ch == '\\':
                current.append(ch)
                escape = True
                continue
            if quote:
                current.append(ch)
                if ch == quote:
                    quote = None
                continue
            if ch in ("'", '"'):
                current.append(ch)
                quote = ch
                continue
            if ch == ',':
                item = ''.join(current).strip()
                if item:
                    parts.append(item)
                current = []
                continue
            current.append(ch)

        item = ''.join(current).strip()
        if item:
            parts.append(item)

        if not parts:
            return None

        normalized: list[str] = []
        for item in parts:
            item = item.strip()
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                normalized.append(item[1:-1])
            else:
                normalized.append(item)
        return _drop_leading_driver_target([item for item in normalized if str(item).strip()])

    if isinstance(parsed, str):
        nested_text = parsed.strip()
        if nested_text:
            try:
                nested_parsed = ast.literal_eval(nested_text)
                normalized_items = _coerce_same_screen_targets(nested_parsed)
                if normalized_items:
                    return normalized_items
            except Exception:
                pass
        normalized_item = _normalize_same_screen_target_value(parsed)
        return [normalized_item] if normalized_item else None
    if isinstance(parsed, (list, tuple)):
        normalized_items = _coerce_same_screen_targets(parsed)
        return normalized_items or None
    return None


def _xml_contains_target(root: ET.Element, xpath: str) -> bool:
    normalized_xpath = _strip_text_predicates(xpath)
    hints = _extract_xpath_hints(normalized_xpath)

    for node in root.iter():
        if "resource-id" in hints and str(node.attrib.get("resource-id", "") or "").strip() == hints["resource-id"]:
            return True
        if "class" in hints and str(node.attrib.get("class", "") or "").strip() == hints["class"]:
            return True
    return False


def _xml_contains_target_prefix(root: ET.Element, locator_type: str | None, locator_value: str | None) -> bool:
    if not locator_type or not locator_value:
        return False

    for node in root.iter():
        if _locator_matches_prefix(node, locator_type, locator_value):
            return True
    return False


def _is_toast_target(assertion_line: str, current_snapshot: dict[str, Any] | None) -> bool:
    line = (assertion_line or "").lower()
    if "toast" in line:
        return True

    nodes = _snapshot_nodes(current_snapshot)
    if nodes:
        for node in nodes:
            attributes = _element_attributes(node)
            class_name = str(attributes.get("class", "") or "").lower()
            resource_id = str(attributes.get("resource-id", "") or "").lower()
            text = str(attributes.get("text", "") or "").lower()
            content_desc = str(attributes.get("content-desc", "") or "").lower()

            if "toast" in class_name:
                return True
            if any("toast" in value for value in (resource_id, text, content_desc)):
                return True

        return False

    page_source = _snapshot_page_source(current_snapshot)
    if not page_source:
        return False

    root = _parse_page_source(page_source)
    if root is None:
        return "toast" in page_source.lower()

    for node in root.iter():
        class_name = str(node.attrib.get("class", "") or "").lower()
        resource_id = str(node.attrib.get("resource-id", "") or "").lower()
        text = str(node.attrib.get("text", "") or "").lower()
        content_desc = str(node.attrib.get("content-desc", "") or "").lower()

        if "toast" in class_name:
            return True
        if any("toast" in value for value in (resource_id, text, content_desc)):
            return True

    return False


def _element_exists_in_snapshot(
    snapshot: dict[str, Any] | None,
    locator_type: str | None,
    locator_value: str | None,
    include_page_sources: bool = False,
) -> bool:
    if not snapshot or not locator_type or not locator_value:
        return False

    if _snapshot_contains_locator(snapshot, locator_type, locator_value, include_page_sources=include_page_sources):
        return True

    roots = _snapshot_roots(snapshot, include_page_sources=include_page_sources)
    if not roots:
        return False
    locator_value = locator_value.strip()
    for root in roots:
        for node in root.iter():
            node_resource_id = str(node.attrib.get("resource-id", "") or "").strip()
            node_class = str(node.attrib.get("class", "") or "").strip()

            if locator_type == "id":
                if node_resource_id and node_resource_id == locator_value:
                    return True
                continue

            if locator_type != "xpath":
                continue

            hints = _extract_xpath_hints(_strip_text_predicates(locator_value))
            hint_resource_id = hints.get("resource-id", "").strip()
            hint_class = hints.get("class", "").strip()

            if hint_resource_id and node_resource_id:
                if node_resource_id == hint_resource_id:
                    return True

            if hint_class and node_class:
                if node_class.startswith(hint_class) or hint_class.startswith(node_class):
                    return True

    return False


def _element_exists_in_snapshot_with_prefix_fallback(
    snapshot: dict[str, Any] | None,
    locator_type: str | None,
    locator_value: str | None,
    include_page_sources: bool = False,
) -> tuple[bool, str]:
    if not snapshot or not locator_type or not locator_value:
        return False, ""

    if _snapshot_contains_locator(snapshot, locator_type, locator_value, include_page_sources=include_page_sources):
        return True, "exact"

    for root in _snapshot_roots(snapshot, include_page_sources=include_page_sources):
        if _xml_contains_target_prefix(root, locator_type, locator_value):
            return True, "prefix"

    return False, ""


def _select_similarity_history_snapshot(
    snapshot_history: list[dict[str, Any]] | None,
    current_snapshot: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if not snapshot_history:
        return None

    ranked_candidates = sorted(
        enumerate(snapshot_history),
        key=lambda item: (
            _snapshot_similarity(current_snapshot, item[1]),
            item[0],
        ),
        reverse=True,
    )
    return ranked_candidates[0][1] if ranked_candidates else None


def _is_session_lost_exception(exc: Exception) -> bool:
    message = str(exc).lower()
    return (
        "no such drivererror" in message
        or "a session is either terminated or not started" in message
        or "invalid session id" in message
        or "session not created" in message
        or "session terminated" in message
    )

def check_generated_assertion(
    driver,
    assertion_line: str,
    snapshot_history: list[dict[str, Any]] | None = None,
    current_snapshot: dict[str, Any] | None = None,
    task_description: str | None = None,
    include_page_sources: bool = False,
) -> AssertionCheckResult:
    assertion_line = assertion_line.strip()
    if not assertion_line:
        return AssertionCheckResult(False, "empty assertion")

    format_violation = _describe_format_violation(assertion_line)
    if format_violation != "未匹配允许的断言模板":
        return AssertionCheckResult(False, format_violation)

    locator_type, locator_value = _extract_locator(assertion_line)
    nonexistence_assertion = _is_nonexistence_assertion(assertion_line)

    if _is_toast_target(assertion_line, current_snapshot):
        return AssertionCheckResult(True, "toast target treated as passed")

    # 非删除意图的不存在性断言直接放行
    if nonexistence_assertion and not _task_description_has_delete_intent(task_description):
        return AssertionCheckResult(True, "nonexistence assertion skipped because task is not a delete/remove task", locator_value)

    # 删除意图的不存在性断言：只检查历史快照中是否存在目标
    if nonexistence_assertion and locator_type and locator_value:
        candidate_snapshot = _select_similarity_history_snapshot(snapshot_history, current_snapshot)
        if candidate_snapshot is not None:
            exists_in_history = _element_exists_in_snapshot(
                candidate_snapshot,
                locator_type,
                locator_value,
                include_page_sources=include_page_sources,
            )
            if exists_in_history:
                return AssertionCheckResult(
                    True,
                    "target element found in similar pre-delete snapshot",
                    locator_value,
                )
            else:
                return AssertionCheckResult(
                    False,
                    "target element not found in similar pre-delete snapshot; cannot confirm deletion",
                    locator_value,
                )
        else:
            return AssertionCheckResult(
                False,
                "no similar pre-delete snapshot found; cannot confirm deletion",
                locator_value,
            )

    # 以下是非不存在性断言（存在性断言）的正常检查
    same_screen_targets = _parse_check_elements_same_screen_args(assertion_line)
    if same_screen_targets is not None:
        if _has_malformed_same_screen_quotes(assertion_line):
            return AssertionCheckResult(
                False,
                "引号格式错误：`check_elements_same_screen` 中的 XPath 字符串包含未转义的引号，请把内部文本引号改成双引号或使用转义后重新生成",
            )

        missing_targets = []
        prefix_matched_targets = []
        for xpath in same_screen_targets:
            current_exists, match_mode = _element_exists_in_snapshot_with_prefix_fallback(
                current_snapshot,
                "xpath",
                xpath,
                include_page_sources=include_page_sources,
            )
            if current_exists and match_mode == "exact":
                continue
            if current_exists and match_mode == "prefix":
                prefix_matched_targets.append(xpath)
                continue
            missing_targets.append(xpath)

        if missing_targets:
            return AssertionCheckResult(
                False,
                f"target element(s) not present on the current screen: {missing_targets}",
            )
        if prefix_matched_targets:
            return AssertionCheckResult(
                True,
                f"matched by xpath/resource-id prefix on current screen: {prefix_matched_targets}",
                ",".join(prefix_matched_targets),
            )
        return AssertionCheckResult(True, "all target elements present on the current screen")

    if locator_type == "id" and locator_value:
        try:
            current_exists, match_mode = _element_exists_in_snapshot_with_prefix_fallback(
                current_snapshot,
                locator_type,
                locator_value,
                include_page_sources=include_page_sources,
            )
            if current_exists:
                reason = "matched by exact resource-id"
                matched_by = locator_value
                return AssertionCheckResult(True, reason, matched_by)

            elem, found = find_element_with_scroll(driver, resource_id=locator_value)
            if found:
                return AssertionCheckResult(True, "matched by exact resource-id", locator_value)
            return AssertionCheckResult(
                False,
                "target element not found by exact resource-id; do not fabricate non-existent controls, choose the target from actual UI controls",
                locator_value,
            )
        except Exception as exc:
            if _is_session_lost_exception(exc):
                return AssertionCheckResult(False, f"driver session unavailable: {exc}", locator_value, True)
            return AssertionCheckResult(False, f"checker exception: {exc}", locator_value)

    if locator_type == "xpath" and locator_value:
        structured_xpath = _strip_text_predicates(locator_value)
        try:
            current_exists, match_mode = _element_exists_in_snapshot_with_prefix_fallback(
                current_snapshot,
                locator_type,
                structured_xpath,
                include_page_sources=include_page_sources,
            )
            if current_exists:
                reason = "matched by xpath" if match_mode == "exact" else "matched by xpath/class prefix on current screen"
                matched_by = structured_xpath if match_mode == "exact" else f"prefix:{structured_xpath}"
                return AssertionCheckResult(True, reason, matched_by)

            elem, found = find_element_with_scroll(driver, xpath=structured_xpath)
            if found:
                return AssertionCheckResult(True, "matched by xpath", structured_xpath)
            return AssertionCheckResult(
                False,
                "target element not found by xpath/class prefix; do not fabricate non-existent controls, choose the target from actual UI controls",
                structured_xpath,
            )
        except Exception as exc:
            if _is_session_lost_exception(exc):
                return AssertionCheckResult(False, f"driver session unavailable: {exc}", structured_xpath, True)
            return AssertionCheckResult(False, f"checker exception: {exc}", structured_xpath)

    if (
        "check_current_app_package(" in assertion_line
        or "check_dark_mode(" in assertion_line
        or "check_light_mode(" in assertion_line
        or "check_row_checked_by_text(" in assertion_line
    ):
        return AssertionCheckResult(True, "specialized assertion handled elsewhere")

    try:
        exec_globals: dict[str, Any] = {"driver": driver, "By": AppiumBy}
        exec(assertion_line, exec_globals)
        return AssertionCheckResult(True, "executed directly")
    except Exception as exc:
        if _is_session_lost_exception(exc):
            return AssertionCheckResult(False, f"driver session unavailable: {exc}", fatal=True)
        return AssertionCheckResult(False, f"direct execution failed: {exc}")