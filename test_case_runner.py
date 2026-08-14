from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from appium.webdriver.common.appiumby import AppiumBy
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

from action_handlers import click_by_field_value, find_element_with_scroll, hide_keyboard_if_open, swipe_element
from assertion_checker import (
    check_generated_assertion,
    _assertion_is_allowed,
    _describe_format_violation,
    _extract_assertion_target_key,
    _extract_locator,
    _repair_xpath_predicate_quotes,
)
from assertion_generator import generate_phase_assertions
from assertion_handlers import (
    check_current_app_package,
    check_dark_mode,
    check_elements_same_screen,
    check_light_mode,
    check_row_checked_by_text,
)
from classifier import classify_file, extract_task_description
from baselines.andro_b2o_adapter import generate_assertions_for_step as generate_androb2o_assertions_for_step
from intent_planner import plan_phases
from llm_client import load_llm_settings
from snapshot_utils import create_action_output_paths, create_generated_output_paths, capture_step_snapshot, write_json_snapshot


SUPPORTED_ACTION_NAMES = [
    'send_keys_and_hide_keyboard',
    'send_keys_and_enter',
    'click_by_text_contains',
    'long_click_by_text_contains',
    'click_by_content_desc',
    'long_click_by_content_desc',
    'click_by_text',
    'long_click_by_text',
    'swipe_right',
    'swipe_left',
    'long_click',
    'go_home',
    'click',
]

ASSERTION_LLM_SETTINGS = load_llm_settings("ASSERTION")
LLM_BASE_URL = ASSERTION_LLM_SETTINGS.base_url
LLM_API_KEY = ASSERTION_LLM_SETTINGS.api_key
LLM_MODEL = ASSERTION_LLM_SETTINGS.model
MAX_ASSERTION_RETRIES = 3
ACTION_POST_CHANGE_CAPTURE_DELAY = 1
PHASE_END_CAPTURE_DELAY = 3


def _merge_action_snapshots(action_snapshots: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not action_snapshots:
        return None

    last_snapshot = action_snapshots[-1]
    merged_snapshot = dict(last_snapshot)
    merged_ui: dict[str, Any] = dict(last_snapshot.get("ui") or {})

    page_sources: list[str] = []
    raw_page_sources: list[str] = []
    scroll_screenshot_paths: list[str] = []

    for snapshot in action_snapshots:
        ui = snapshot.get("ui") or {}
        if not isinstance(ui, dict):
            continue

        raw_page_source = ui.get("raw_page_source")
        if isinstance(raw_page_source, str) and raw_page_source.strip():
            raw_page_sources.append(raw_page_source.strip())

        snapshot_page_sources = ui.get("page_sources")
        if isinstance(snapshot_page_sources, list):
            for source in snapshot_page_sources:
                if isinstance(source, str) and source.strip():
                    page_sources.append(source.strip())

        snapshot_scroll_paths = ui.get("scroll_screenshot_paths")
        if isinstance(snapshot_scroll_paths, list):
            for path in snapshot_scroll_paths:
                if isinstance(path, str) and path.strip():
                    scroll_screenshot_paths.append(path.strip())

        screenshot_path = snapshot.get("screenshot_path")
        if isinstance(screenshot_path, str) and screenshot_path.strip():
            scroll_screenshot_paths.append(screenshot_path.strip())

    def _dedupe(items: list[str]) -> list[str]:
        seen: set[str] = set()
        deduped: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            deduped.append(item)
        return deduped

    merged_ui["raw_page_source"] = raw_page_sources[0] if raw_page_sources else str(merged_ui.get("raw_page_source") or "")
    merged_ui["page_sources"] = _dedupe(page_sources or raw_page_sources)
    merged_ui["scroll_screenshot_paths"] = _dedupe(scroll_screenshot_paths)
    merged_snapshot["ui"] = merged_ui
    merged_snapshot["step_type"] = "merged"
    merged_snapshot["step_label"] = "all_actions"
    return merged_snapshot


def parse_test_case_sections(file_path: Path) -> tuple[list[str], list[str], list[str]]:
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    header_lines: list[str] = []
    test_steps: list[str] = []
    reached_body = False
    for line in lines:
        if not reached_body and line.strip() == '---':
            reached_body = True
            header_lines.append(line.rstrip('\n'))
            continue
        if not reached_body:
            header_lines.append(line.rstrip('\n'))
        else:
            if not line.startswith('appPackage:') and not line.startswith('appActivity:') and not line.startswith('---'):
                test_steps.append(line)

    action_descriptions: list[str] = []
    for raw_line in test_steps:
        stripped_line = raw_line.strip()
        match = re.match(r'^\d+\. \((action|assertion)\) (.*)', stripped_line)
        if match and match.group(1) == 'action':
            action_descriptions.append(match.group(2))

    return header_lines, test_steps, action_descriptions


def _debug_log(message: str):
    print(f"[ASSERTION_DEBUG] {message}")


def extract_locator_from_assertion(content):
    locator = None
    end = None

    if 'By.XPATH' in content:
        start = content.find('By.XPATH')
        comma = content.find(',', start)
        end = content.find(')', comma)
        if end == -1:
            end = content.find(');', comma)
        raw = content[comma + 1:end].strip() if comma != -1 and end != -1 else ''
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            xpath_val = raw[1:-1]
        else:
            xpath_val = raw
        locator = ('XPATH', xpath_val)

    elif 'By.ID' in content:
        start = content.find('By.ID')
        comma = content.find(',', start)
        end = content.find(')', comma)
        if end == -1:
            end = content.find(');', comma)
        raw = content[comma + 1:end].strip() if comma != -1 and end != -1 else ''
        if (raw.startswith("'") and raw.endswith("'")) or (raw.startswith('"') and raw.endswith('"')):
            id_val = raw[1:-1]
        else:
            id_val = raw
        locator = ('ID', id_val)

    if locator:
        return locator[0], locator[1], end
    return None, None, None


def extract_single_string_argument(content, func_name):
    prefix = f"{func_name}("
    if not content.startswith(prefix) or not content.endswith(')'):
        return None

    arg = content[len(prefix):-1].strip()
    if (arg.startswith("'") and arg.endswith("'")) or (arg.startswith('"') and arg.endswith('"')):
        return arg[1:-1]
    return arg or None


def extract_string_arguments(content, func_name):
    prefix = f"{func_name}("
    if not content.startswith(prefix) or not content.endswith(')'):
        return []

    raw_args = content[len(prefix):-1].strip()
    if not raw_args:
        return []

    args = []
    current = []
    quote = None
    escape = False

    for ch in raw_args:
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
            arg = ''.join(current).strip()
            if arg:
                if (arg.startswith("'") and arg.endswith("'")) or (arg.startswith('"') and arg.endswith('"')):
                    args.append(arg[1:-1])
                else:
                    args.append(arg)
            current = []
            continue
        current.append(ch)

    arg = ''.join(current).strip()
    if arg:
        if (arg.startswith("'") and arg.endswith("'")) or (arg.startswith('"') and arg.endswith('"')):
            args.append(arg[1:-1])
        else:
            args.append(arg)

    return args


def _build_retry_feedback(failure_reasons: list[str]) -> str:
    if not failure_reasons:
        return ""
    return "\n".join(f"- {reason}" for reason in failure_reasons)


def _is_quote_format_failure_reason(reason: str) -> bool:
    normalized = (reason or "").strip()
    if not normalized:
        return False
    return "引号格式错误" in normalized or "未转义的引号" in normalized


def _normalize_assertion_text_for_insertion(assertion: str) -> str:
    normalized = (assertion or "").strip()
    if not normalized:
        return assertion
    if "By.XPATH" in normalized:
        return _repair_xpath_predicate_quotes(normalized)
    return normalized


def _apply_static_quote_escape_fallback(assertions: list[str]) -> list[str]:
    repaired: list[str] = []
    seen: set[str] = set()
    for assertion in assertions:
        candidate = _normalize_assertion_text_for_insertion(assertion)
        if candidate in seen:
            continue
        seen.add(candidate)
        repaired.append(candidate)
    return repaired


def normalize_action_params(action_name, params_raw):
    if not params_raw:
        return []

    if action_name in {
        'send_keys_and_hide_keyboard',
        'send_keys_and_enter',
        'click_by_text',
        'long_click_by_text',
        'click_by_text_contains',
        'long_click_by_text_contains',
        'click_by_content_desc',
        'long_click_by_content_desc',
    }:
        return [params_raw.strip()]

    return [p.strip() for p in params_raw.split(',') if p.strip()]


def _allowed_assertions_for_category(category: int | None) -> list[str]:
    common = [
        'check_row_checked_by_text',
        'elem = driver.find_element(By.XPATH, ...); assert elem.is_displayed()',
        'elem = driver.find_element(By.XPATH, ...); assert elem.text == ...',
        'elem = driver.find_element(By.XPATH, ...); assert len(elems) == 0',
        'elem = driver.find_element(By.ID, ...); assert elem.is_displayed()',
        'elem = driver.find_element(By.ID, ...); assert elem.text == ...',
        'elem = driver.find_element(By.ID, ...); assert len(elems) == 0',
    ]
    category2_only = ['check_elements_same_screen']
    category1_only = ['check_dark_mode', 'check_light_mode']
     # 当 category 为 None（即 no-classifier 消融）时，返回全部断言类型
    if category is None:
        return category1_only + category2_only + ['check_current_app_package'] + common
    if category == 1:
        return category1_only + common
    if category == 2:
        return category2_only + common
    if category in (3, 4):
        return ['check_current_app_package'] + common
    return common


def _render_output_test_case(header_lines: list[str], steps: list[str]) -> str:
    lines = [*header_lines]
    lines.extend(steps)
    return '\n'.join(lines).rstrip() + '\n'


def _renumber_steps(raw_steps: list[str]) -> list[str]:
    renumbered = []
    for index, step in enumerate(raw_steps, 1):
        renumbered.append(re.sub(r'^\d+\.', f'{index}.', step, count=1))
    return renumbered


def _format_assertion_line(step_number: int, assertion: str) -> str:
    return f"{step_number}. (assertion) {assertion}"


def _step_content_key(step_line: str) -> str:
    normalized = step_line.strip()
    match = re.match(r'^\d+\. \((action|assertion)\) (.*)$', normalized)
    if match:
        return re.sub(r'\s+', ' ', match.group(2).strip())
    return re.sub(r'\s+', ' ', normalized)


def _normalize_assertion_text_for_insertion(assertion: str) -> str:
    normalized = re.sub(r'\s+', ' ', (assertion or '').strip())
    if not normalized:
        return normalized
    if 'By.XPATH' in normalized:
        return _repair_xpath_predicate_quotes(normalized)
    return normalized


def _build_retry_feedback(failure_reasons: list[str]) -> str:
    format_failures = [reason for reason in failure_reasons if reason.startswith("格式违规：")]
    locator_failures = [reason for reason in failure_reasons if reason.startswith("定位失败：")]

    lines: list[str] = []
    if format_failures:
        lines.append("【格式校验失败】")
        for reason in format_failures:
            assertion_part = reason[len("格式违规："):].strip()
            format_detail = _describe_format_violation(assertion_part.strip("'\""))
            lines.append(f"- {assertion_part} -> {format_detail}")

    if locator_failures:
        lines.append("【元素定位失败】")
        for reason in locator_failures:
            locator_detail = reason[len('定位失败：'):].strip()
            lines.append(f"- {locator_detail}")
            if "引号格式错误" in locator_detail or "未转义的引号" in locator_detail:
                lines.append("  -> 这是引号格式问题，请把 XPath 内部文本中的单引号改成双引号，或对内部引号进行转义后再生成")

    if not lines:
        return "No valid assertions were produced. Please return only strictly valid assertions."
    return "\n".join(lines)


def _is_quote_format_failure_reason(reason: str) -> bool:
    return "引号格式错误" in reason or "未转义的引号" in reason


def _apply_static_quote_escape_fallback(assertions: list[str]) -> list[str]:
    escaped_assertions: list[str] = []
    seen: set[str] = set()
    for assertion in assertions:
        escaped = _normalize_assertion_text_for_insertion(assertion)
        normalized = escaped
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        escaped_assertions.append(escaped)
    return escaped_assertions


def _execute_assertion_line(driver, assertion_line: str):
    assertion_line = assertion_line.strip()
    if assertion_line.startswith('check_dark_mode('):
        rid = extract_single_string_argument(assertion_line, 'check_dark_mode')
        return check_dark_mode(driver, rid)
    if assertion_line.startswith('check_light_mode('):
        rid = extract_single_string_argument(assertion_line, 'check_light_mode')
        return check_light_mode(driver, rid)
    if assertion_line.startswith('check_current_app_package('):
        args = extract_string_arguments(assertion_line, 'check_current_app_package')
        if not args:
            raise ValueError('check_current_app_package requires at least one package argument.')
        expected_package = args[0]
        expected_activity = args[1] if len(args) > 1 else None
        return check_current_app_package(driver, expected_package, expected_activity)
    if assertion_line.startswith('check_row_checked_by_text('):
        target_text = extract_single_string_argument(assertion_line, 'check_row_checked_by_text')
        if not target_text:
            raise ValueError('check_row_checked_by_text requires a target text argument.')
        return check_row_checked_by_text(driver, target_text)
    if assertion_line.startswith('check_elements_same_screen('):
        exec_globals = {
            'driver': driver,
            'By': AppiumBy,
            'WebDriverWait': WebDriverWait,
            'EC': EC,
            'check_elements_same_screen': check_elements_same_screen,
            'check_current_app_package': check_current_app_package,
        }
        exec(assertion_line, exec_globals)
        return None
    exec_globals = {
        'driver': driver,
        'By': AppiumBy,
        'WebDriverWait': WebDriverWait,
        'EC': EC,
        'check_elements_same_screen': check_elements_same_screen,
        'check_current_app_package': check_current_app_package,
    }
    if 'find_elements' in assertion_line or 'find_element' in assertion_line:
        exec(assertion_line, exec_globals)
        return None
    exec(assertion_line, exec_globals)
    return None


def _build_phase_mappings(phases: list[dict]) -> tuple[dict[int, dict], set[int]]:
    phase_by_end_action_index: dict[int, dict] = {}
    phase_end_steps: set[int] = set()
    for phase in phases:
        if not isinstance(phase, dict):
            continue
        action_range = phase.get("action_range") or []
        if not (isinstance(action_range, list) and len(action_range) == 2):
            continue
        start, end = action_range
        if isinstance(start, int) and isinstance(end, int) and start >= 1 and end >= start:
            phase_by_end_action_index[end] = phase
            phase_end_steps.add(end)
    return phase_by_end_action_index, phase_end_steps


def _handle_phase_assertions(
    driver,
    phase_info: dict,
    phase_start_snapshot: dict | None,
    phase_end_snapshot: dict,
    snapshot_history: list[dict],
    task_description: str,
    allowed_assertions: list[str],
    result_logger,
    use_control_selector: bool = True,
    use_assertion_checker: bool = True,
    include_page_sources: bool = False,
    dedupe_exact_assertions: bool = False,
) -> tuple[bool, list[str], list[dict[str, str]], int, int | None]:
    quote_retry_attempted = False
    pending_feedback: str | None = None
    collected_valid_assertions: list[str] = []
    collected_selected_targets: list[dict[str, str]] = []
    selected_insert_after_step: int | None = None
    total_token_usage = 0
    last_generation: Any | None = None
    last_failure_reasons: list[str] = []
    last_quote_failed_assertions: list[str] = []
    for attempt in range(1, MAX_ASSERTION_RETRIES + 1):
        _debug_log(
            f"Generating phase assertions: attempt={attempt}, phase_id={phase_info.get('phase_id')}, "
            f"action_range={phase_info.get('action_range')}, intent={phase_info.get('intent')!r}"
        )
        generation = generate_phase_assertions(
            phase_info=phase_info,
            phase_start_snapshot=phase_start_snapshot,
            phase_end_snapshot=phase_end_snapshot,
            allowed_assertions=allowed_assertions,
            task_description=task_description,
            use_control_selector=use_control_selector,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
            feedback=pending_feedback,
        )

        _debug_log(
            f"Phase assertion generation result: reason={generation.reason!r}, count={len(generation.assertions)}"
        )
        last_generation = generation
        total_token_usage += getattr(generation, "token_usage", 0) or 0
        if result_logger:
            result_logger.write(
                f"\n[phase {phase_info.get('phase_id')}] generation attempt {attempt}: {len(generation.assertions)} assertion(s)\n"
            )
            if generation.selected_targets:
                result_logger.write(
                    "[phase-selected-targets] "
                    + json.dumps(generation.selected_targets, ensure_ascii=False)
                    + "\n"
                )

        if not generation.assertions:
            return True, collected_valid_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, generation.insert_after_step

        if not use_assertion_checker:
            for assertion in generation.assertions:
                if assertion not in collected_valid_assertions:
                    collected_valid_assertions.append(assertion)
            if generation.selected_targets:
                for target in generation.selected_targets:
                    if target not in collected_selected_targets:
                        collected_selected_targets.append(target)
            return True, collected_valid_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, generation.insert_after_step

        valid_assertions: list[str] = []
        failure_reasons: list[str] = []
        seen_target_keys: set[str] = set()
        seen_assertion_texts: set[str] = set()
        current_quote_failed_assertions: list[str] = []
        for assertion in generation.assertions:
            normalized_assertion = re.sub(r"\s+", " ", assertion.strip())
            if dedupe_exact_assertions:
                if normalized_assertion in seen_assertion_texts:
                    _debug_log(f"Skipping duplicate exact assertion before validation: {assertion!r}")
                    continue
                seen_assertion_texts.add(normalized_assertion)
                dedup_key = None   # 这里不会使用，但保留变量一致性
            else:
                dedup_key = None   # 用于存储本次断言的目标键（raw_key 或 target_key）
                # 针对 find_element 断言，优先使用原始定位字符串精确去重
                locator_type, locator_value = _extract_locator(assertion)
                if locator_type and locator_value:
                    raw_key = f"{locator_type}:{locator_value}"
                    if raw_key in seen_target_keys:
                        _debug_log(f"Skipping duplicate assertion for same locator: {assertion!r} (raw_key={raw_key!r})")
                        continue
                    dedup_key = raw_key
                else:
                    # 非 find_element 断言（如 check_*），使用原有的目标键去重
                    target_key = _extract_assertion_target_key(assertion)
                    if target_key and target_key in seen_target_keys:
                        _debug_log(f"Skipping duplicate assertion for same target: {assertion!r} (key={target_key!r})")
                        continue
                    dedup_key = target_key   # 可能为 None

            # 检查断言格式是否允许
            if not _assertion_is_allowed(assertion, allowed_assertions):
                _debug_log(f"Assertion rejected by category policy: {assertion!r}")
                failure_reasons.append(f"格式违规：{assertion!r}")
                continue

            # 执行断言检查（定位）
            check_result = check_generated_assertion(
                driver,
                assertion,
                snapshot_history=snapshot_history,
                current_snapshot=phase_end_snapshot,
                task_description=task_description,
                include_page_sources=include_page_sources,
            )
            _debug_log(
                "Phase assertion checker result: "
                f"passed={check_result.passed}, reason={check_result.reason!r}, matched_by={check_result.matched_by!r}"
            )

            if check_result.passed:
                valid_assertions.append(assertion)
                # 只有验证通过后，才将本次的键加入已用集合，防止失败断言阻塞后续重试
                if dedup_key:
                    seen_target_keys.add(dedup_key)
            else:
                if "引号格式错误" in check_result.reason or "未转义的引号" in check_result.reason:
                    failure_reasons.append(f"格式违规：{assertion!r} -> {check_result.reason}")
                    current_quote_failed_assertions.append(assertion)
                else:
                    failure_reasons.append(f"定位失败：{assertion!r} -> {check_result.reason}")

        last_failure_reasons = failure_reasons
        last_quote_failed_assertions = current_quote_failed_assertions

        if valid_assertions:
            for assertion in valid_assertions:
                if assertion not in collected_valid_assertions:
                    collected_valid_assertions.append(assertion)
            if generation.selected_targets:
                for target in generation.selected_targets:
                    if target not in collected_selected_targets:
                        collected_selected_targets.append(target)

        if valid_assertions and not failure_reasons:
            selected_insert_after_step = generation.insert_after_step if isinstance(generation.insert_after_step, int) else selected_insert_after_step
            return True, collected_valid_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, selected_insert_after_step

        if valid_assertions and failure_reasons:
            # 部分成功：先保留成功的断言，失败的断言继续带着反馈重试。
            # 只要还有未通过的断言，就继续尝试下一轮生成。
            pending_feedback = _build_retry_feedback(failure_reasons + [f"已通过断言：{a!r}" for a in valid_assertions])
            quote_only_failures = bool(failure_reasons) and all(_is_quote_format_failure_reason(reason) for reason in failure_reasons)
            if quote_only_failures:
                if quote_retry_attempted:
                    escaped_failed_assertions = _apply_static_quote_escape_fallback(last_quote_failed_assertions)
                    final_assertions = list(collected_valid_assertions)
                    for assertion in escaped_failed_assertions:
                        if assertion not in final_assertions:
                            final_assertions.append(assertion)
                    if final_assertions:
                        _debug_log(
                            "Phase assertions hit quote-format errors twice; applying static escape fallback immediately."
                        )
                        if result_logger:
                            result_logger.write("[quote-fallback] applied static escape fallback after one retry\n")
                        selected_insert_after_step = generation.insert_after_step if isinstance(generation.insert_after_step, int) else selected_insert_after_step
                        return True, final_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, selected_insert_after_step
                quote_retry_attempted = True
                print(f"[warning] Phase assertions partially passed on attempt {attempt}; retrying quote-format failures once.")
                continue
            if attempt < MAX_ASSERTION_RETRIES:
                print(f"[warning] Phase assertions partially passed on attempt {attempt}; retrying failed ones.")
                continue
            if quote_only_failures:
                escaped_failed_assertions = _apply_static_quote_escape_fallback(last_quote_failed_assertions)
                final_assertions = list(collected_valid_assertions)
                for assertion in escaped_failed_assertions:
                    if assertion not in final_assertions:
                        final_assertions.append(assertion)
                if final_assertions:
                    _debug_log(
                        "Phase assertions exhausted retries with quote-format issues; applying static escape fallback."
                    )
                    if result_logger:
                        result_logger.write("[quote-fallback] applied static escape fallback after max retries\n")
                    selected_insert_after_step = generation.insert_after_step if isinstance(generation.insert_after_step, int) else selected_insert_after_step
                    return True, final_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, selected_insert_after_step
            selected_insert_after_step = generation.insert_after_step if isinstance(generation.insert_after_step, int) else selected_insert_after_step
            return True, collected_valid_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, selected_insert_after_step

        if attempt < MAX_ASSERTION_RETRIES:
            pending_feedback = _build_retry_feedback(failure_reasons)
            if failure_reasons and all(_is_quote_format_failure_reason(reason) for reason in failure_reasons):
                if quote_retry_attempted:
                    escaped_failed_assertions = _apply_static_quote_escape_fallback(last_quote_failed_assertions)
                    final_assertions = list(collected_valid_assertions)
                    for assertion in escaped_failed_assertions:
                        if assertion not in final_assertions:
                            final_assertions.append(assertion)
                    if final_assertions:
                        _debug_log(
                            "Phase assertions hit quote-format errors twice; applying static escape fallback immediately."
                        )
                        if result_logger:
                            result_logger.write("[quote-fallback] applied static escape fallback after one retry\n")
                        final_insert_after_step = (
                            generation.insert_after_step
                            if isinstance(generation.insert_after_step, int)
                            else selected_insert_after_step
                        )
                        return True, final_assertions, collected_selected_targets or generation.selected_targets or [], total_token_usage, final_insert_after_step
                quote_retry_attempted = True
                print(f"[warning] Phase assertions failed validation on attempt {attempt}; retrying quote-format failures once.")
            else:
                print(f"[warning] Phase assertions failed validation on attempt {attempt}; retrying.")
        else:
            pending_feedback = None

    if last_failure_reasons and all(_is_quote_format_failure_reason(reason) for reason in last_failure_reasons):
        escaped_failed_assertions = _apply_static_quote_escape_fallback(last_quote_failed_assertions)
        final_assertions = list(collected_valid_assertions)
        for assertion in escaped_failed_assertions:
            if assertion not in final_assertions:
                final_assertions.append(assertion)
        if final_assertions:
            _debug_log("Phase assertions exhausted retries with quote-format issues; applying static escape fallback.")
            if result_logger:
                result_logger.write("[quote-fallback] applied static escape fallback after max retries\n")
            final_insert_after_step = (
                last_generation.insert_after_step
                if isinstance(last_generation.insert_after_step, int)
                else selected_insert_after_step
            ) if last_generation is not None else selected_insert_after_step
            return True, final_assertions, collected_selected_targets, total_token_usage, final_insert_after_step

    return bool(collected_valid_assertions), collected_valid_assertions, collected_selected_targets, total_token_usage, selected_insert_after_step


def _insert_generated_assertions(
    generated_steps: list[tuple[int, str | None, str]],
    action_step_index: int,
    assertions: list[str],
    phase_id: str | int | None = None,
    log_prefix: str = "Inserted assertion",
) -> None:
    for assertion in assertions:
        assertion_step = _format_assertion_line(
            action_step_index + 1 + len([x for x in generated_steps if x[0] == action_step_index]),
            assertion,
        )
        generated_steps.append((action_step_index, phase_id, assertion_step))
        print(f"{log_prefix} after step {action_step_index}: {assertion}")
        _debug_log(f"{log_prefix} after step {action_step_index}: {assertion}")


def _resolve_swipe_target(driver, element, xpath: str):
    """Prefer swiping a clickable/ancestor container instead of a leaf text node."""
    xpath = (xpath or '').strip()

    candidates = []
    if xpath:
        candidates.extend([
            f"({xpath})/ancestor::*[@clickable='true'][1]",
            f"({xpath})/parent::*[1]",
            f"({xpath})/ancestor::*[1]",
        ])

    for candidate_xpath in candidates:
        try:
            target = driver.find_element(AppiumBy.XPATH, candidate_xpath)
            if target:
                return target, candidate_xpath
        except Exception:
            continue

    return element, None


def execute_test_case(
    driver,
    file_path,
    result_logger,
    generate_assertions: bool = True,
    artifact_root: str | None = None,
    parsed_sections: tuple[list[str], list[str], list[str]] | None = None,
    use_intent_planner: bool = True,
    use_control_selector: bool = True,
    use_assertion_checker: bool = True,
    use_androb2o_baseline: bool = False,
    use_classifier: bool = True,
):
    if use_androb2o_baseline:
        use_intent_planner = False
        use_control_selector = False

    test_case_path = Path(file_path)
    test_case_stem, screenshot_dir, json_path = create_action_output_paths(file_path, artifact_root=artifact_root)
    _, output_txt_path, output_json_path = create_generated_output_paths(file_path, artifact_root=artifact_root)
    case = extract_task_description(test_case_path)
    category = classify_file(test_case_path) if use_classifier else None
    allowed_assertions = _allowed_assertions_for_category(category)

    _debug_log(f"Starting test case: {test_case_path}")
    if use_classifier:
        _debug_log(f"Classified category: {category}")
    else:
        _debug_log("Classifier disabled; using category=None and generic allowed assertions.")
    _debug_log(f"TaskDescription: {case.task_description}")
    _debug_log(f"Allowed assertions for this category: {allowed_assertions}")

    total_steps = 0
    passed_steps = 0
    action_snapshots = []
    generated_steps: list[tuple[int, str | None, str]] = []
    generated_phase_debug: list[dict[str, Any]] = []
    total_token_usage = 0
    action_name_regex = '|'.join(re.escape(name) for name in sorted(SUPPORTED_ACTION_NAMES, key=len, reverse=True))
    action_step_index = 0

    if parsed_sections is None:
        header_lines, test_steps, all_action_descriptions = parse_test_case_sections(test_case_path)
    else:
        header_lines, test_steps, all_action_descriptions = parsed_sections

    total_action_count = len(all_action_descriptions)

    phase_by_end_action_index: dict[int, dict] = {}
    phase_end_steps: set[int] = set()
    if generate_assertions and use_intent_planner and not use_androb2o_baseline:
        phase_plan = plan_phases(
            task_description=case.task_description,
            category=category,
            action_descriptions=all_action_descriptions,
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            model=LLM_MODEL,
        )
        total_token_usage += getattr(phase_plan, "token_usage", 0) or 0
        phase_by_end_action_index, phase_end_steps = _build_phase_mappings(phase_plan.phases)
        _debug_log(f"Planned phases: {phase_plan.phases}")
        _debug_log(f"Phase end action indexes: {sorted(phase_end_steps)}")

    total_steps = len([line for line in test_steps if line.strip()])
    _debug_log(f"Parsed {total_steps} executable step(s) from input file.")

    _debug_log(
        "Collected full action sequence for locator context: "
        f"count={len(all_action_descriptions)}, actions={all_action_descriptions}"
    )

    last_processed_step_index = 0
    last_action_step_index = 0

    for i, line in enumerate(test_steps, 1):
        line = line.strip()
        if not line:
            continue
        last_processed_step_index = i

        print(f"Executing step {i}: {line}")
        _debug_log(f"Executing step {i}: {line}")
        result_logger.write(f"Step {i}: {line} - ")

        match = re.match(r'^\d+\. \((action|assertion)\) (.*)', line)
        if not match:
            print(f"Warning: Could not parse line: {line}")
            result_logger.write("SKIPPED (parse error)\n")
            continue

        step_type, content = match.groups()

        try:
            if step_type == 'action':
                action_step_index += 1
                last_action_step_index = i
                action_match = re.match(
                    rf'^resource-id:\s*(?P<resource_id>.*?),\s*xpath:\s*(?P<xpath>.*?),\s*(?P<action>{action_name_regex})(?:,\s*(?P<params>.*))?$',
                    content,
                )
                if not action_match:
                    print(f"Warning: Could not parse action content: {content}")
                    result_logger.write("SKIPPED (action parse error)\n")
                    continue

                resource_id = action_match.group('resource_id').strip()
                xpath = action_match.group('xpath').strip()
                action_name = action_match.group('action').strip()
                params_raw = action_match.group('params')
                params = normalize_action_params(action_name, params_raw)

                if action_name in {'click_by_text', 'click_by_content_desc', 'click_by_text_contains', 'long_click_by_text', 'long_click_by_content_desc', 'long_click_by_text_contains'}:
                    if not params:
                        raise ValueError(f"{action_name} requires a target value.")
                    target_value = params[0]
                    field_name = 'text' if 'content_desc' not in action_name else 'content-desc'
                    exact = 'contains' not in action_name
                    long_press = action_name.startswith('long_click_')
                    clicked = click_by_field_value(driver, target_value, field_name=field_name, exact=exact, long_press=long_press)
                    if not clicked:
                        raise LookupError(f"Target field '{target_value}' was not found before reaching the bottom of the screen.")
                elif action_name == 'go_home':
                    try:
                        driver.press_keycode(3)
                    except Exception:
                        driver.execute_script('mobile: shell', {'command': 'input', 'args': ['keyevent', '3']})
                else:
                    element, found = find_element_with_scroll(driver, resource_id=resource_id, xpath=xpath)
                    if not found:
                        raise LookupError('Target element was not found before reaching the bottom of the screen.')

                    if action_name == 'click':
                        element.click()
                    elif action_name == 'long_click':
                        from selenium.webdriver import ActionChains
                        ActionChains(driver).move_to_element(element).click_and_hold().pause(2).release().perform()
                    elif action_name == 'send_keys_and_hide_keyboard':
                        element.clear()
                        element.send_keys(params[0])
                        hide_keyboard_if_open(driver)
                    elif action_name == 'send_keys_and_enter':
                        element.clear()
                        element.send_keys(params[0])
                        hide_keyboard_if_open(driver)
                        driver.press_keycode(66)
                    elif action_name == 'swipe_right':
                        swipe_target, swipe_target_xpath = _resolve_swipe_target(driver, element, xpath)
                        if swipe_target_xpath:
                            _debug_log(f"swipe_right resolved target xpath: {swipe_target_xpath}")
                        _debug_log(
                            "About to execute swipe_right: "
                            f"resource_id={resource_id!r}, xpath={xpath!r}, "
                            f"element_tag={getattr(swipe_target, 'tag_name', None)!r}, "
                            f"element_text={getattr(swipe_target, 'text', None)!r}, "
                            f"element_location={getattr(swipe_target, 'location', None)!r}, "
                            f"element_size={getattr(swipe_target, 'size', None)!r}"
                        )
                        swipe_element(driver, swipe_target, 400, 0, 200)
                        _debug_log("swipe_right completed without raising an exception.")
                    elif action_name == 'swipe_left':
                        swipe_target, swipe_target_xpath = _resolve_swipe_target(driver, element, xpath)
                        if swipe_target_xpath:
                            _debug_log(f"swipe_left resolved target xpath: {swipe_target_xpath}")
                        _debug_log(
                            "About to execute swipe_left: "
                            f"resource_id={resource_id!r}, xpath={xpath!r}, "
                            f"element_tag={getattr(swipe_target, 'tag_name', None)!r}, "
                            f"element_text={getattr(swipe_target, 'text', None)!r}, "
                            f"element_location={getattr(swipe_target, 'location', None)!r}, "
                            f"element_size={getattr(swipe_target, 'size', None)!r}"
                        )
                        swipe_element(driver, swipe_target, -400, 0, 200)
                        _debug_log("swipe_left completed without raising an exception.")
                    else:
                        print(f"Unknown action: {action_name}")

                # ========== 截图前延迟，区分普通动作、阶段结束动作和 no-planner 最后一步 ==========
                is_final_action_step = action_step_index == total_action_count
                is_phase_end_action = generate_assertions and action_step_index in phase_end_steps
                is_no_planner_final_action = generate_assertions and not use_intent_planner and is_final_action_step

                if is_phase_end_action or is_no_planner_final_action:
                    capture_delay = PHASE_END_CAPTURE_DELAY
                    _debug_log(
                        f"{'No-planner final' if is_no_planner_final_action else 'Phase end'} action at step {action_step_index}, "
                        f"waiting {capture_delay}s before capture"
                    )
                else:
                    capture_delay = ACTION_POST_CHANGE_CAPTURE_DELAY
                time.sleep(capture_delay)

                capture_step_snapshot(
                    driver=driver,
                    screenshot_dir=screenshot_dir,
                    json_snapshots=action_snapshots,
                    test_case_stem=test_case_stem,
                    step_index=i,
                    step_type='action',
                    step_label=action_name,
                    line=line,
                    extra_scrolls=3 if (is_phase_end_action or is_no_planner_final_action) else 0,
                    restore_after_scroll=not (
                        (is_phase_end_action or is_no_planner_final_action)
                        and is_final_action_step
                    ),
                )
                write_json_snapshot(json_path, action_snapshots)

                _debug_log(
                    f"Captured snapshot for step {i}. "
                    f"snapshot_count={len(action_snapshots)}, screenshot_dir={screenshot_dir}, information_json={json_path}"
                )

                current_snapshot = action_snapshots[-1]
                previous_snapshot = action_snapshots[-2] if len(action_snapshots) > 1 else None
                _debug_log(
                    f"Snapshot summary for step {i}: current_step_label={current_snapshot.get('step_label')!r}, "
                    f"previous_step_exists={previous_snapshot is not None}"
                )
                if generate_assertions and use_androb2o_baseline:
                    full_xml = current_snapshot.get("ui", {}).get("raw_page_source", "")
                    if full_xml:
                        try:
                            # 传入 allowed_assertions
                            assertions, phase_tokens = generate_androb2o_assertions_for_step(
                                step_index=i,
                                full_xml=full_xml,
                                task_description=case.task_description,
                                allowed_assertions=allowed_assertions,   # 新增
                            )
                            total_token_usage += phase_tokens
                            if assertions:
                                # 确保 _insert_generated_assertions 传入正确参数
                                _insert_generated_assertions(
                                    generated_steps,
                                    i,                      # action_step_index
                                    assertions,             # assertions 列表
                                    phase_id="androb2o",
                                    log_prefix="[AndroB2O] Inserted assertion",
                                )
                        except Exception as e:
                            result_logger.write(f"[AndroB2O] Error: {e}\n")
                elif generate_assertions and use_intent_planner:
                    if action_step_index in phase_end_steps:
                        phase_info = phase_by_end_action_index.get(action_step_index)
                        if phase_info:
                            phase_range = phase_info.get("action_range") or []
                            phase_start_index = phase_range[0] if isinstance(phase_range, list) and len(phase_range) == 2 else action_step_index
                            phase_start_snapshot = None
                            if isinstance(phase_start_index, int) and 1 <= phase_start_index <= len(action_snapshots):
                                phase_start_snapshot = action_snapshots[phase_start_index - 1]
                            success, assertions, selected_targets, phase_tokens, _ = _handle_phase_assertions(
                                driver=driver,
                                phase_info=phase_info,
                                phase_start_snapshot=phase_start_snapshot,
                                phase_end_snapshot=current_snapshot,
                                snapshot_history=action_snapshots[:-1],
                                task_description=case.task_description,
                                allowed_assertions=allowed_assertions,
                                result_logger=result_logger,
                                use_control_selector=use_control_selector,
                                use_assertion_checker=use_assertion_checker,
                                include_page_sources=True,
                                dedupe_exact_assertions=False,
                            )
                            total_token_usage += phase_tokens
                            if not success:
                                _debug_log(f"Phase assertion generation failed permanently at action step {action_step_index}.")
                                break

                            generated_phase_debug.append({
                                "phase_id": phase_info.get("phase_id"),
                                "action_range": phase_info.get("action_range"),
                                "intent": phase_info.get("intent"),
                                "selected_targets": selected_targets,
                                "assertions": assertions,
                            })

                            _insert_generated_assertions(generated_steps, i, assertions, phase_id=phase_info.get("phase_id"))
                        else:
                            _debug_log(f"No phase info found for action step {action_step_index}.")
                else:
                    _debug_log(f"Assertion generation skipped for step {i} (verification mode).")

                result_logger.write('SUCCESS\n')
                passed_steps += 1

            elif step_type == 'assertion':
                if generate_assertions:
                    _debug_log(
                        f"Skipping execution of assertion step {i} in assertion-generation mode: {content!r}"
                    )
                    result_logger.write('SKIPPED (generation mode)\n')
                    continue

                _execute_assertion_line(driver, content)
                time.sleep(ACTION_POST_CHANGE_CAPTURE_DELAY)
                capture_step_snapshot(
                    driver=driver,
                    screenshot_dir=screenshot_dir,
                    json_snapshots=action_snapshots,
                    test_case_stem=test_case_stem,
                    step_index=i,
                    step_type='assertion',
                    step_label=content,
                    line=line,
                )
                write_json_snapshot(json_path, action_snapshots)
                _debug_log(f"Captured assertion snapshot for step {i}; snapshot_count={len(action_snapshots)}")
                result_logger.write('SUCCESS\n')
                passed_steps += 1

            print(f"Step {i} executed successfully.")

        except Exception as e:
            print(f"Error executing step {i}: {line}")
            print(f"Exception: {e}")
            _debug_log(f"Step {i} failed with exception: {e}")
            result_logger.write(f"FAIL: {e}\n")
            driver.save_screenshot(f'error_screenshot_step_{i}.png')
            print(f"Screenshot saved to 'error_screenshot_step_{i}.png'")
            break

    if generate_assertions and not use_intent_planner and not use_androb2o_baseline and action_snapshots:
        merged_snapshot = _merge_action_snapshots(action_snapshots)
        if merged_snapshot is None:
            _debug_log("Planner-ablated mode could not build a merged snapshot; skipping final assertion generation.")
        else:
            candidate_insertion_steps: list[dict[str, Any]] = []
            for step_index, snapshot in enumerate(action_snapshots, 1):
                candidate_insertion_steps.append({
                    "step_index": step_index,
                    "step_label": snapshot.get("step_label"),
                    "step_type": snapshot.get("step_type"),
                    "current_package": snapshot.get("current_package"),
                    "current_activity": snapshot.get("current_activity"),
                })

            phase_info = {
                "phase_id": 1,
                "action_range": [1, total_action_count],
                "intent": case.task_description,
                "candidate_insertion_steps": candidate_insertion_steps,
            }
            phase_start_snapshot = action_snapshots[0] if action_snapshots else None
            success, assertions, selected_targets, phase_tokens, insert_after_step = _handle_phase_assertions(
                driver=driver,
                phase_info=phase_info,
                phase_start_snapshot=phase_start_snapshot,
                phase_end_snapshot=merged_snapshot,
                snapshot_history=action_snapshots[:-1],
                task_description=case.task_description,
                allowed_assertions=allowed_assertions,
                result_logger=result_logger,
                use_control_selector=use_control_selector,
                use_assertion_checker=use_assertion_checker,
                include_page_sources=True,
                dedupe_exact_assertions=True,
            )
            total_token_usage += phase_tokens
            if success and assertions:
                generated_phase_debug.append({
                    "phase_id": phase_info.get("phase_id"),
                    "action_range": phase_info.get("action_range"),
                    "intent": phase_info.get("intent"),
                    "selected_targets": selected_targets,
                    "assertions": assertions,
                    "ablation_mode": "no_intent_planner",
                    "insert_after_step": insert_after_step,
                })
                insertion_step = insert_after_step
                if not isinstance(insertion_step, int):
                    insertion_step = last_action_step_index or last_processed_step_index or len(test_steps)
                _insert_generated_assertions(
                    generated_steps,
                    insertion_step,
                    assertions,
                    phase_id=phase_info.get("phase_id"),
                    log_prefix="Planner-ablated assertion inserted",
                )
            elif not success:
                _debug_log("Planner-ablated assertion generation failed permanently at end of test case.")

    if generate_assertions:
        # 构建最终输出（包含你的去重逻辑）
        final_steps = []
        generated_map: dict[int, list[tuple[str | None, str]]] = {}
        for action_step_index, phase_id, assertion_step in generated_steps:
            generated_map.setdefault(action_step_index, []).append((phase_id, assertion_step))

        existing_assertion_keys = {
            _step_content_key(line)
            for line in test_steps
            if re.match(r'^\d+\. \(assertion\) ', line.strip())
        }
        emitted_generated_keys_by_phase: dict[str | None, set[str]] = {}

        for line in test_steps:
            final_steps.append(line.rstrip())
            match = re.match(r'^(\d+)\. \((action|assertion)\)', line.strip())
            if match:
                step_num = int(match.group(1))
                inserted_list = generated_map.get(step_num, [])
                if inserted_list:
                    for phase_id, inserted in inserted_list:
                        inserted_key = _step_content_key(inserted)
                        emitted_for_phase = emitted_generated_keys_by_phase.setdefault(phase_id, set())

                        if use_androb2o_baseline:
                            # AndroB2O 模式：不跳过去重
                            final_steps.append(inserted)
                            emitted_for_phase.add(inserted_key)
                            continue

                        if inserted_key in existing_assertion_keys or inserted_key in emitted_for_phase:
                            _debug_log(f"Skipping duplicate generated assertion in final output: {inserted!r}")
                            continue
                        final_steps.append(inserted)
                        emitted_for_phase.add(inserted_key)

        final_steps = _renumber_steps(final_steps)

        output_text = _render_output_test_case(header_lines, final_steps)
        Path(output_txt_path).write_text(output_text, encoding='utf-8')
        output_payload = {
            'category': category,
            'classifier_enabled': use_classifier,
            'task_description': case.task_description,
            'input_file': str(test_case_path),
            'output_text': str(output_txt_path),
            'allowed_assertions': allowed_assertions,
            'phase_generation_debug': generated_phase_debug,
        }
        Path(output_json_path).write_text(json.dumps(output_payload, ensure_ascii=False, indent=2), encoding='utf-8')

        _debug_log(f"Final output written to: {output_txt_path}")
        _debug_log(f"Output metadata written to: {output_json_path}")

    # ===== 关键：函数末尾统一返回 =====
    return total_steps, passed_steps, total_token_usage