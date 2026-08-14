# -*- coding: utf-8 -*-
"""AndroB2O baseline adapter – generalized for task-oriented assertions.

输入压缩机制，确保每次 LLM 调用总输入 <= 10KB，
以缓解本地 Qwen3-VL 模型因长输入导致的 CUDA OOM 问题。
"""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from typing import Optional, List, Tuple

from llm_client import (
    get_openai_chat_completion,
    load_llm_settings,
    make_openai_client,
    extract_openai_token_usage,
)

# 复用主框架中的控件精简函数（与 control_selector 逻辑一致）
try:
    from assertion_generator import _build_compact_ui_controls, _serialize_compact_controls
except ImportError:
    # 防御性回退（实际运行中不会触发）
    def _build_compact_ui_controls(xml, drop_container_controls=False):
        return []
    def _serialize_compact_controls(controls):
        return ""

# 常量：单次 LLM 输入预算（10 KB）
MAX_INPUT_BYTES = 10 * 1024


def _log(message: str) -> None:
    print(f"[AndroB2O] {message}")


# ---------- 新增：输入压缩函数 ----------
def _compress_xml_for_prompt(raw_xml: str) -> str:
    """
    对原始 XML 进行逐步压缩，确保最终字符串长度 <= 10KB。
    步骤：
      1. 仅保留每个控件的 xpath, resource-id, class, text, content-desc 五个关键属性。
      2. 若仍超限，删除容器控件（与主框架 control_selector 逻辑一致）。
      3. 若仍超限，后台打印警告，返回当前最精简的结果。
    """
    if not raw_xml:
        return ""

    raw_bytes = len(raw_xml.encode('utf-8'))
    if raw_bytes <= MAX_INPUT_BYTES:
        return raw_xml

    # 步骤 1：精简为五个关键属性
    try:
        controls = _build_compact_ui_controls(raw_xml, drop_container_controls=False)
        compact = _serialize_compact_controls(controls)
        if len(compact.encode('utf-8')) <= MAX_INPUT_BYTES:
            _log(f"XML 已压缩至 {len(compact.encode('utf-8'))} 字节（仅保留 5 个关键属性）。")
            return compact
    except Exception as e:
        _log(f"警告：按属性精简 XML 失败：{e}")

    # 步骤 2：进一步删除容器控件
    try:
        controls = _build_compact_ui_controls(raw_xml, drop_container_controls=True)
        compact = _serialize_compact_controls(controls)
        if len(compact.encode('utf-8')) <= MAX_INPUT_BYTES:
            _log(f"XML 已压缩至 {len(compact.encode('utf-8'))} 字节（删除容器控件后）。")
            return compact
    except Exception as e:
        _log(f"警告：删除容器控件时失败：{e}")

    # 步骤 3：仍超限，打印警告并返回当前最精简结果
    final_bytes = len(compact.encode('utf-8')) if 'compact' in locals() else len(raw_xml.encode('utf-8'))
    _log(f"[WARNING] AndroB2O 输入经全部压缩后仍超过 10KB（{final_bytes} 字节）。")
    return compact if 'compact' in locals() else raw_xml


# ---------- 辅助函数：分块提示 ----------
def _build_split_messages(header: str, chunks: list[str], final_prompt: str) -> list[str]:
    messages = [header]
    for idx, chunk in enumerate(chunks, start=1):
        if idx < len(chunks):
            messages.append(
                f'Do not answer yet. Just acknowledge this part with "Part {idx}/{len(chunks)} received".\n'
                f"[START PART {idx}/{len(chunks)}]{chunk}[END PART {idx}/{len(chunks)}]"
            )
        else:
            messages.append(
                f"[START PART {idx}/{len(chunks)}]{chunk}[END PART {idx}/{len(chunks)}]\n"
                f"ALL PARTS SENT. {final_prompt}"
            )
    return messages


def split_prompt_element(n: int, xml_content: str, task_description: str) -> list[str]:
    header = (
        "The total length of the content that I want to send you is too large to send in only one piece.\n"
        "For sending you that content, I will follow this rule:\n"
        "[START PART 1/10]\nthis is the content of the part 1 out of 10 in total\n"
        "[END PART 1/10]\nThen you just answer: \"Received part 1/10\"\n"
        "And when I tell you \"ALL PARTS SENT\", then you can continue processing the data and answering my requests."
    )
    chunks = [xml_content[i : i + n] for i in range(0, len(xml_content), n)] or [""]
    final_prompt = (
        f"Given the previous XML view hierarchy of the current screen and the following task description:\n"
        f'"{task_description}"\n'
        "First, decide whether this screen is a point where the task expects some verification. "
        "A verification point is a moment when the test should check if a certain condition is met "
        "(e.g., a new task appears, a page has loaded, a setting is enabled, the app has navigated to another app, etc.). "
        "Even if the task description does not contain explicit keywords like 'check' or 'verify', "
        "it may still imply a verification (e.g., 'then you will see...', 'the screen should show...', 'the app should redirect to...'). "
        "If you judge that an assertion should be inserted after this step, output:\n"
        "INSERT_ASSERTION: YES\n"
        "NODE 1:\nXML: [exact XML node of the UI element that is relevant to the verification, if any; if no specific element, output NODE: NONE]\n"
        "If you judge that no assertion is needed for this step, output only:\n"
        "INSERT_ASSERTION: NO"
    )
    return _build_split_messages(header, chunks, final_prompt)


def split_prompt_assertion(
    n: int,
    xml_content: str,
    task_description: str,
    node_xml: Optional[str],
    allowed_assertions: List[str],
) -> list[str]:
    header = (
        "The total length of the content that I want to send you is too large to send in only one piece.\n"
        "For sending you that content, I will follow this rule:\n"
        "[START PART 1/10]\nthis is the content of the part 1 out of 10 in total\n"
        "[END PART 1/10]\nThen you just answer: \"Received part 1/10\"\n"
        "And when I tell you \"ALL PARTS SENT\", then you can continue processing the data and answering my requests."
    )
    chunks = [xml_content[i : i + n] for i in range(0, len(xml_content), n)] or [""]
    node_info = f"The following XML node has been identified as relevant to the verification:\n{node_xml}" if node_xml else "No specific UI element was identified; you may need to infer the target from the task description."
    allowed_list = "\n".join(f"- {a}" for a in allowed_assertions)

    final_prompt = (
        f"Given the previous XML view hierarchy and the following task description:\n"
        f'"{task_description}"\n'
        f"{node_info}\n"
        "Now, generate a **single Python assertion** that verifies the expected condition. "
        "The assertion must use the Appium driver object 'driver' and the By class (imported as By).\n"
        "You must choose one of the following allowed assertion patterns:\n"
        f"{allowed_list}\n"
        "Construct an appropriate locator (XPath or resource-id) based on the available UI evidence. "
        "If the verification involves the current app package, you may use check_current_app_package. "
        "If it involves dark/light mode, use check_dark_mode / check_light_mode. "
        "If it involves a checked row, use check_row_checked_by_text. "
        "If it involves multiple elements on the same screen, use check_elements_same_screen. "
        "Use double quotes for XPath strings, and single quotes for attribute values inside the XPath.\n"
        "Return only the assertion code, no extra explanation or markdown.\n"
        "Now, generate the assertion:"
    )
    return _build_split_messages(header, chunks, final_prompt)


# ---------- 辅助解析与模板函数 ----------
def _parse_element_response(text: str) -> Tuple[bool, Optional[str]]:
    """Returns (should_insert, node_xml). node_xml may be None if no specific element."""
    if not text:
        return False, None
    text = text.strip()
    text = re.sub(r"^```(?:xml|plaintext)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)

    should_insert = False
    node_xml = None

    insert_match = re.search(r"INSERT_ASSERTION\s*:\s*(YES|NO)", text, re.IGNORECASE)
    if insert_match and insert_match.group(1).upper() == "YES":
        should_insert = True

    lines = text.splitlines()
    for line in lines:
        if "XML:" in line:
            idx = line.find("XML:")
            node_xml = line[idx + 4:].strip()
            break
        elif line.strip().startswith("<"):
            node_xml = line.strip()
            break

    if not node_xml:
        match = re.search(r"<[^>]+>", text)
        if match:
            node_xml = match.group(0)

    return should_insert, node_xml


def _extract_assertion_code(text: str) -> Optional[str]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:python)?\s*", "", text, flags=re.IGNORECASE | re.MULTILINE)
    text = re.sub(r"```\s*$", "", text, flags=re.MULTILINE)
    patterns = [
        r"elem\s*=\s*driver\.find_element.*?assert\s+.*",
        r"elems\s*=\s*driver\.find_elements.*?assert\s+.*",
        r"check_current_app_package\s*\(.*\)",
        r"check_dark_mode\s*\(.*\)",
        r"check_light_mode\s*\(.*\)",
        r"check_row_checked_by_text\s*\(.*\)",
        r"check_elements_same_screen\s*\(.*\)",
    ]
    for pat in patterns:
        match = re.search(pat, text, re.DOTALL)
        if match:
            return match.group(0).strip()
    return text.strip() or None


def _build_existence_assertion(target_text: str, resource_id: Optional[str] = None) -> str:
    if resource_id:
        return f"elem = driver.find_element(By.XPATH, \"//*[@resource-id='{resource_id}' and @text='{target_text}']\"); assert elem.is_displayed()"
    escaped = target_text.replace("'", "\\'")
    return f"elem = driver.find_element(By.XPATH, \"//*[@text='{escaped}']\"); assert elem.is_displayed()"


def _build_nonexistence_assertion(target_text: str, resource_id: Optional[str] = None) -> str:
    if resource_id:
        return f"elems = driver.find_elements(By.XPATH, \"//*[@resource-id='{resource_id}' and @text='{target_text}']\"); assert len(elems) == 0"
    escaped = target_text.replace("'", "\\'")
    return f"elems = driver.find_elements(By.XPATH, \"//*[@text='{escaped}']\"); assert len(elems) == 0"


# ---------- 核心函数（修改点：插入压缩步骤）----------
def generate_androb2o_assertions_for_step(
    step_index: int,
    full_xml: str,
    task_description: str,
    allowed_assertions: List[str],
    model: Optional[str] = None,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
) -> tuple[list[str], int]:
    settings = load_llm_settings("BASELINE") if model is None else None
    client = make_openai_client(
        base_url or (settings.base_url if settings else None),
        api_key or (settings.api_key if settings else None),
    )
    if client is None:
        _log("LLM client unavailable, falling back to no assertion.")
        return [], 0

    model = model or (settings.model if settings else "Qwen/Qwen3-VL-32B-Instruct")
    total_tokens = 0

    # ===== 新增：压缩 XML 输入 =====
    compressed_xml = _compress_xml_for_prompt(full_xml)
    _log(f"Step {step_index}: 原始 XML {len(full_xml.encode('utf-8'))} 字节，压缩后 {len(compressed_xml.encode('utf-8'))} 字节。")

    # ---- Stage 1: Decision and element extraction ----
    _log(f"Step {step_index}: Stage 1 - Decide insertion and extract element")
    element_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in split_prompt_element(4000, compressed_xml, task_description):
        element_messages.append({"role": "user", "content": msg})

    try:
        element_response = get_openai_chat_completion(client, model=model, messages=element_messages, temperature=0)
        element_output = element_response.choices[0].message.content or ""
        total_tokens += extract_openai_token_usage(element_response)
        _log(f"Element extraction response (first 500 chars): {element_output[:500]}")
    except Exception as e:
        _log(f"Element extraction failed: {e}")
        return [], total_tokens

    should_insert, node_xml = _parse_element_response(element_output)
    if not should_insert:
        _log("No assertion needed for this step.")
        return [], total_tokens

    _log(f"Will insert assertion. Extracted node: {node_xml[:200] if node_xml else 'None'}")

    if node_xml and not node_xml.endswith("/>"):
        node_xml = node_xml.rstrip(">") + "/>"

    # ---- Stage 2: Assertion generation ----
    _log(f"Step {step_index}: Stage 2 - Generate assertion from allowed types")
    assertion_messages = [{"role": "system", "content": "You are a helpful assistant."}]
    for msg in split_prompt_assertion(4000, compressed_xml, task_description, node_xml, allowed_assertions):
        assertion_messages.append({"role": "user", "content": msg})

    try:
        assertion_response = get_openai_chat_completion(client, model=model, messages=assertion_messages, temperature=0)
        assertion_output = assertion_response.choices[0].message.content or ""
        total_tokens += extract_openai_token_usage(assertion_response)
        _log(f"Assertion generation response (first 500 chars): {assertion_output[:500]}")
    except Exception as e:
        _log(f"Assertion generation failed: {e}")
        if node_xml:
            try:
                node = ET.fromstring(node_xml)
                resource_id = node.attrib.get("resource-id", "").split("/")[-1] if "/" in node.attrib.get("resource-id", "") else node.attrib.get("resource-id", "")
                text = node.attrib.get("text", "")
                if "no" in task_description.lower() and ("check" in task_description.lower() or "verify" in task_description.lower()):
                    assertion = _build_nonexistence_assertion(text, resource_id)
                else:
                    assertion = _build_existence_assertion(text, resource_id)
                return [assertion], total_tokens
            except Exception:
                return [], total_tokens
        else:
            return [], total_tokens

    assertion_code = _extract_assertion_code(assertion_output)
    if assertion_code:
        assertion_code = re.sub(r'(By\.XPATH\s*,\s*)(["\'])(.*?)\2', r'\1"\3"', assertion_code, flags=re.DOTALL)
        return [assertion_code], total_tokens
    else:
        _log("Could not extract assertion code from LLM response; using fallback template.")
        if node_xml:
            try:
                node = ET.fromstring(node_xml)
                resource_id = node.attrib.get("resource-id", "").split("/")[-1] if "/" in node.attrib.get("resource-id", "") else node.attrib.get("resource-id", "")
                text = node.attrib.get("text", "")
                if "no" in task_description.lower() and ("check" in task_description.lower() or "verify" in task_description.lower()):
                    assertion = _build_nonexistence_assertion(text, resource_id)
                else:
                    assertion = _build_existence_assertion(text, resource_id)
                return [assertion], total_tokens
            except Exception:
                return [], total_tokens
        else:
            return [], total_tokens


# 保留兼容别名
def generate_assertions_for_step(step_index, full_xml, task_description, allowed_assertions=None, **kwargs):
    if allowed_assertions is None:
        allowed_assertions = [
            "elem = driver.find_element(By.XPATH, ...); assert elem.is_displayed()",
            "elem = driver.find_element(By.XPATH, ...); assert len(elems) == 0",
        ]
    return generate_androb2o_assertions_for_step(step_index, full_xml, task_description, allowed_assertions, **kwargs)