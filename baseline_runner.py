from __future__ import annotations

import base64
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from appium import webdriver

# 只导入必要的工具函数，不导入断言相关模块
from classifier import classify_file, extract_task_description
from llm_client import get_openai_chat_completion, load_llm_settings, make_openai_client
from llm_client import extract_openai_chat_content
from snapshot_utils import create_action_output_paths, create_generated_output_paths
from test_case_runner import execute_test_case
from test_executor import build_driver

# 导入 assertion_generator 中的控件精简函数
from assertion_generator import _build_compact_ui_controls, _serialize_compact_controls


# ============ 配置 ============
DEFAULT_OUTPUT_DIR = Path("output")
DEFAULT_ARTIFACT_ROOT = Path("baseline")
APPIUM_SERVER_URL = "http://localhost:4723"

# LLM 配置（使用 BASELINE 配置）
BASELINE_LLM_SETTINGS = load_llm_settings("BASELINE")
LLM_BASE_URL = BASELINE_LLM_SETTINGS.base_url
LLM_API_KEY = BASELINE_LLM_SETTINGS.api_key
LLM_MODEL = BASELINE_LLM_SETTINGS.model

# ============ XML 配置 ============
MAX_XML_CHARS_PER_STEP = 0
BASELINE_XML_KEEP_RATIO = 1.0
MAX_XML_TOTAL_BYTES = int(os.environ.get("BASELINE_MAX_XML_TOTAL_BYTES", str(64 * 1024)))
MAX_PROMPT_VISUAL_BYTES = int(os.environ.get("BASELINE_MAX_PROMPT_VISUAL_BYTES", str(64 * 1024)))
MAX_PROMPT_VISUAL_BYTES = max(MAX_PROMPT_VISUAL_BYTES, 1)

# ============ 多模态配置 ============
ENABLE_MULTIMODAL = True
# 设置为 0 表示不限制截图数量，发送所有步骤的截图
MAX_SCREENSHOTS_PER_STEP = int(os.environ.get("BASELINE_MAX_SCREENSHOTS_PER_STEP", "0"))
MAX_IMAGE_SIZE_MB = 5
MAX_IMAGE_DIMENSION = 960

# 环境变量覆盖
ENABLE_MULTIMODAL = os.environ.get("BASELINE_ENABLE_MULTIMODAL", "1" if ENABLE_MULTIMODAL else "0").strip().lower() in {"1", "true", "yes", "on"}
MAX_SCREENSHOTS_PER_STEP = int(os.environ.get("BASELINE_MAX_SCREENSHOTS_PER_STEP", str(MAX_SCREENSHOTS_PER_STEP)))

# 允许的断言类型
BASELINE_ALLOWED_ASSERTIONS = [
    "check_current_app_package('app package name')",
    "check_dark_mode('.....')",
    "check_light_mode('.....')",
    "check_elements_same_screen(driver, ['//xpath1', '//xpath2',....])",
    "check_row_checked_by_text('row text')",
    "elem = driver.find_element(By.XPATH, ...); assert elem.is_displayed()",
    "elem = driver.find_element(By.XPATH, ...); assert elem.text == ...",
    "elem = driver.find_elements(By.XPATH, ...); assert len(elems) == 0",
    "elem = driver.find_element(By.ID, ...); assert elem.is_displayed()",
    "elem = driver.find_element(By.ID, ...); assert elem.text == ...",
    "elem = driver.find_elements(By.ID, ...); assert len(elems) == 0",
]


@dataclass(frozen=True)
class BaselineInsertion:
    """单条断言插入记录"""
    after_step_index: int
    assertion: str
    reason: str = ""


@dataclass(frozen=True)
class BaselineGenerationResult:
    """Baseline 生成结果"""
    insertions: list[BaselineInsertion]
    reason: str
    raw_response: str
    elapsed_seconds: float = 0.0
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


# ============ 工具函数 ============

def _debug_log(message: str):
    print(f"[BASELINE_DEBUG] {message}")


def _preview_text(value: str | None, limit: int = 1200) -> str:
    text = value or ""
    if len(text) <= limit:
        return text
    return f"{text[:limit]}...\n[TRUNCATED {len(text) - limit} CHARS]"


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


# 删除旧的 _get_page_source（不再使用），直接使用完整 XML


def _truncate_xml_to_limit(raw_page_source: str, max_chars: int) -> str:
    if max_chars <= 0 or len(raw_page_source) <= max_chars:
        return raw_page_source
    head = max(max_chars // 2, 1)
    tail = max(max_chars - head, 1)
    return f"{raw_page_source[:head]}\n...[TRUNCATED {len(raw_page_source) - max_chars} CHARACTERS]...\n{raw_page_source[-tail:]}"


def _truncate_text_to_byte_limit(text: str, max_bytes: int, marker: str = "\n...[TRUNCATED]...\n") -> str:
    """按 UTF-8 字节数截断文本，并尽量保留首尾上下文。"""
    if max_bytes <= 0:
        return ""

    raw = text or ""
    raw_bytes = raw.encode("utf-8")
    if len(raw_bytes) <= max_bytes:
        return raw

    marker_bytes = marker.encode("utf-8")
    if max_bytes <= len(marker_bytes):
        # 预算极小的时候，直接返回 marker 的可容纳前缀。
        return marker_bytes[:max_bytes].decode("utf-8", errors="ignore")

    available = max_bytes - len(marker_bytes)
    head_bytes = available // 2
    tail_bytes = available - head_bytes

    head = raw_bytes[:head_bytes].decode("utf-8", errors="ignore")
    tail = raw_bytes[-tail_bytes:].decode("utf-8", errors="ignore") if tail_bytes > 0 else ""
    result = f"{head}{marker}{tail}"

    # 兜底：若由于解码边界导致少量超限，继续收缩前后文直到满足预算。
    while len(result.encode("utf-8")) > max_bytes and (head or tail):
        if len(head) >= len(tail) and head:
            head = head[:-1]
        elif tail:
            tail = tail[1:]
        result = f"{head}{marker}{tail}"

    return result


def _truncate_xml_to_byte_limit(raw_page_source: str, max_bytes: int) -> str:
    return _truncate_text_to_byte_limit(raw_page_source, max_bytes)


def _apply_total_xml_budget(step_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if MAX_XML_TOTAL_BYTES <= 0:
        return step_payloads

    total_xml_chars = sum(len(str(payload.get("ui_xml", ""))) for payload in step_payloads)
    if total_xml_chars <= MAX_XML_TOTAL_BYTES:
        _debug_log(f"XML budget check passed: total={total_xml_chars} chars <= limit={MAX_XML_TOTAL_BYTES}")
        return step_payloads

    _debug_log(
        f"XML budget exceeded: total={total_xml_chars} chars > limit={MAX_XML_TOTAL_BYTES}; applying proportional truncation"
    )

    budget = MAX_XML_TOTAL_BYTES
    non_xml_overhead = max(total_xml_chars, 1)
    adjusted: list[dict[str, Any]] = []
    remaining_budget = budget

    for index, payload in enumerate(step_payloads):
        xml_text = str(payload.get("ui_xml", ""))
        xml_len = len(xml_text)
        if xml_len <= 0:
            adjusted.append(payload)
            continue

        if index == len(step_payloads) - 1:
            target_len = max(1, min(xml_len, remaining_budget))
        else:
            target_len = max(1, min(xml_len, int(budget * (xml_len / non_xml_overhead))))
            remaining_budget -= target_len
            remaining_budget = max(remaining_budget, 1)

        new_payload = dict(payload)
        new_payload["ui_xml"] = _truncate_xml_to_limit(xml_text, target_len)
        new_payload["ui_xml_length"] = len(new_payload["ui_xml"])
        adjusted.append(new_payload)

    final_total = sum(len(str(payload.get("ui_xml", ""))) for payload in adjusted)
    _debug_log(f"XML budget applied: final_total={final_total} chars")
    return adjusted


# ---------- 压缩 data URL 图片的函数 ----------
def _compress_data_url_image(
    data_url: str | None,
    max_dimension: int,
    jpeg_quality: int,
) -> str | None:
    """压缩 data URL 图片，返回新的 data URL（JPEG 格式）。若失败则返回原 URL。"""
    if not data_url or not data_url.startswith("data:image/"):
        return data_url

    try:
        header, encoded = data_url.split(",", 1)
        image_bytes = base64.b64decode(encoded)
    except Exception:
        return data_url

    try:
        from PIL import Image
        import io
        img = Image.open(io.BytesIO(image_bytes))
        # 调整尺寸
        if max_dimension > 0 and (img.width > max_dimension or img.height > max_dimension):
            img.thumbnail((max_dimension, max_dimension), Image.Resampling.LANCZOS)
        # 转 RGB 以保存为 JPEG
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGB")
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=jpeg_quality, optimize=True)
        compressed_bytes = buffer.getvalue()
        encoded_compressed = base64.b64encode(compressed_bytes).decode("ascii")
        return f"data:image/jpeg;base64,{encoded_compressed}"
    except Exception as e:
        _debug_log(f"Failed to compress data URL: {e}")
        return data_url


def _shrink_payloads_to_visual_budget(step_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    压缩截图（而非丢弃）以满足 MAX_PROMPT_VISUAL_BYTES 预算。
    若压缩截图后仍超限，则对 XML 进行精简：仅保留控件的五个关键属性。
    """
    if MAX_PROMPT_VISUAL_BYTES <= 0:
        return step_payloads

    payloads = [dict(payload) for payload in step_payloads]

    def _image_bytes(item: dict[str, Any]) -> int:
        return len((item.get("image_data_url") or "").encode("utf-8"))

    def _budget_usage(items: list[dict[str, Any]]) -> tuple[int, int]:
        image_total = sum(_image_bytes(item) for item in items)
        xml_total = sum(len(str(item.get("ui_xml", "")).encode("utf-8")) for item in items)
        return image_total, xml_total

    # 第一步：压缩截图
    image_bytes, xml_bytes = _budget_usage(payloads)
    total_bytes = xml_bytes + image_bytes

    if total_bytes <= MAX_PROMPT_VISUAL_BYTES:
        _debug_log(
            f"Visual budget check passed: xml={xml_bytes}B, images={image_bytes}B, total={total_bytes}B ≤ limit={MAX_PROMPT_VISUAL_BYTES}B"
        )
        return payloads

    _debug_log(
        f"Visual budget exceeded: xml={xml_bytes}B, images={image_bytes}B, total={total_bytes}B > limit={MAX_PROMPT_VISUAL_BYTES}B, "
        "starting progressive compression of screenshots..."
    )

    # 压缩级别：从较高质量到低质量，逐步降低
    compression_levels = [
        (960, 55),
        (720, 45),
        (640, 40),
        (512, 35),
        (384, 30),
        (256, 25),
        (160, 20),
        (96, 10),
    ]

    # 对每个级别尝试压缩截图
    for max_dimension, quality in compression_levels:
        # 压缩所有 payload 中的截图
        for payload in payloads:
            original_url = payload.get("image_data_url")
            if original_url:
                compressed = _compress_data_url_image(original_url, max_dimension, quality)
                payload["image_data_url"] = compressed

        # 重新计算大小
        new_image_bytes, new_xml_bytes = _budget_usage(payloads)
        new_total = new_xml_bytes + new_image_bytes
        _debug_log(
            f"Compression level {max_dimension}x{max_dimension}, quality={quality}: "
            f"images={new_image_bytes}B, xml={new_xml_bytes}B, total={new_total}B"
        )
        if new_total <= MAX_PROMPT_VISUAL_BYTES:
            _debug_log(f"Visual budget satisfied after screenshot compression at level {max_dimension}x{max_dimension}, quality={quality}")
            return payloads

    # 第二步：若截图压缩后仍超限，则精简 XML（仅保留五个关键属性）
    _debug_log(
        "Screenshot compression alone insufficient. Now compressing XML to key attributes only."
    )

    for payload in payloads:
        ui_xml = payload.get("ui_xml")
        if not ui_xml:
            continue
        # 注意：此时 ui_xml 是完整的原始 XML（因为 _build_step_payloads 已改为存储完整 XML）
        compact_controls = _build_compact_ui_controls(ui_xml, drop_container_controls=False)
        compact_xml = _serialize_compact_controls(compact_controls)
        payload["ui_xml"] = compact_xml
        payload["ui_xml_length"] = len(compact_xml)

    # 重新计算总大小
    final_image_bytes, final_xml_bytes = _budget_usage(payloads)
    final_total = final_xml_bytes + final_image_bytes

    if final_total <= MAX_PROMPT_VISUAL_BYTES:
        _debug_log(
            f"Visual budget satisfied after XML compaction: images={final_image_bytes}B, xml={final_xml_bytes}B, total={final_total}B"
        )
        return payloads

    # 若仍超限，输出警告并保留当前压缩结果
    _debug_log(
        f"[WARNING] Visual budget still exceeded after all compression and XML compaction: "
        f"images={final_image_bytes}B, xml={final_xml_bytes}B, total={final_total}B > limit={MAX_PROMPT_VISUAL_BYTES}B"
    )
    return payloads


def _encode_image_as_data_url(image_path: str | None) -> str | None:
    """将图片编码为 base64 data URL"""
    if not image_path:
        return None
    path = Path(image_path)
    if not path.is_file():
        return None
    
    try:
        with open(path, "rb") as f:
            image_data = f.read()
        
        # 如果图片太大，尝试压缩
        if len(image_data) > MAX_IMAGE_SIZE_MB * 1024 * 1024:
            try:
                from PIL import Image
                import io
                img = Image.open(io.BytesIO(image_data))
                if MAX_IMAGE_DIMENSION > 0 and (img.width > MAX_IMAGE_DIMENSION or img.height > MAX_IMAGE_DIMENSION):
                    img.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
                if img.mode in ("RGBA", "LA", "P"):
                    img = img.convert("RGB")
                buffer = io.BytesIO()
                img.save(buffer, format="JPEG", quality=55, optimize=True)
                image_data = buffer.getvalue()
                _debug_log(f"Compressed image: {path.stat().st_size / 1024:.1f}KB -> {len(image_data) / 1024:.1f}KB")
            except ImportError:
                _debug_log(f"PIL not available, skipping compression for {path}")
            except Exception as e:
                _debug_log(f"Compression failed: {e}")
        
        encoded = base64.b64encode(image_data).decode("ascii")
        mime_type = "image/jpeg" if len(image_data) < path.stat().st_size else "image/png"
        return f"data:{mime_type};base64,{encoded}"
    except Exception as e:
        _debug_log(f"Failed to encode image: {e}")
        return None


def _data_url_payload_size_bytes(data_url: str | None) -> int:
    if not data_url:
        return 0
    if not isinstance(data_url, str):
        return len(str(data_url).encode("utf-8"))

    try:
        _, encoded = data_url.split(",", 1)
    except ValueError:
        return len(data_url.encode("utf-8"))

    # data URL 的实际传输成本以 base64 文本为主，采用 UTF-8 字节长度估算即可。
    return len(encoded.encode("utf-8"))


def _xml_payload_size_bytes(xml_text: str | None) -> int:
    return len((xml_text or "").encode("utf-8"))


def _ensure_prompt_visual_budget(step_payloads: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _shrink_payloads_to_visual_budget(step_payloads)


def _extract_usage_stats(response: Any) -> tuple[int | None, int | None, int | None]:
    usage = getattr(response, "usage", None)
    if usage is None:
        if isinstance(response, dict):
            usage = response.get("usage")
    if usage is None:
        return None, None, None

    if isinstance(usage, dict):
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        total_tokens = usage.get("total_tokens")
    else:
        prompt_tokens = getattr(usage, "prompt_tokens", None)
        completion_tokens = getattr(usage, "completion_tokens", None)
        total_tokens = getattr(usage, "total_tokens", None)
    
    def _as_int(value: Any) -> int | None:
        return int(value) if isinstance(value, (int, float)) else None
    
    return _as_int(prompt_tokens), _as_int(completion_tokens), _as_int(total_tokens)


def _coalesce_total_tokens(prompt_tokens: int | None, completion_tokens: int | None, total_tokens: int | None) -> int:
    if total_tokens is not None:
        return total_tokens
    if prompt_tokens is not None and completion_tokens is not None:
        return prompt_tokens + completion_tokens
    return 0


# ============ 文件操作 ============

def iter_test_files(output_dir: Path):
    yield from sorted(path for path in output_dir.glob("*.txt") if path.is_file())


def collect_test_files(paths: list[str]) -> list[Path]:
    if not paths:
        return list(iter_test_files(DEFAULT_OUTPUT_DIR))

    collected: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = iter_test_files(path)
        elif path.is_file() and path.suffix.lower() == ".txt":
            candidates = [path]
        else:
            raise FileNotFoundError(f"Input path not found: {path}")

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(candidate)

    return collected


def _read_test_case_sections(test_file: Path) -> tuple[list[str], list[str], list[str]]:
    """读取测试用例文件，返回 header、body 和 action 描述"""
    with test_file.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    header_lines: list[str] = []
    body_lines: list[str] = []
    reached_body = False
    
    for line in lines:
        if not reached_body and line.strip() == "---":
            reached_body = True
            header_lines.append(line.rstrip("\n"))
            continue
        if not reached_body:
            header_lines.append(line.rstrip("\n"))
        else:
            if not line.startswith("appPackage:") and not line.startswith("appActivity:") and not line.startswith("---"):
                body_lines.append(line.rstrip("\n"))

    action_lines: list[str] = []
    for line in body_lines:
        match = re.match(r"^\d+\. \((action|assertion)\) (.*)", line.strip())
        if match and match.group(1) == "action":
            action_lines.append(match.group(2))

    return header_lines, body_lines, action_lines


# ============ Prompt 构建 ============

def _build_step_payloads(action_snapshots: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """
    构建每个步骤的 payload，包含完整 XML 和截图。
    XML 不做截断，完整传输。
    """
    payloads: list[dict[str, Any]] = []
    
    for snapshot in action_snapshots:
        # 直接获取完整的原始 XML，不进行任何截断
        ui = snapshot.get("ui", {})
        full_page_source = ui.get("raw_page_source") or ""
        
        payload = {
            "step_index": snapshot.get("step_index"),
            "step_type": snapshot.get("step_type"),
            "step_label": snapshot.get("step_label"),
            "line": snapshot.get("line"),
            "current_package": snapshot.get("current_package"),
            "current_activity": snapshot.get("current_activity"),
            "ui_xml_length": len(full_page_source),
            "ui_xml": full_page_source, 
        }
        
        if ENABLE_MULTIMODAL:
            screenshot_path = snapshot.get("screenshot_path")
            if screenshot_path:
                image_url = _encode_image_as_data_url(screenshot_path)
                if image_url:
                    payload["image_data_url"] = image_url
        
        payloads.append(payload)
    
    return _ensure_prompt_visual_budget(payloads)


def _build_prompt_messages(
    task_description: str,
    action_snapshots: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    构建发送给 LLM 的 Prompt。
    """
    
    system_prompt = (
        "Given a task description and a sequence of "
        "UI snapshots from an Android app test execution, you should generate test assertions.\n\n"
        "Return ONLY valid JSON with the following structure:\n"
        "{\n"
        "  \"insertions\": [\n"
        "    {\n"
        "      \"after_step_index\": 1,\n"
        "      \"assertion\": \".....\",\n"
        "      \"reason\": \"why this assertion is needed\"\n"
        "    }\n"
        "  ],\n"
        "  \"notes\": \"optional overall notes\"\n"
        "}\n\n"
        "Allowed assertion types:\n"
        + "\n".join(f"- {a}" for a in BASELINE_ALLOWED_ASSERTIONS) +
        "\n\nRules:\n"
        "1. Use the step_index from the snapshots as after_step_index.\n"
        "2. Do NOT renumber the test case yourself.\n"
        "3. If no assertion is needed, return an empty insertions array.\n"
    )

    # 构建用户消息
    instruction = {
        "task_description": task_description,
        "task_instruction": (
            "Generate assertions that meet the needs described in the task. "
        ),
        "output_format": {
            "insertions": [
                {
                    "after_step_index": "<step number to insert after>",
                    "assertion": "<Python assertion code>",
                    "reason": "<why this assertion is needed>"
                }
            ],
            "notes": "<optional notes>"
        }
    }

    instruction_text = json.dumps(instruction, ensure_ascii=False, indent=2)

    # 构建步骤数据（包含完整 XML）
    step_payloads = _build_step_payloads(action_snapshots)
    
    if not ENABLE_MULTIMODAL:
        # 纯文本模式
        steps_text = ""
        for payload in step_payloads:
            payload_copy = {k: v for k, v in payload.items() if k != "image_data_url"}
            steps_text += "\n\n" + json.dumps(payload_copy, ensure_ascii=False, indent=2)
        
        user_content = instruction_text + steps_text
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
    
    # 多模态模式
    user_content_parts = [{"type": "text", "text": instruction_text}]
    
    # 先添加所有步骤的文本摘要（包含完整 XML）
    for payload in step_payloads:
        payload_copy = {k: v for k, v in payload.items() if k not in ("image_data_url", "screenshot_path")}
        steps_text = json.dumps(payload_copy, ensure_ascii=False, indent=2)
        user_content_parts.append({
            "type": "text",
            "text": f"\n--- Step {payload.get('step_index', '?')} ---\n{steps_text}"
        })
    
    # 再添加所有截图（不受数量限制）
    for payload in step_payloads:
        image_url = payload.get("image_data_url")
        if image_url:
            user_content_parts.append({
                "type": "text",
                "text": f"\n[Screenshot for Step {payload.get('step_index', '?')}]"
            })
            user_content_parts.append({
                "type": "image_url",
                "image_url": {"url": image_url}
            })
    
    _debug_log(f"Built multimodal prompt with {len([p for p in step_payloads if p.get('image_data_url')])} images")
    _debug_log(f"Total XML characters across all steps: {sum(p.get('ui_xml_length', 0) for p in step_payloads)}")
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content_parts},
    ]


# ============ LLM 调用和解析 ============

def _load_client():
    client = make_openai_client(LLM_BASE_URL, LLM_API_KEY)
    if client is None:
        _debug_log("Failed to initialize OpenAI client")
    return client


def _normalize_generation_output(raw_text: str) -> BaselineGenerationResult:
    """解析 LLM 返回的 JSON"""
    text = raw_text.strip()
    if not text:
        return BaselineGenerationResult([], "empty response", raw_text)

    # 移除 markdown 代码块
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()

    try:
        payload = json.loads(text)
    except Exception:
        # 尝试从非 JSON 响应中提取
        insertions: list[BaselineInsertion] = []
        pattern = r"after_step_index\s*[:=]\s*(\d+).{0,200}?assertion\s*[:=]\s*([\"'])(.+?)\2"
        for match in re.finditer(pattern, text, re.IGNORECASE | re.DOTALL):
            step_index = int(match.group(1))
            assertion = match.group(3).strip()
            if assertion:
                insertions.append(BaselineInsertion(step_index, assertion, "recovered from non-JSON"))
        if insertions:
            return BaselineGenerationResult(insertions, "recovered from non-JSON", raw_text)
        return BaselineGenerationResult([], "non-JSON response", raw_text)

    insertions_raw = payload.get("insertions", [])
    notes = str(payload.get("notes", payload.get("reason", "ok")))
    insertions: list[BaselineInsertion] = []

    if isinstance(insertions_raw, list):
        for item in insertions_raw:
            if not isinstance(item, dict):
                continue
            step_index = item.get("after_step_index")
            assertion = item.get("assertion")
            reason = str(item.get("reason", ""))
            if isinstance(step_index, int) and isinstance(assertion, str) and assertion.strip():
                insertions.append(BaselineInsertion(step_index, assertion.strip(), reason))

    return BaselineGenerationResult(insertions, notes, raw_text)


def _generate_baseline_insertions(
    task_description: str,
    action_snapshots: list[dict[str, Any]],
    result_logger=None,
) -> BaselineGenerationResult:
    """核心函数：调用 LLM 生成断言插入点"""
    client = _load_client()
    if client is None:
        return BaselineGenerationResult([], "LLM client unavailable", "")

    _debug_log(f"Multimodal: {'ON' if ENABLE_MULTIMODAL else 'OFF'}")
    
    messages = _build_prompt_messages(task_description, action_snapshots)

    _debug_log(
        f"Calling LLM: model={LLM_MODEL}, snapshots={len(action_snapshots)}, "
        f"messages={len(messages)}"
    )

    try:
        started_at = time.perf_counter()
        response = get_openai_chat_completion(
            client,
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
        )
        elapsed = time.perf_counter() - started_at
        prompt_tokens, completion_tokens, total_tokens = _extract_usage_stats(response)
        total_tokens = _coalesce_total_tokens(prompt_tokens, completion_tokens, total_tokens)
        content = extract_openai_chat_content(response)
        
        _debug_log(f"Raw response: {content!r}")
        
        if result_logger:
            result_logger.write("\n[baseline] LLM Response:\n")
            result_logger.write(content + "\n")

        result = _normalize_generation_output(content)
        return BaselineGenerationResult(
            insertions=result.insertions,
            reason=result.reason,
            raw_response=result.raw_response,
            elapsed_seconds=elapsed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )
    except Exception as exc:
        import traceback
        _debug_log(f"Generation failed: {exc}")
        _debug_log(traceback.format_exc())
        if result_logger:
            result_logger.write(f"\n[baseline] Generation failed: {exc}\n")
        return BaselineGenerationResult([], f"failed: {exc}", "")


# ============ 输出渲染 ============

def _render_output_test_case(header_lines: list[str], steps: list[str]) -> str:
    lines = [*header_lines]
    lines.extend(steps)
    return "\n".join(lines).rstrip() + "\n"


def _renumber_steps(raw_steps: list[str]) -> list[str]:
    renumbered = []
    for index, step in enumerate(raw_steps, 1):
        renumbered.append(re.sub(r"^\d+\.", f"{index}.", step, count=1))
    return renumbered


def _format_assertion_line(step_number: int, assertion: str) -> str:
    return f"{step_number}. (assertion) {assertion}"


def _step_content_key(step_line: str) -> str:
    match = re.match(r'^\d+\. \((action|assertion)\) (.*)$', step_line.strip())
    if match:
        return re.sub(r'\s+', ' ', match.group(2).strip())
    return re.sub(r'\s+', ' ', step_line.strip())


def _render_final_steps(body_lines: list[str], insertions: list[BaselineInsertion]) -> list[str]:
    insertion_map: dict[int, list[str]] = {}
    for insertion in insertions:
        insertion_map.setdefault(insertion.after_step_index, []).append(insertion.assertion)

    final_steps: list[str] = []
    for line in body_lines:
        final_steps.append(line.rstrip())
        match = re.match(r"^(\d+)\. \((action|assertion)\)", line.strip())
        if not match:
            continue
        step_num = int(match.group(1))
        for assertion in insertion_map.get(step_num, []):
            final_steps.append(_format_assertion_line(step_num + 1, assertion))

    return _renumber_steps(final_steps)


def _render_final_output_test_case(
    header_lines: list[str],
    body_lines: list[str],
    insertions: list[BaselineInsertion],
) -> str:
    final_steps = _render_final_steps(body_lines, insertions)

    existing_assertion_keys = {
        _step_content_key(line)
        for line in body_lines
        if re.match(r'^\d+\. \(assertion\) ', line.strip())
    }
    emitted_generated_keys: set[str] = set()

    deduped_steps: list[str] = []
    for line in final_steps:
        match = re.match(r'^(\d+)\. \(assertion\) (.*)$', line.strip())
        if match:
            assertion_key = _step_content_key(line)
            if assertion_key in existing_assertion_keys or assertion_key in emitted_generated_keys:
                continue
            emitted_generated_keys.add(assertion_key)
        deduped_steps.append(line)

    return _render_output_test_case(header_lines, _renumber_steps(deduped_steps))


# ============ 主流程 ============

def run_one_test_case(test_file: Path, artifact_root: Path) -> tuple[int, int]:
    """运行单个测试用例的 Baseline"""
    started_at = time.perf_counter()
    options = build_driver(str(test_file))
    driver = None
    
    try:
        driver = webdriver.Remote(APPIUM_SERVER_URL, options=options)
        driver.implicitly_wait(10)

        # 创建输出路径
        test_case_stem, _, json_path = create_action_output_paths(
            str(test_file), artifact_root=str(artifact_root)
        )
        json_path = Path(json_path)
        _, output_txt_path, output_json_path = create_generated_output_paths(
            str(test_file), artifact_root=str(artifact_root)
        )
        output_txt_path = Path(output_txt_path)
        output_json_path = Path(output_json_path)

        result_file = artifact_root / test_case_stem / "result.txt"
        result_file.parent.mkdir(parents=True, exist_ok=True)

        with result_file.open("w", encoding="utf-8") as result_logger:
            result_logger.write("--- Test Execution Log ---\n")
            
            # 执行测试用例（不生成断言）
            # 注意：execute_test_case 返回三个值，使用 _ 忽略第三个（token 数）
            total_steps, passed_steps, _ = execute_test_case(
                driver,
                str(test_file),
                result_logger,
                generate_assertions=False,
                artifact_root=str(artifact_root),
            )

            result_logger.write("\n--- Baseline Generation ---\n")
            
            # 读取测试用例信息
            header_lines, body_lines, action_lines = _read_test_case_sections(test_file)
            task_case = extract_task_description(test_file)
            category = classify_file(test_file)

            _debug_log(
                f"Test: {test_file.name}, category={category}, "
                f"task={task_case.task_description!r}, steps={len(body_lines)}"
            )

            # 加载执行过程中收集的快照
            if json_path.exists():
                action_snapshots = json.loads(json_path.read_text(encoding="utf-8"))
            else:
                action_snapshots = []
                _debug_log(f"Warning: No snapshot JSON found at {json_path}")

            _debug_log(f"Loaded {len(action_snapshots)} snapshots")

            # 调用 LLM 生成断言
            generation = _generate_baseline_insertions(
                task_description=task_case.task_description,
                action_snapshots=action_snapshots,
                result_logger=result_logger,
            )

            # 生成最终测试用例
            output_text = _render_final_output_test_case(
                header_lines, body_lines, generation.insertions
            )
            output_txt_path.write_text(output_text, encoding="utf-8")

            _debug_log(f"Output written to: {output_txt_path}")

            # 保存元数据
            output_payload = {
                "category": category,
                "task_description": task_case.task_description,
                "input_file": str(test_file),
                "output_file": str(output_txt_path),
                "multimodal_enabled": ENABLE_MULTIMODAL,
                "xml_truncated": False, 
                "max_xml_chars": MAX_XML_CHARS_PER_STEP,
                "baseline": {
                    "insertions": [
                        {
                            "after_step_index": i.after_step_index,
                            "assertion": i.assertion,
                            "reason": i.reason,
                        }
                        for i in generation.insertions
                    ],
                    "reason": generation.reason,
                    "raw_response": generation.raw_response,
                    "generation_stats": {
                        "elapsed_seconds": generation.elapsed_seconds,
                        "prompt_tokens": generation.prompt_tokens,
                        "completion_tokens": generation.completion_tokens,
                        "total_tokens": generation.total_tokens,
                    },
                },
                "snapshot_json": str(json_path),
            }
            output_json_path.write_text(
                json.dumps(output_payload, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )

            result_logger.write(f"Insertions: {len(generation.insertions)}\n")
            result_logger.write(f"Elapsed: {generation.elapsed_seconds:.3f}s\n")
            result_logger.write(
                "Token usage: "
                f"prompt={generation.prompt_tokens or 0}, "
                f"completion={generation.completion_tokens or 0}, "
                f"total={generation.total_tokens or 0}\n"
            )
            result_logger.write(f"Total tokens consumed: {generation.total_tokens or 0}\n")
            result_logger.write(f"Total runtime: {time.perf_counter() - started_at:.3f}s\n")
            result_logger.write(f"XML truncated: {MAX_XML_CHARS_PER_STEP > 0 or BASELINE_XML_KEEP_RATIO < 1}\n")
            result_logger.write(f"Visual budget limit: {MAX_PROMPT_VISUAL_BYTES} bytes\n")

        return total_steps, passed_steps
        
    finally:
        if driver is not None:
            driver.quit()


def main() -> int:
    try:
        test_files = collect_test_files(sys.argv[1:])
    except FileNotFoundError as exc:
        print(exc)
        return 1

    if not test_files:
        print(f"No .txt test files found in {DEFAULT_OUTPUT_DIR}")
        return 1

    print(f"Found {len(test_files)} test file(s)")
    print(f"Appium: {APPIUM_SERVER_URL}")
    print(f"Artifact root: {DEFAULT_ARTIFACT_ROOT}")
    print(f"Multimodal: {'ENABLED' if ENABLE_MULTIMODAL else 'DISABLED'}")
    print(f"XML truncation: DISABLED (full XML transmitted)")
    print(f"Model: {LLM_MODEL}")

    artifact_root = DEFAULT_ARTIFACT_ROOT
    artifact_root.mkdir(parents=True, exist_ok=True)

    for test_file in test_files:
        print(f"\n=== Running baseline for {test_file.name} ===")
        try:
            total_steps, passed_steps = run_one_test_case(test_file, artifact_root)
            print(f"Completed: {passed_steps}/{total_steps} steps")
        except Exception as exc:
            print(f"Failed: {exc}")
            import traceback
            traceback.print_exc()

    print("\n=== Baseline complete ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())