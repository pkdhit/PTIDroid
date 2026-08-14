from __future__ import annotations

import base64
import io
import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

from llm_client import get_openai_chat_completion, load_llm_settings, make_openai_client, extract_openai_token_usage


LLM_SELECTION_INPUT_BYTE_LIMIT = 10 * 1024


def _normalize_text(value: str | None) -> str:
    if not value:
        return ""
    value = value.lower().strip()
    value = re.sub(r"[._/\\-]+", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip()


def _tokenize_for_matching(value: str | None) -> set[str]:
    normalized = _normalize_text(value)
    if not normalized:
        return set()
    return {token for token in normalized.split() if len(token) >= 3}


def _tokenize_loose(value: str | None) -> set[str]:
    text = _normalize_text(value)
    if not text:
        return set()
    tokens = set(re.findall(r"[a-z0-9\u4e00-\u9fff]+", text))
    stopwords = {
        "a", "an", "as", "at", "be", "if", "in", "is", "it", "of", "on", "or",
        "so", "the", "and", "for", "not", "with", "that", "this", "when",
        "where", "which", "whether","appear", "click", "clickable", "close", "continue", "dismiss"
    }
    return {token for token in tokens if len(token) >= 2 and token not in stopwords}


def _soft_token_overlap_score(intent_tokens: set[str], candidate_tokens: set[str]) -> int:
    if not intent_tokens or not candidate_tokens:
        return 0

    score = len(intent_tokens & candidate_tokens) * 6
    if score:
        return score

    for intent_token in intent_tokens:
        for candidate_token in candidate_tokens:
            if intent_token == candidate_token:
                continue
            if len(intent_token) < 3 or len(candidate_token) < 2:
                continue

            if intent_token in candidate_token or candidate_token in intent_token:
                score += 2
                continue

            if intent_token[:3] == candidate_token[:3] or intent_token[-3:] == candidate_token[-3:]:
                score += 1

    return score


def _default_model() -> str:
    return load_llm_settings("ASSERTION").model


def _load_client(base_url: str | None = None, api_key: str | None = None):
    # 如果未显式传入，则从全局配置读取
    if base_url is None or api_key is None:
        settings = load_llm_settings("ASSERTION")
        if base_url is None:
            base_url = settings.base_url
        if api_key is None:
            api_key = settings.api_key
    client = make_openai_client(base_url, api_key)
    if client is None:
        _debug_log("Failed to initialize OpenAI-compatible client for control selection.")
    return client


def _debug_log(message: str):
    print(f"[CONTROL_SELECTOR_DEBUG] {message}")


def _format_budget_usage(used_bytes: int, limit_bytes: int) -> str:
    if limit_bytes <= 0:
        return f"{used_bytes}B/0B"
    return f"{used_bytes}B/{limit_bytes}B ({used_bytes / limit_bytes:.1%})"


def _is_container_like(summary: dict[str, str]) -> bool:
    class_name = _normalize_text(summary.get("class"))
    if not class_name:
        return False
    container_keywords = (
        "layout", "viewgroup", "framelayout", "linearlayout", "relativelayout",
        "constraintlayout", "scrollview", "listview", "recyclerview",
        "drawerlayout", "coordinatorlayout", "nestedscrollview",
    )
    return any(keyword in class_name for keyword in container_keywords)


def _filter_non_container_ui_candidates(candidates: list[dict[str, str]]) -> list[dict[str, str]]:
    filtered: list[dict[str, str]] = []
    for item in candidates:
        if not isinstance(item, dict):
            continue
        summary: dict[str, str] = {}
        for key in ("xpath", "resource-id", "text", "content-desc", "class"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                summary[key] = value.strip()
        if not summary:
            continue
        if _is_container_like(summary) and ((not summary.get("text")) or (not summary.get("content-desc"))):
            continue
        filtered.append(summary)
    return filtered


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


def _snapshot_image_url(snapshot: dict[str, Any] | None) -> str | None:
    if not snapshot:
        return None
    return _encode_image_as_data_url(snapshot.get("screenshot_path"))


def _snapshot_scroll_image_urls(snapshot: dict[str, Any] | None) -> list[str]:
    if not snapshot:
        return []

    ui = snapshot.get("ui") if isinstance(snapshot, dict) else {}
    if not isinstance(ui, dict):
        return []

    scroll_paths = ui.get("scroll_screenshot_paths")
    if not isinstance(scroll_paths, list):
        return []

    urls: list[str] = []
    for path in scroll_paths:
        if not isinstance(path, str) or not path.strip():
            continue
        url = _encode_image_as_data_url(path.strip())
        if url:
            urls.append(url)
    return urls


def _compact_candidate_controls_for_llm(
    candidates: list[dict[str, str]],
    drop_container_controls: bool = False,
) -> list[dict[str, str]]:
    compacted: list[dict[str, str]] = []
    seen_signatures: set[str] = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue

        control: dict[str, str] = {}
        for key in ("xpath", "resource-id", "text", "content-desc", "class"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                control[key] = value.strip()

        if not control:
            continue
        if drop_container_controls and _is_container_like(control):
            continue

        signature = "|".join(control.get(key, "") for key in ("xpath", "resource-id", "text", "content-desc", "class"))
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)
        compacted.append(control)
    return compacted


def _compact_control_xml_candidates(
    snapshot: dict[str, Any] | None,
    candidates: list[dict[str, str]],
    drop_container_controls: bool = False,
) -> list[dict[str, str]]:
    if candidates:
        return _compact_candidate_controls_for_llm(candidates, drop_container_controls=drop_container_controls)
    return _extract_ui_summary(snapshot)


def _estimate_llm_selection_input_bytes(
    intent: str,
    screenshot_url: str | None,
    scroll_screenshot_urls: list[str],
    candidates: list[dict[str, str]],
) -> int:
    payload = {
        "phase_intent": intent,
        "candidate_controls": candidates,
    }
    parts = [json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")]
    if screenshot_url:
        parts.append(screenshot_url.encode("utf-8"))
    for url in scroll_screenshot_urls:
        if url:
            parts.append(url.encode("utf-8"))
    return sum(len(part) for part in parts)


def _prepare_llm_selection_inputs(
    snapshot: dict[str, Any] | None,
    intent: str,
    candidates: list[dict[str, str]],
    limit: int,
) -> tuple[str | None, list[str], list[dict[str, str]], int]:
    screenshot_url = _snapshot_image_url(snapshot)
    scroll_screenshot_urls = _snapshot_scroll_image_urls(snapshot)
    compact_candidates = _compact_control_xml_candidates(snapshot, candidates, drop_container_controls=False)

    screenshot_steps = [
        (960, 55),
        (720, 45),
        (640, 40),
        (512, 35),
        (384, 30),
        (256, 25),
        (160, 20),
        (96, 10),
    ]

    # 1) 先压缩截图质量和尺寸
    for max_dimension, jpeg_quality in screenshot_steps:
        compressed_screenshot = _compress_data_url_image(screenshot_url, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
        compressed_scroll_screenshots = [
            _compress_data_url_image(url, max_dimension=max_dimension, jpeg_quality=jpeg_quality)
            for url in scroll_screenshot_urls
        ]
        compressed_scroll_screenshots = [url for url in compressed_scroll_screenshots if url]
        total_bytes = _estimate_llm_selection_input_bytes(
            intent,
            compressed_screenshot,
            compressed_scroll_screenshots,
            compact_candidates,
        )
        _debug_log(
            "LLM selection budget probe after screenshot compression: "
            f"dimension={max_dimension}, quality={jpeg_quality}, total={_format_budget_usage(total_bytes, LLM_SELECTION_INPUT_BYTE_LIMIT)}"
        )
        if total_bytes <= LLM_SELECTION_INPUT_BYTE_LIMIT:
            return compressed_screenshot, compressed_scroll_screenshots, compact_candidates, total_bytes

    # 2) 再对所有控件仅保留关键字段
    compact_candidates = _compact_candidate_controls_for_llm(compact_candidates, drop_container_controls=False)
    total_bytes = _estimate_llm_selection_input_bytes(intent, screenshot_url, scroll_screenshot_urls, compact_candidates)
    _debug_log(
        "LLM selection budget probe after control field compaction: "
        f"controls={len(compact_candidates)}, total={_format_budget_usage(total_bytes, LLM_SELECTION_INPUT_BYTE_LIMIT)}"
    )
    if total_bytes <= LLM_SELECTION_INPUT_BYTE_LIMIT:
        return screenshot_url, scroll_screenshot_urls, compact_candidates, total_bytes

    # 3) 最后去除 XML 中的容器控件信息
    compact_candidates = _compact_candidate_controls_for_llm(compact_candidates, drop_container_controls=True)
    total_bytes = _estimate_llm_selection_input_bytes(intent, screenshot_url, scroll_screenshot_urls, compact_candidates)
    _debug_log(
        "LLM selection budget probe after dropping container controls: "
        f"controls={len(compact_candidates)}, total={_format_budget_usage(total_bytes, LLM_SELECTION_INPUT_BYTE_LIMIT)}"
    )

    return screenshot_url, scroll_screenshot_urls, compact_candidates, total_bytes


def _build_llm_selection_messages(
    intent: str,
    screenshot_url: str,
    scroll_screenshot_urls: list[str],
    candidates: list[dict[str, str]],
    limit: int,
) -> list[dict[str, Any]]:
    system_prompt = (
        "You are a UI control selector for Android test assertions. "
        "First analyze the phase intent, then inspect the screenshot to identify the target control and its approximate text and location, "
        "and finally choose the best controls from the provided XML candidate list. "
        "Return ONLY JSON with keys selected_targets and reason. "
        f"Select at most {limit} controls. "
        "You must rely on the screenshot to infer the target control's visible text and approximate information before choosing candidates. "
        "Do not invent control attributes that do not exist in the candidate list."
    )
    user_payload = {
        "phase_intent": intent,
        "candidate_controls": candidates,
        "rules": [
            "Analyze screenshot first.",
            "Choose up to limit controls.",
            "Copy exact evidence from candidates.",
            "Include xpath/resource-id/text/content-desc/class and reason.",
            "Do not return container-only controls unless clearly targeted.",
        ],
    }
    content: list[dict[str, Any]] = [{"type": "text", "text": json.dumps(user_payload, ensure_ascii=False, separators=(",", ":"))}]
    if screenshot_url:
        content.append({"type": "text", "text": "Screenshot for control selection:"})
        content.append({"type": "image_url", "image_url": {"url": screenshot_url}})
    if scroll_screenshot_urls:
        content.append({"type": "text", "text": "Additional screenshots captured after three downward scrolls:"})
        for index, url in enumerate(scroll_screenshot_urls, 1):
            content.append({"type": "text", "text": f"Scroll screenshot {index}:"})
            content.append({"type": "image_url", "image_url": {"url": url}})
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": content},
    ]


def _build_llm_selection_messages_with_budget(
    intent: str,
    screenshot_url: str | None,
    scroll_screenshot_urls: list[str],
    candidates: list[dict[str, str]],
    limit: int,
) -> list[dict[str, Any]]:
    return _build_llm_selection_messages(
        intent,
        screenshot_url or "",
        scroll_screenshot_urls,
        candidates,
        limit,
    )


def _parse_llm_selection_response(raw_text: str, limit: int) -> list[dict[str, str]]:
    text = (raw_text or "").strip()
    if not text:
        return []
    fenced_match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced_match:
        text = fenced_match.group(1).strip()
    try:
        payload = json.loads(text)
    except Exception:
        return []

    selected = payload.get("selected_targets") if isinstance(payload, dict) else payload
    if not isinstance(selected, list):
        return []

    normalized: list[dict[str, str]] = []
    for item in selected[:limit]:
        if not isinstance(item, dict):
            continue
        candidate: dict[str, str] = {}
        for key in ("xpath", "resource-id", "text", "content-desc", "class", "reason"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                candidate[key] = value.strip()
        if candidate:
            normalized.append(candidate)
    return normalized


def _select_related_ui_summary_with_llm(
    snapshot: dict[str, Any] | None,
    intent: str,
    candidates: list[dict[str, str]],
    limit: int,
    base_url: str | None = None,
    api_key: str | None = None,
    model: str | None = None,
) -> tuple[list[dict[str, str]], int]:
    _debug_log("⚠️ Entering LLM-based control selection (static matching failed)")
    if not snapshot or not candidates:
        return [], 0

    client = _load_client(base_url, api_key)
    if client is None:
        return [], 0

    screenshot_url = _snapshot_image_url(snapshot)
    if not screenshot_url:
        return [], 0
    scroll_screenshot_urls = _snapshot_scroll_image_urls(snapshot)

    compressed_screenshot_url, compressed_scroll_screenshot_urls, compact_candidates, total_bytes = _prepare_llm_selection_inputs(
        snapshot,
        intent,
        candidates,
        limit,
    )
    _debug_log(f"Prepared LLM selection inputs: total={_format_budget_usage(total_bytes, LLM_SELECTION_INPUT_BYTE_LIMIT)}")

    try:
        response = get_openai_chat_completion(
            client,
            model=model or _default_model(),
            messages=_build_llm_selection_messages_with_budget(
                intent,
                compressed_screenshot_url,
                compressed_scroll_screenshot_urls,
                compact_candidates,
                limit,
            ),
            temperature=0,
        )
        token_usage = extract_openai_token_usage(response)
        raw_content = getattr(response.choices[0].message, "content", "") or ""
        _debug_log(f"Raw LLM control-selection response: {raw_content!r}")
        selected = _parse_llm_selection_response(raw_content, limit)
        if selected:
            return selected, token_usage
    except Exception as exc:
        _debug_log(f"LLM control selection failed: {exc}")

    return [], 0


def _iter_snapshot_page_sources(snapshot: dict[str, Any] | None) -> list[str]:
    if not snapshot:
        return []

    ui = snapshot.get("ui") if isinstance(snapshot, dict) else {}
    if not isinstance(ui, dict):
        return []

    page_sources = ui.get("page_sources")
    if isinstance(page_sources, list):
        collected = [str(source or "") for source in page_sources if str(source or "").strip()]
        if collected:
            return collected

    raw_page_source = str(ui.get("raw_page_source") or "")
    return [raw_page_source] if raw_page_source else []


def _collect_ui_summary_candidates(snapshot: dict[str, Any] | None) -> list[dict[str, str]]:
    if not snapshot:
        return []

    summaries: list[tuple[int, dict[str, str]]] = []
    seen_signatures: set[str] = set()
    for page_source in _iter_snapshot_page_sources(snapshot):
        try:
            root = ET.fromstring(page_source)
        except Exception:
            continue

        for node in root.iter():
            resource_id = str(node.attrib.get("resource-id", "") or "").strip()
            text = str(node.attrib.get("text", "") or "").strip()
            content_desc = str(node.attrib.get("content-desc", "") or "").strip()
            class_name = str(node.attrib.get("class", "") or "").strip()

            if not any([resource_id, text, content_desc, class_name]):
                continue

            summary: dict[str, str] = {}
            xpath = _build_xpath_hint_from_node(node)
            if xpath:
                summary["xpath"] = xpath
            if resource_id:
                summary["resource-id"] = resource_id
            if text:
                summary["text"] = text
            if content_desc:
                summary["content-desc"] = content_desc
            if class_name:
                summary["class"] = class_name

            signature = "|".join(summary.get(key, "") for key in ("xpath", "resource-id", "text", "content-desc", "class"))
            if signature in seen_signatures:
                continue
            seen_signatures.add(signature)

            priority = 0
            if text:
                priority += 3
            if resource_id:
                priority += 2
            if content_desc:
                priority += 1
            summaries.append((priority, summary))

    summaries.sort(key=lambda item: item[0], reverse=True)
    return [item[1] for item in summaries]


def _extract_ui_summary(snapshot: dict[str, Any] | None, limit: int = 15) -> list[dict[str, str]]:
    return _collect_ui_summary_candidates(snapshot)[:limit]


def _extract_ui_summary_from_page_source(page_source: str | None, limit: int | None = 15) -> list[dict[str, str]]:
    if not page_source:
        return []

    try:
        root = ET.fromstring(page_source)
    except Exception:
        return []

    summaries: list[tuple[int, dict[str, str]]] = []
    for node in root.iter():
        resource_id = str(node.attrib.get("resource-id", "") or "").strip()
        text = str(node.attrib.get("text", "") or "").strip()
        content_desc = str(node.attrib.get("content-desc", "") or "").strip()
        class_name = str(node.attrib.get("class", "") or "").strip()

        if not any([resource_id, text, content_desc, class_name]):
            continue

        summary: dict[str, str] = {}
        if resource_id:
            summary["resource-id"] = resource_id
        if text:
            summary["text"] = text
        if content_desc:
            summary["content-desc"] = content_desc
        if class_name:
            summary["class"] = class_name

        xpath = _build_xpath_hint_from_node(node)
        if xpath:
            summary["xpath"] = xpath

        priority = 0
        if text:
            priority += 3
        if resource_id:
            priority += 2
        if content_desc:
            priority += 1
        summaries.append((priority, summary))

    summaries.sort(key=lambda item: item[0], reverse=True)
    items = [item[1] for item in summaries]
    return items[:limit] if limit is not None else items


def _build_xpath_hint_from_node(node: ET.Element) -> str:
    resource_id = str(node.attrib.get("resource-id", "") or "").strip()
    class_name = str(node.attrib.get("class", "") or "").strip()
    text = str(node.attrib.get("text", "") or "").strip()
    content_desc = str(node.attrib.get("content-desc", "") or "").strip()

    if not class_name:
        return ""

    predicates: list[str] = []
    if resource_id:
        predicates.append(f"@resource-id={resource_id!r}")
    if text:
        predicates.append(f"@text={text!r}")
    if content_desc:
        predicates.append(f"@content-desc={content_desc!r}")

    if predicates:
        return f"//{class_name}[{' and '.join(predicates)}]"
    return f"//{class_name}"


def _snapshot_page_source(snapshot: dict[str, Any] | None) -> str:
    sources = _iter_snapshot_page_sources(snapshot)
    return sources[0] if sources else ""


def _extract_related_ui_summary(snapshot: dict[str, Any] | None, intent: str, limit: int = 8) -> tuple[list[dict[str, str]], int]:
    summaries = _collect_ui_summary_candidates(snapshot)
    if not summaries:
        summaries = _extract_ui_summary_from_page_source(_snapshot_page_source(snapshot), limit=None)
    if not summaries:
        return [], 0

    summaries = _filter_non_container_ui_candidates(summaries)
    if not summaries:
        return [], 0

    intent_tokens = _tokenize_loose(intent)
    if not intent_tokens:
        # 没有 intent 词元时直接返回全部候选
        return summaries, 0

    intent_text = _normalize_text(intent)
    intent_counter = Counter(_tokenize_loose(intent))

    scored: list[tuple[int, dict[str, str]]] = []
    for summary in summaries:
        blob = " ".join(summary.get(key, "") for key in ("xpath", "resource-id", "text", "content-desc", "class"))
        tokens = _tokenize_loose(blob)
        if not tokens:
            continue

        token_counter = Counter(tokens)
        overlap_tokens = intent_tokens & tokens
        overlap_weight = sum(intent_counter[token] * token_counter[token] for token in overlap_tokens)
        soft_overlap_weight = _soft_token_overlap_score(intent_tokens - overlap_tokens, tokens - overlap_tokens)
        if overlap_weight <= 0 and soft_overlap_weight <= 0:
            continue

        score = overlap_weight * 12 + soft_overlap_weight * 4
        exact_text_hits = 0
        for key in ("resource-id", "text", "content-desc", "class"):
            value = summary.get(key, "")
            if not value:
                continue
            value_tokens = _tokenize_loose(value)
            exact_text_hits += sum(intent_counter[token] for token in value_tokens if token in intent_counter)
        score += exact_text_hits * 3
        if intent_text and any(token in intent_text for token in tokens):
            score += 2
        if summary.get("text"):
            score += 3
        if summary.get("resource-id"):
            score += 2
        if summary.get("content-desc"):
            score += 1
        scored.append((score, summary))

    scored.sort(key=lambda item: item[0], reverse=True)
    related = [item[1] for item in scored[:limit]]
    if related:
        return related, 0

    SKIP_LLM_CANDIDATE_COUNT = 10
    SKIP_LLM_TEXT_BYTES = 4096  # 4KB
    total_text_bytes = sum(len(str(summary)) for summary in summaries)
    if len(summaries) <= SKIP_LLM_CANDIDATE_COUNT or total_text_bytes <= SKIP_LLM_TEXT_BYTES:
        _debug_log(f"Skipping LLM control selection: candidates={len(summaries)}, text_bytes={total_text_bytes}")
        return summaries, 0

    # 否则调用 LLM 兜底
    llm_selected, token_usage = _select_related_ui_summary_with_llm(snapshot, intent, summaries, limit)
    if llm_selected:
        return llm_selected[:limit], token_usage

    # 最终 fallback：返回全部
    return summaries, 0


def _format_ui_summary_for_prompt(ui_summary: list[dict[str, str]], limit: int = 15) -> str:
    if not ui_summary:
        return "[]"

    lines: list[str] = ["["]
    for index, item in enumerate(ui_summary[:limit], 1):
        parts = []
        for key in ("resource-id", "text", "content-desc", "class", "xpath"):
            value = item.get(key)
            if value:
                parts.append(f'{key}={value!r}')
        lines.append(f"  {index}. " + ", ".join(parts))
    lines.append("]")
    return "\n".join(lines)