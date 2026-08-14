import json
import os
import re
import time
import xml.etree.ElementTree as ET


def sanitize_filename(name):
    if not name:
        return "unnamed"
    name = re.sub(r'[<>:"/\\|?*]+', '_', str(name))
    name = name.strip().strip('.')
    return name or "unnamed"


def get_test_case_stem(file_path):
    base_name = os.path.basename(file_path)
    stem, _ = os.path.splitext(base_name)
    return sanitize_filename(stem)


def create_action_output_paths(file_path, artifact_root=None):
    test_case_stem = get_test_case_stem(file_path)
    if artifact_root:
        base_dir = os.path.abspath(os.path.join(artifact_root, test_case_stem))
        information_dir = os.path.join(base_dir, "information")
    else:
        test_case_dir = os.path.dirname(os.path.abspath(file_path))
        information_dir = os.path.abspath(os.path.join(test_case_dir, "..", "information"))
    os.makedirs(information_dir, exist_ok=True)
    screenshot_dir = os.path.join(information_dir, f"screenshot-{test_case_stem}")
    json_path = os.path.join(information_dir, f"{test_case_stem}.json")
    os.makedirs(screenshot_dir, exist_ok=True)
    return test_case_stem, screenshot_dir, json_path


def create_generated_output_paths(file_path, artifact_root=None):
    test_case_stem = get_test_case_stem(file_path)
    if artifact_root:
        output_dir = os.path.abspath(os.path.join(artifact_root, test_case_stem))
    else:
        output_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(file_path)), "..", "output"))
    os.makedirs(output_dir, exist_ok=True)
    txt_path = os.path.join(output_dir, f"{test_case_stem}.txt")
    json_path = os.path.join(output_dir, f"{test_case_stem}.json")
    return test_case_stem, txt_path, json_path


def safe_text_for_json(value):
    if value is None:
        return None
    return str(value)


def _safe_swipe_down_screen(driver):
    try:
        from action_handlers import swipe_down_screen

        swipe_down_screen(driver)
        return True
    except Exception as exc:
        print(f"❌ Failed to swipe down while collecting UI tree: {exc}")
        return False


def _safe_swipe_up_screen(driver):
    try:
        size = driver.get_window_size()
        start_x = size['width'] / 2
        start_y = size['height'] * 0.5
        end_y = size['height'] * 0.8

        from selenium.webdriver import ActionChains
        from selenium.webdriver.common.actions import interaction
        from selenium.webdriver.common.actions.action_builder import ActionBuilder
        from selenium.webdriver.common.actions.pointer_input import PointerInput

        actions = ActionChains(driver)
        pointer = PointerInput(interaction.POINTER_TOUCH, "touch")
        actions.w3c_actions = ActionBuilder(driver, mouse=pointer)
        actions.w3c_actions.pointer_action.move_to_location(start_x, start_y)
        actions.w3c_actions.pointer_action.pointer_down()
        actions.w3c_actions.pointer_action.pause(0.3)
        actions.w3c_actions.pointer_action.move_to_location(start_x, end_y)
        actions.w3c_actions.pointer_action.release()
        actions.perform()
        return True
    except Exception as exc:
        print(f"❌ Failed to swipe up while restoring UI tree: {exc}")
        return False


def collect_ui_tree(
    driver,
    extra_scrolls=0,
    scroll_delay=1,
    restore_after_scroll=True,
    scroll_screenshot_dir=None,
    scroll_screenshot_prefix=None,
):
    page_source = driver.page_source
    page_sources = [page_source]
    scroll_screenshot_paths = []

    last_page_source = page_source
    performed_scrolls = 0
    for _ in range(max(int(extra_scrolls or 0), 0)):
        if not _safe_swipe_down_screen(driver):
            break
        performed_scrolls += 1
        time.sleep(scroll_delay)
        current_page_source = driver.page_source
        if current_page_source == last_page_source:
            break
        page_sources.append(current_page_source)
        if scroll_screenshot_dir:
            os.makedirs(scroll_screenshot_dir, exist_ok=True)
            prefix = scroll_screenshot_prefix or "scroll"
            scroll_index = len(scroll_screenshot_paths) + 1
            scroll_screenshot_name = f"{prefix}_scroll_{scroll_index:02d}.png"
            scroll_screenshot_path = os.path.join(scroll_screenshot_dir, scroll_screenshot_name)
            try:
                driver.save_screenshot(scroll_screenshot_path)
                scroll_screenshot_paths.append(scroll_screenshot_path)
            except Exception as exc:
                print(f"❌ Failed to save scroll screenshot {scroll_index}: {exc}")
        last_page_source = current_page_source

    if restore_after_scroll:
        for _ in range(performed_scrolls):
            if not _safe_swipe_up_screen(driver):
                break
            time.sleep(0.3)

    try:
        root = ET.fromstring(page_source)
    except Exception as e:
        print(f"❌ Failed to parse page source for JSON snapshot: {e}")
        return {
            "raw_page_source": page_source,
            "elements": [],
            "page_sources": page_sources,
            "scroll_screenshot_paths": scroll_screenshot_paths,
        }

    elements = []

    def _walk(node, depth=0):
        elements.append({
            "depth": depth,
            "tag": node.tag,
            "attributes": {k: safe_text_for_json(v) for k, v in node.attrib.items()},
        })
        for child in list(node):
            _walk(child, depth + 1)

    _walk(root)
    return {
        "raw_page_source": page_source,
        "elements": elements,
        "page_sources": page_sources,
        "scroll_screenshot_paths": scroll_screenshot_paths,
    }


def write_json_snapshot(json_path, snapshots):
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(snapshots, f, ensure_ascii=False, indent=2)


def capture_step_snapshot(
    driver,
    screenshot_dir,
    json_snapshots,
    test_case_stem,
    step_index,
    step_type,
    step_label,
    line,
    extra_scrolls=0,
    scroll_delay=1,
    restore_after_scroll=True,
):
    screenshot_name = f"step_{step_index:03d}_{sanitize_filename(step_type)}_{sanitize_filename(step_label)}.png"
    screenshot_path = os.path.join(screenshot_dir, screenshot_name)
    driver.save_screenshot(screenshot_path)

    snapshot = {
        "step_index": step_index,
        "test_case": test_case_stem,
        "step_type": step_type,
        "step_label": step_label,
        "line": line,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "current_package": safe_text_for_json(getattr(driver, "current_package", None)),
        "current_activity": safe_text_for_json(getattr(driver, "current_activity", None)),
        "screenshot_path": screenshot_path,
        "ui": collect_ui_tree(
            driver,
            extra_scrolls=extra_scrolls,
            scroll_delay=scroll_delay,
            restore_after_scroll=restore_after_scroll,
            scroll_screenshot_dir=screenshot_dir,
            scroll_screenshot_prefix=f"step_{step_index:03d}_{sanitize_filename(step_type)}_{sanitize_filename(step_label)}",
        ),
    }
    json_snapshots.append(snapshot)