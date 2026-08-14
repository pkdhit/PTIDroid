import os
import time
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from action_handlers import find_element_with_scroll, swipe_down_screen


def is_image_dark(image_path, threshold=70):
    try:
        with Image.open(image_path) as img:
            grayscale = img.convert('L')
            brightness = np.mean(np.array(grayscale))
            print(f"图片平均亮度: {brightness:.2f} (阈值: {threshold})")
            return brightness < threshold
    except Exception as e:
        print(f"读取图片失败: {e}")
        return False


def is_image_light(image_path, threshold=180):
    try:
        with Image.open(image_path) as img:
            grayscale = img.convert('L')
            brightness = np.mean(np.array(grayscale))
            print(f"图片平均亮度: {brightness:.2f} (阈值: {threshold})")
            return brightness >= threshold
    except Exception as e:
        print(f"读取图片失败: {e}")
        return False


def check_dark_mode(driver, rid):
    temp_screenshot = "temp_dark_mode_check_executor.png"
    print(f"Taking screenshot for dark mode check on element '{rid}'...")
    driver.save_screenshot(temp_screenshot)
    if os.path.exists(temp_screenshot):
        print("Analyzing screenshot...")
        is_dark = is_image_dark(temp_screenshot)
        os.remove(temp_screenshot)
        print("Temporary screenshot removed.")
        assert is_dark, "Dark mode check failed: UI is not dark."
    else:
        raise FileNotFoundError("Failed to save screenshot for dark mode check.")


def check_light_mode(driver, rid):
    temp_screenshot = "temp_light_mode_check_executor.png"
    print(f"Taking screenshot for light mode check on element '{rid}'...")
    driver.save_screenshot(temp_screenshot)
    if os.path.exists(temp_screenshot):
        print("Analyzing screenshot...")
        is_light = is_image_light(temp_screenshot)
        os.remove(temp_screenshot)
        print("Temporary screenshot removed.")
        assert is_light, "Light mode check failed: UI is not light."
    else:
        raise FileNotFoundError("Failed to save screenshot for light mode check.")


def check_elements_same_screen(driver, xpath_list, timeout=10):
    if not xpath_list:
        raise ValueError("xpath_list must not be empty")
    found = []
    for xpath in xpath_list:
        elem, exists = find_element_with_scroll(driver, xpath=xpath, max_scrolls=timeout)
        if not exists:
            raise AssertionError(f"Element not present on the current screen: {xpath}")
        found.append(elem)
    return found


def _is_checked_widget(node):
    if node is None:
        return False
    checked = node.attrib.get('checked', 'false').lower() == 'true'
    if not checked:
        return False
    class_name = node.attrib.get('class', '').lower()
    return any(keyword in class_name for keyword in ('switch', 'checkbox', 'checkedtextview', 'toggle'))


def check_row_checked_by_text(driver, target_text, max_scrolls=5):
    if not target_text:
        raise ValueError("target_text must not be empty")
    last_page_source = None
    for _ in range(max_scrolls + 1):
        page_source = driver.page_source
        if page_source == last_page_source:
            break
        last_page_source = page_source
        try:
            root = ET.fromstring(page_source)
        except Exception as e:
            raise AssertionError(f"Failed to parse page source XML: {e}") from e

        parent_map = {child: parent for parent in root.iter() for child in parent}
        candidate_nodes = [node for node in root.iter() if node.attrib.get('text', '') == target_text]

        for node in candidate_nodes:
            current = node
            while current is not None:
                if _is_checked_widget(current):
                    print(f"Found checked widget for '{target_text}' on node: {current.attrib.get('class', '')}")
                    return True
                for descendant in current.iter():
                    if descendant is current:
                        continue
                    if _is_checked_widget(descendant):
                        print(f"Found checked widget for '{target_text}' in container: {descendant.attrib.get('class', '')}")
                        return True
                current = parent_map.get(node) if current is node else parent_map.get(current)

        swipe_down_screen(driver)
        time.sleep(1)

    raise AssertionError(f"Checked switch/checkbox not found for row text: {target_text}")


def check_current_app_package(driver, expected_package, expected_activity=None, timeout=10):
    if not expected_package:
        raise ValueError("expected_package must not be empty")
    last_package = None
    last_activity = None
    for _ in range(timeout + 1):
        current_package = None
        current_activity = None
        try:
            current_package = driver.current_package
        except Exception:
            pass
        try:
            current_activity = driver.current_activity
        except Exception:
            pass
        print(f"Current app package: {current_package}, activity: {current_activity}; expected package: {expected_package}")
        if current_package == expected_package and (expected_activity is None or current_activity == expected_activity):
            return True
        if current_package == last_package and current_activity == last_activity:
            time.sleep(1)
            continue
        last_package = current_package
        last_activity = current_activity
        time.sleep(1)
    if expected_activity is None:
        raise AssertionError(f"Current app package mismatch: expected '{expected_package}', got '{last_package}'")
    raise AssertionError(
        f"Current app mismatch: expected package '{expected_package}' and activity '{expected_activity}', got package '{last_package}' and activity '{last_activity}'"
    )