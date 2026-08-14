import re
import time

from selenium.webdriver import ActionChains
from selenium.webdriver.common.actions import interaction
from selenium.webdriver.common.actions.action_builder import ActionBuilder
from selenium.webdriver.common.actions.pointer_input import PointerInput
from selenium.webdriver.common.by import By


def _parse_bounds(bounds_text):
    if not bounds_text:
        return None
    match = re.match(r"\[(\d+),(\d+)\]\[(\d+),(\d+)\]", bounds_text.strip())
    if not match:
        return None
    x1, y1, x2, y2 = map(int, match.groups())
    return x1, y1, x2, y2


def _normalize_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def swipe_element(driver, element, x_offset=0, y_offset=0, duration_ms=200):
    def _get_rect(target_element):
        try:
            rect = target_element.rect
            if rect:
                return rect
        except Exception:
            pass
        return {
            'x': target_element.location['x'],
            'y': target_element.location['y'],
            'width': target_element.size['width'],
            'height': target_element.size['height'],
        }

    rect = _get_rect(element)
    width = max(int(rect['width']), 1)
    height = max(int(rect['height']), 1)
    left = int(rect['x'])
    top = int(rect['y'])

    print(
        "[SWIPE_DEBUG] swipe_element prepared: "
        f"left={left}, top={top}, width={width}, height={height}, "
        f"x_offset={x_offset}, y_offset={y_offset}, duration_ms={duration_ms}"
    )

    if x_offset > 0:
        direction = 'right'
        percent = min(max(abs(x_offset) / width, 0.2), 0.9)
    elif x_offset < 0:
        direction = 'left'
        percent = min(max(abs(x_offset) / width, 0.2), 0.9)
    elif y_offset > 0:
        direction = 'down'
        percent = min(max(abs(y_offset) / height, 0.2), 0.9)
    else:
        direction = 'up'
        percent = min(max(abs(y_offset) / height if y_offset else 0.6, 0.2), 0.9)

    print(
        "[SWIPE_DEBUG] trying mobile: swipeGesture: "
        f"direction={direction}, percent={percent:.3f}, "
        f"bounds=({left},{top},{width},{height})"
    )

    if direction in {'right', 'left'}:
        start_x = int(left + width * 0.2) if direction == 'right' else int(left + width * 0.8)
        end_x = int(left + width * 0.8) if direction == 'right' else int(left + width * 0.2)
        start_y = end_y = int(top + height * 0.5)
    else:
        start_y = int(top + height * 0.2) if direction == 'down' else int(top + height * 0.8)
        end_y = int(top + height * 0.8) if direction == 'down' else int(top + height * 0.2)
        start_x = end_x = int(left + width * 0.5)

    print(
        "[SWIPE_DEBUG] planned absolute swipe path: "
        f"start=({start_x},{start_y}), end=({end_x},{end_y}), duration_ms={duration_ms}"
    )

    try:
        driver.execute_script(
            'mobile: shell',
            {
                'command': 'input',
                'args': [
                    'swipe',
                    str(start_x),
                    str(start_y),
                    str(end_x),
                    str(end_y),
                    str(int(duration_ms)),
                ],
            },
        )
        print("[SWIPE_DEBUG] mobile: shell input swipe executed successfully")
        return
    except Exception as exc:
        print(f"[SWIPE_DEBUG] mobile: shell input swipe failed: {exc}")

    try:
        driver.execute_script(
            'mobile: swipeGesture',
            {
                'left': left,
                'top': top,
                'width': width,
                'height': height,
                'direction': direction,
                'percent': percent,
            },
        )
        print("[SWIPE_DEBUG] mobile: swipeGesture executed successfully")
        return
    except Exception as exc:
        print(f"[SWIPE_DEBUG] mobile: swipeGesture failed: {exc}")

    start_x = left + width * 0.5
    start_y = top + height * 0.5
    end_x = start_x + x_offset
    end_y = start_y + y_offset

    print(
        "[SWIPE_DEBUG] falling back to W3C touch action: "
        f"start=({int(start_x)},{int(start_y)}), end=({int(end_x)},{int(end_y)}), duration_ms={duration_ms}"
    )

    actions = ActionChains(driver)
    pointer = PointerInput(interaction.POINTER_TOUCH, "touch")
    actions.w3c_actions = ActionBuilder(driver, mouse=pointer)
    actions.w3c_actions.pointer_action.move_to_location(int(start_x), int(start_y))
    actions.w3c_actions.pointer_action.pointer_down()
    actions.w3c_actions.pointer_action.pause(duration_ms / 1000)
    actions.w3c_actions.pointer_action.move_to_location(int(end_x), int(end_y))
    actions.w3c_actions.pointer_action.release()
    try:
        actions.perform()
        print("[SWIPE_DEBUG] W3C touch action executed successfully")
    except Exception as exc:
        print(f"[SWIPE_DEBUG] W3C touch action failed: {exc}")
        raise


def swipe_down_screen(driver, duration_ms=300):
    size = driver.get_window_size()
    start_x = size['width'] / 2
    start_y = size['height'] * 0.9
    end_y = size['height'] * 0.6

    actions = ActionChains(driver)
    pointer = PointerInput(interaction.POINTER_TOUCH, "touch")
    actions.w3c_actions = ActionBuilder(driver, mouse=pointer)
    actions.w3c_actions.pointer_action.move_to_location(start_x, start_y)
    actions.w3c_actions.pointer_action.pointer_down()
    actions.w3c_actions.pointer_action.pause(duration_ms / 1000)
    actions.w3c_actions.pointer_action.move_to_location(start_x, end_y)
    actions.w3c_actions.pointer_action.release()
    actions.perform()


def tap_at_position(driver, x, y, press_duration_ms=1):
    try:
        driver.execute_script("mobile: shell", {"command": "input", "args": ["tap", str(int(x)), str(int(y))]})
    except Exception:
        try:
            driver.execute_script("mobile: clickGesture", {"x": int(x), "y": int(y)})
        except Exception:
            actions = ActionChains(driver)
            pointer = PointerInput(interaction.POINTER_TOUCH, "touch")
            actions.w3c_actions = ActionBuilder(driver, mouse=pointer)
            actions.w3c_actions.pointer_action.move_to_location(x, y)
            actions.w3c_actions.pointer_action.pointer_down()
            actions.w3c_actions.pointer_action.release()
            actions.perform()


def long_press_at_position(driver, x, y, press_duration_ms=2000):
    try:
        driver.execute_script("mobile: longClickGesture", {"x": int(x), "y": int(y), "duration": int(press_duration_ms)})
    except Exception:
        tap_at_position(driver, x, y, press_duration_ms=press_duration_ms)


def hide_keyboard_if_open(driver):
    try:
        if driver.is_keyboard_shown():
            driver.hide_keyboard()
    except Exception:
        try:
            driver.hide_keyboard()
        except Exception:
            pass


def _find_node_by_field(page_source, field_value, field_name='text', exact=True):
    if not page_source or not field_value:
        return None

    import xml.etree.ElementTree as ET

    try:
        root = ET.fromstring(page_source)
    except Exception as e:
        print(f"❌ Failed to parse page source XML: {e}")
        return None

    parent_map = {child: parent for parent in root.iter() for child in parent}
    target_value = _normalize_text(field_value)

    for node in root.iter():
        candidate = _normalize_text(node.attrib.get(field_name, ''))
        matched = candidate == target_value if exact else target_value in candidate
        if not matched and field_name == 'text':
            content_desc = _normalize_text(node.attrib.get('content-desc', ''))
            matched = content_desc == target_value if exact else target_value in content_desc

        if matched:
            target_node = node
            while target_node is not None:
                bounds = _parse_bounds(target_node.attrib.get('bounds', ''))
                if target_node.attrib.get('clickable', 'false') == 'true' and bounds:
                    return target_node, bounds
                target_node = parent_map.get(target_node)

            bounds = _parse_bounds(node.attrib.get('bounds', ''))
            if bounds:
                return node, bounds

    return None


def click_by_field_value(driver, field_value, field_name='text', exact=True, max_scrolls=5, long_press=False):
    last_page_source = None
    for _ in range(max_scrolls + 1):
        page_source = driver.page_source
        if page_source == last_page_source:
            break
        last_page_source = page_source

        found = _find_node_by_field(page_source, field_value, field_name=field_name, exact=exact)
        if found:
            _, bounds = found
            x1, y1, x2, y2 = bounds
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)
            if long_press:
                print(f"Found '{field_value}' by {field_name} with bounds {bounds}, long-pressing at ({center_x}, {center_y})")
                long_press_at_position(driver, center_x, center_y)
            else:
                print(f"Found '{field_value}' by {field_name} with bounds {bounds}, tapping at ({center_x}, {center_y})")
                tap_at_position(driver, center_x, center_y)
            return True

        swipe_down_screen(driver)
        time.sleep(1)

    return False


def find_element_with_scroll(driver, resource_id=None, xpath=None, max_scrolls=6):
    locator = None
    if xpath and xpath.strip() != '':
        locator = (By.XPATH, xpath.strip())
        print(f" Locating element by XPath: {xpath}")
    elif resource_id and resource_id.strip() != '':
        locator = (By.ID, resource_id.strip())
        print(f" Locating element by resource-id: {resource_id}")
    else:
        raise ValueError("At least resource-id or xpath must be provided.")

    last_page_source = None
    for _ in range(max_scrolls + 1):
        try:
            element = driver.find_element(locator[0], locator[1])
            return element, True
        except Exception:
            current_page_source = driver.page_source
            if current_page_source == last_page_source:
                break
            last_page_source = current_page_source
            swipe_down_screen(driver)
            time.sleep(1)

    return None, False