from __future__ import annotations

import sys
import types


def _ensure_appium_stub() -> None:
    """Provide a minimal appium stub when the dependency is unavailable."""
    if "appium" in sys.modules:
        return

    appium = types.ModuleType("appium")
    webdriver = types.ModuleType("appium.webdriver")
    common = types.ModuleType("appium.webdriver.common")
    appiumby = types.ModuleType("appium.webdriver.common.appiumby")

    class _DummyAppiumBy:
        XPATH = "xpath"
        ID = "id"

    appiumby.AppiumBy = _DummyAppiumBy
    common.appiumby = appiumby
    webdriver.common = common
    appium.webdriver = webdriver

    sys.modules["appium"] = appium
    sys.modules["appium.webdriver"] = webdriver
    sys.modules["appium.webdriver.common"] = common
    sys.modules["appium.webdriver.common.appiumby"] = appiumby


_ensure_appium_stub()

from assertion_checker import (
    _describe_format_violation,
    _find_malformed_xpath_predicate_quotes,
    _repair_xpath_predicate_quotes,
    _has_malformed_same_screen_quotes,
)
from test_case_runner import _apply_static_quote_escape_fallback, _is_quote_format_failure_reason


def _assert_contains(text: str, needle: str) -> None:
    if needle not in text:
        raise AssertionError(f"expected {needle!r} in {text!r}")


def main() -> None:
    dq = '"'
    sq = "'"

    cases = [
        (
            f'check_dark_mode({dq}dark mode{dq})',
            False,
            None,
        ),
        (
            f"check_dark_mode({sq}dark mode{sq})",
            False,
            None,
        ),
        (
            f'check_dark_mode({dq}dark \\\"mode\\\"{dq})',
            False,
            None,
        ),
        (
            f'check_dark_mode({dq}dark {dq}mode{dq}{dq})',
            True,
            "引号格式错误",
        ),
        (
            f"check_light_mode({sq}light {sq}mode{sq}{sq})",
            True,
            "引号格式错误",
        ),
        (
            'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' and @text=\'Passwords don\'t match\']"); assert elem.is_displayed()',
            False,
            None,
        ),
        (
            'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' and @text=\'Passwords don\'t match\']"); assert elem.is_displayed()',
            False,
            None,
        ),
        (
            'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' and @text=\'The user name \\\"Aurora\\\" is not available. Please choose a different name.\']"); assert elem.is_displayed()',
            False,
            None,
        ),
        (
            f'check_elements_same_screen(driver, [{dq}//android.widget.TextView[@text=\\\"Breaking News\\\"]{dq}])',
            False,
            None,
        ),
        (
            f'check_elements_same_screen(driver, [{dq}//android.widget.TextView[@text={dq}Breaking News{dq}]{dq}])',
            True,
            "引号格式错误",
        ),
        (
            f"check_elements_same_screen(driver, [{sq}//android.widget.TextView[@text={sq}Breaking News{sq}]{sq}])",
            True,
            "引号格式错误",
        ),
    ]

    for assertion, expected_malformed, expected_violation in cases:
        malformed = _has_malformed_same_screen_quotes(assertion)
        if assertion.startswith("check_elements_same_screen("):
            assert malformed is expected_malformed, (assertion, malformed, expected_malformed)

        violation = _describe_format_violation(assertion)
        if expected_violation is None:
            if violation == "引号格式错误：断言中的字符串包含未转义的同类引号，请检查单引号/双引号嵌套并重新生成":
                raise AssertionError(f"unexpected quote violation for {assertion!r}")
        else:
            _assert_contains(violation, expected_violation)

    malformed_xpath = 'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' and @text=\'Passwords don\'t match\']"); assert elem.is_displayed()'
    details = _find_malformed_xpath_predicate_quotes(malformed_xpath)
    assert details == ["'Passwords don't match'"], details

    violation = _describe_format_violation(malformed_xpath)
    _assert_contains(violation, "具体问题文本：'Passwords don't match'")

    repaired_single_quote = _repair_xpath_predicate_quotes(malformed_xpath)
    assert _find_malformed_xpath_predicate_quotes(repaired_single_quote) == [], repaired_single_quote

    malformed_double_quote = (
        'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' '
        'and @text="The user name "Aurora" is not available. Please choose a different name."]"); assert elem.is_displayed()'
    )
    repaired_double_quote = _repair_xpath_predicate_quotes(malformed_double_quote)
    assert _find_malformed_xpath_predicate_quotes(repaired_double_quote) == [], repaired_double_quote

    assert _is_quote_format_failure_reason(
        '格式违规：\'elem = driver.find_element(By.XPATH, "//a[@text=\'broken\']"); assert elem.is_displayed()\' -> 引号格式错误：XPath 字符串中的文本内容包含未转义的同类引号'
    )
    assert not _is_quote_format_failure_reason('定位失败：元素未找到')

    fallback_input = [
        'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' and @text=\'Passwords don\'t match\']"); assert elem.is_displayed()',
        'elem = driver.find_element(By.XPATH, "//android.widget.TextView[@resource-id=\'org.wikipedia:id/textinput_error\' and @text=\'Passwords don\'t match\']"); assert elem.is_displayed()',
    ]
    fallback_output = _apply_static_quote_escape_fallback(fallback_input)
    assert len(fallback_output) == 1, fallback_output
    assert _find_malformed_xpath_predicate_quotes(fallback_output[0]) == [], fallback_output[0]

    print("all quote-format cases passed")


if __name__ == "__main__":
    main()