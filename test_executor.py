from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from appium import webdriver
from appium.options.android import UiAutomator2Options

from test_case_runner import execute_test_case
from snapshot_utils import get_test_case_stem


DESIRED_CAPS = {
    'platformName': 'Android',
    'platformVersion': '13.0',
    'deviceName': 'emulator-5554',
    'autoGrantPermissions': True,
    'noReset': True,
    'newCommandTimeout': 600,
}


def load_app_info(test_file):
    app_package = None
    app_activity = None

    with open(test_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.startswith('appPackage:'):
                app_package = line.split(':', 1)[1].strip()
            elif line.startswith('appActivity:'):
                app_activity = line.split(':', 1)[1].strip()
            elif line.startswith('---'):
                break

    return app_package, app_activity


def build_driver(test_file):
    app_package, app_activity = load_app_info(test_file)
    if not app_package or not app_activity:
        raise ValueError('Error: appPackage or appActivity not found in test file.')

    desired_caps = dict(DESIRED_CAPS)
    desired_caps['appPackage'] = app_package
    desired_caps['appActivity'] = app_activity

    options = UiAutomator2Options()
    for key, value in desired_caps.items():
        options.set_capability(key, value)

    return options


def parse_args(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(description='Run assertion generation for a single test case.')
    parser.add_argument('test_file', help='Path to the test case file.')
    parser.add_argument(
        '--baseline',
        choices=['none', 'androb2o'],
        default='none',
        help='Use a specific baseline for assertion generation (none or androb2o).',
    )
    parser.add_argument(
        '--ablation',
        choices=['none', 'no_planner', 'no_selector', 'no_checker', 'no_classifier'],
        default='none',
        help='Disable a specific module for ablation experiments.',
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])
    test_file = args.test_file
    test_case_stem = get_test_case_stem(test_file)
    output_root = Path('baselines') / 'output' if args.baseline == 'androb2o' else Path('output')
    output_dir = output_root / test_case_stem
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / 'result.txt'
    options = build_driver(test_file)

    driver = None
    start_time = time.perf_counter()
    try:
        with result_file.open('w', encoding='utf-8') as result_logger:
            print('Connecting to Appium server...')
            result_logger.write('--- Test Execution Log ---\n')
            result_logger.write(f'Output directory: {output_dir}\n')
            driver = webdriver.Remote('http://localhost:4723', options=options)
            driver.implicitly_wait(10)
            print('Connection successful. Starting test execution.')

            artifact_root = str(Path('baselines') / 'output') if args.baseline == 'androb2o' else 'output'
            total_steps, passed_steps, total_tokens = execute_test_case(
                driver,
                test_file,
                result_logger,
                artifact_root=artifact_root,
                use_intent_planner=args.ablation != 'no_planner' and args.baseline == 'none',
                use_control_selector=args.ablation != 'no_selector' and args.baseline == 'none',
                use_assertion_checker=args.ablation != 'no_checker',
                use_androb2o_baseline=args.baseline == 'androb2o',
                use_classifier=args.ablation != 'no_classifier',
            )

            elapsed_seconds = time.perf_counter() - start_time

            result_logger.write('\n--- Test Summary ---\n')
            result_logger.write(f'Total steps: {total_steps}\n')
            result_logger.write(f'Passed steps: {passed_steps}\n')
            result_logger.write('Result: PASS\n' if total_steps == passed_steps and total_steps > 0 else 'Result: FAIL\n')
            result_logger.write(f'Total tokens consumed: {total_tokens}\n')
            result_logger.write(f'Total runtime seconds: {elapsed_seconds:.3f}\n')
            print(f'Test results saved to {result_file}')

    except Exception as e:
        elapsed_seconds = time.perf_counter() - start_time
        try:
            with result_file.open('a', encoding='utf-8') as result_logger:
                result_logger.write('\n--- Test Summary ---\n')
                result_logger.write(f'Error: {e}\n')
                result_logger.write('Total tokens consumed: 0\n')
                result_logger.write(f'Total runtime seconds: {elapsed_seconds:.3f}\n')
        except Exception:
            pass
        print(f'An error occurred during setup: {e}')
    finally:
        if driver:
            print('Quitting driver.')
            driver.quit()
        print('Execution finished.')


if __name__ == '__main__':
    main()