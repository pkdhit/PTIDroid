from __future__ import annotations

import sys
from pathlib import Path

from appium import webdriver

from test_case_runner import execute_test_case
from test_executor import build_driver
from snapshot_utils import get_test_case_stem


DEFAULT_OUTPUT_DIR = Path('output')
DEFAULT_ARTIFACT_ROOT = Path('test')
APPIUM_SERVER_URL = 'http://localhost:4723'


def iter_test_files(output_dir: Path):
    yield from sorted(
        path for path in output_dir.glob('*.txt') if path.is_file()
    )


def collect_test_files(paths: list[str]) -> list[Path]:
    if not paths:
        return list(iter_test_files(DEFAULT_OUTPUT_DIR))

    collected: list[Path] = []
    seen: set[Path] = set()
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            candidates = iter_test_files(path)
        elif path.is_file() and path.suffix.lower() == '.txt':
            candidates = [path]
        else:
            raise FileNotFoundError(f'Input path not found or unsupported: {path}')

        for candidate in candidates:
            resolved = candidate.resolve()
            if resolved not in seen:
                seen.add(resolved)
                collected.append(candidate)

    return collected


def run_one_test_case(test_file: Path, artifact_root: Path) -> tuple[int, int]:
    options = build_driver(str(test_file))
    driver = None
    try:
        driver = webdriver.Remote(APPIUM_SERVER_URL, options=options)
        driver.implicitly_wait(10)

        test_case_stem = get_test_case_stem(str(test_file))
        case_artifact_dir = artifact_root / test_case_stem / 'output'
        case_artifact_dir.mkdir(parents=True, exist_ok=True)
        result_file = case_artifact_dir / f'{test_case_stem}.result.txt'
        with result_file.open('w', encoding='utf-8') as result_logger:
            result_logger.write('--- Test Execution Log ---\n')
            total_steps, passed_steps, total_tokens = execute_test_case(
                driver,
                str(test_file),
                result_logger,
                generate_assertions=False,
                artifact_root=str(artifact_root),
            )
            result_logger.write('\n--- Test Summary ---\n')
            result_logger.write(f'Total steps: {total_steps}\n')
            result_logger.write(f'Passed steps: {passed_steps}\n')
            result_logger.write('Result: PASS\n' if total_steps == passed_steps and total_steps > 0 else 'Result: FAIL\n')
            result_logger.write(f'Total tokens consumed: {total_tokens}\n')
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
        target = DEFAULT_OUTPUT_DIR if len(sys.argv) <= 1 else 'provided paths'
        print(f'No .txt test files found in: {target}')
        return 1

    target_desc = ', '.join(sys.argv[1:]) if len(sys.argv) > 1 else str(DEFAULT_OUTPUT_DIR)
    print(f'Found {len(test_files)} test file(s) in {target_desc}')
    print(f'Connecting to Appium server at {APPIUM_SERVER_URL}')

    artifact_root = DEFAULT_ARTIFACT_ROOT
    artifact_root.mkdir(parents=True, exist_ok=True)
    print(f'Writing runtime artifacts to: {artifact_root}')

    summary: list[tuple[Path, int, int]] = []
    for test_file in test_files:
        print(f'\n=== Running {test_file.name} ===')
        try:
            total_steps, passed_steps = run_one_test_case(test_file, artifact_root)
            summary.append((test_file, total_steps, passed_steps))
            status = 'PASS' if total_steps == passed_steps and total_steps > 0 else 'FAIL'
            print(f'Completed {test_file.name}: {passed_steps}/{total_steps} steps -> {status}')
        except Exception as exc:
            summary.append((test_file, 0, 0))
            print(f'Failed to run {test_file.name}: {exc}')

    passed_cases = sum(1 for _, total, passed in summary if total > 0 and total == passed)
    print('\n=== Summary ===')
    print(f'Passed cases: {passed_cases}/{len(summary)}')
    for test_file, total_steps, passed_steps in summary:
        status = 'PASS' if total_steps > 0 and total_steps == passed_steps else 'FAIL'
        print(f'- {test_file.name}: {passed_steps}/{total_steps} -> {status}')

    return 0 if passed_cases == len(summary) else 2


if __name__ == '__main__':
    raise SystemExit(main())