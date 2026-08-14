"""Static classifier for TaskDescription.

The classifier uses a small set of semantic rules, ordered by priority:

4. Invalid / empty / wrong input cases
3. Cross-app or navigation-oriented interaction workflow
2. UI hierarchy / element presence / same-screen inspection
1. Functional single-app workflow / normal feature verification

The rules are intentionally explicit so the classifier stays stable while still
being broad enough to handle new tasks with similar wording.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class TaskCase:
    app_package: Optional[str]
    app_activity: Optional[str]
    task_description: str
    raw_text: str = ""


def normalize_text(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"\s+", " ", value)
    return value


def _compile(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    return [re.compile(pattern, re.IGNORECASE) for pattern in patterns]


def _count_matches(text: str, regexes: Iterable[re.Pattern[str]]) -> int:
    return sum(1 for regex in regexes if regex.search(text))


def _has_match(text: str, regexes: Iterable[re.Pattern[str]]) -> bool:
    return any(regex.search(text) for regex in regexes)


def _match_task_category(task_description: str) -> Optional[int]:
    """Return the category matched by task description keywords, if any."""
    text = normalize_text(task_description)

    # Category 4 always wins for wrong / empty / invalid / error cases.
    if _has_match(text, STRICT_CATEGORY_4_PATTERNS) or _has_match(text, INPUT_CONTEXT_PATTERNS) or _has_match(text, NEGATIVE_PATTERNS):
        return 4

    has_ui_presence_check = _has_match(text, UI_PRESENCE_PATTERNS)
    has_ui_check = _has_match(text, UI_PATTERNS) or _has_match(text, MULTI_CONTROL_PATTERNS) or has_ui_presence_check
    has_ui_hint = _has_match(text, APP_SPECIFIC_UI_HINTS)
    has_interaction = _has_match(text, INTERACTION_PATTERNS)
    has_navigation = _has_match(text, NAVIGATION_CONTEXT_PATTERNS)
    has_function = _has_match(text, FUNCTION_PATTERNS)
    has_access_only = _has_match(text, ACCESS_PATTERNS) and not has_function and not has_interaction and not has_navigation

    if has_interaction or has_navigation:
        return 3

    if has_ui_check or has_ui_hint or (has_access_only and has_ui_presence_check):
        return 2

    if has_function:
        return 1

    return None


NEGATIVE_PATTERNS = _compile(
    [
        r"\bincorrect\b",
        r"\bwrong\b",
        r"\bfalse\b",
        r"\binvalid\b",
        r"\bempty\b",
        r"\bmissing\b",
        r"\bnull\b",
        r"\bnone\b",
        r"\berror\b",
        r"\bfail(?:ed|ure)?\b",
        r"\bexception\b",
        r"\bnot found\b",
        r"\bnot available\b",
        r"\bcannot\b",
        r"\bcan't\b",
        r"\bno (?:more )?chances?\b",
        r"\bwrong pin\b",
        r"\bwrong password\b",
        r"\bwrong username\b",
        r"\bwrong account\b",
        r"\bnot available\b",
        r"\balready exists\b",
        r"\bdoes not exist\b",
    ]
)


STRICT_CATEGORY_4_PATTERNS = _compile(
    [
        r"\bwrong\b",
        r"\bincorrect\b",
        r"\bfalse\b",
        r"\binvalid\b",
        r"\bmistake(?:nly|s)?\b",
        r"\berror\b",
        r"\bempty\b",
    ]
)

INTERACTION_PATTERNS = _compile(
    [
        r"\bredirect(?:ed|s|ing)?\b",
        r"\bjump(?:ed|s|ing)?\b",
        r"\bnavigate(?:d|s|ing)?\b",
        r"\bgo(?:es|ing)? to\b",
        r"\bopen(?:s|ed|ing)?\s+the\s+phone\s+app\b",
        r"\bshare(?:d|s|ing)?\b",
        r"\bsend(?:s|ing|t)?\b.*\b(website|link|article|song|news|vid(?:eo)?|message)\b",
        r"\bcopy link\b",
        r"\bopen in\b",
        r"\bopen with\b",
        r"\bopen blog in\b",
        r"\bsearch and\s+addbookmark\b",
    ]
)


ACCESS_PATTERNS = _compile(
    [
        r"\bopen\b",
        r"\bclick\b",
        r"\btap\b",
        r"\benter\b",
        r"\binput\b",
    ]
)


UI_PRESENCE_PATTERNS = _compile(
    [
        r"\bcheck if there is a control\b",
        r"\bcheck if there is a control named\b",
        r"\bcheck if there is a control called\b",
        r"\bcheck whether there is a control\b",
        r"\bcheck whether there is a control named\b",
        r"\bcheck whether there is a control called\b",
        r"\bcheck if there is a field\b",
        r"\bcheck if there is an? element\b",
        r"\bcheck if there is an? item\b",
        r"\bcheck if there are controls\b",
        r"\bcheck if there are elements\b",
        r"\bcheck if there are items\b",
        r"\bcheck if there are buttons\b",
    ]
)

UI_PATTERNS = _compile(
    [
        r"\bsame screen\b",
        r"\bcurrent interface\b",
        r"\bon the current interface\b",
        r"\bui hierarchy\b",
        r"\bwidget hierarchy\b",
        r"\blayout relation\b",
        r"\bcheck if there is a control\b",
        r"\bcheck if there is a control named\b",
        r"\bcheck if there is a control called\b",
        r"\bcheck whether there is a control\b",
        r"\bcheck whether there is a control named\b",
        r"\bcheck whether there is a control called\b",
        r"\bcheck if there is a field\b",
        r"\bcheck if there is an? element\b",
        r"\bcheck if there is an? item\b",
    ]
)

MULTI_CONTROL_PATTERNS = _compile(
    [
        r"\bcheck if there are controls\b",
        r"\bcheck if there are elements\b",
        r"\bcheck if there are control\b",
        r"\bcheck if there are items\b",
        r"\bcheck if there are buttons\b",
        r"\bcontrols? named\b.*\band\b",
        r"\belements? named\b.*\band\b",
        r"\bcontrols? named\b.*\band\b.*\band\b",
        r"\belements? named\b.*\band\b.*\band\b",
    ]
)

FUNCTION_PATTERNS = _compile(
    [
        r"\bsearch\b",
        r"\bsubscribe\b",
        r"\badd\b",
        r"\bsave\b",
        r"\bshare\b",
        r"\bsend\b",
        r"\bbookmark\b",
        r"\bset\b",
        r"\bdelete\b",
        r"\bcalculate\b",
        r"\bchange dark mode\b",
        r"\bdark mode\b",
        r"\bset alarm\b",
        r"\badd task\b",
        r"\bplay song\b",
    ]
)

INPUT_CONTEXT_PATTERNS = _compile(
    [
        r"\bempty\s+(?:password|username|email|account)\b",
        r"\bincorrect\s+(?:password|username|email|pin|account)\b",
        r"\bwrong\s+(?:password|username|email|pin|account)\b",
        r"\binvalid\s+(?:password|username|email|pin|account)\b",
        r"\brepeated\s+password\b",
        r"\bsame\s+username\b",
        r"\bwrong\s+pin\b",
        r"\bempty\s+login\b",
        r"\bempty\s+password\b",
        r"\bempty\s+username\b",
    ]
)


APP_SPECIFIC_UI_HINTS = _compile(
    [
        r"\bcheckelement(?:s)?\b",
        r"\bsame screen\b",
        r"\bui hierarchy\b",
        r"\bwidget hierarchy\b",
        r"\blayout relation\b",
        r"\bcheckswitch\b",
        r"\bbattery percentage\b",
    ]
)


NAVIGATION_CONTEXT_PATTERNS = _compile(
    [
        r"\bredirect(?:ed|s|ing)?\s+to\b",
        r"\bnavigate(?:d|s|ing)?\s+to\b",
        r"\blaunch(?:ed|es|ing)?(?:\s+it)?\s+in\b",
        r"\bview in\b",
        r"\bgo to\b.*\b(?:app|application|browser|phone app|browser app|tasks app)\b",
        r"\bshare(?:d|s|ing)?\b.*\b(?:url|link|article|news|song|video|vidio|website|app)\b",
        r"\bsend(?:s|ing|t)?\b.*\b(?:website|link|article|song|news|vid(?:eo)?|message)\b",
    ]
)


def extract_task_description(file_path: str | Path) -> TaskCase:
    raw_text = Path(file_path).read_text(encoding="utf-8")
    app_package = None
    app_activity = None
    task_description = ""

    for line in raw_text.splitlines():
        if line.startswith("appPackage:"):
            app_package = line.split(":", 1)[1].strip() or None
        elif line.startswith("appActivity:"):
            app_activity = line.split(":", 1)[1].strip() or None
        elif line.startswith("TaskDescription:"):
            task_description = line.split(":", 1)[1].strip()

    return TaskCase(
        app_package=app_package,
        app_activity=app_activity,
        task_description=task_description,
        raw_text=raw_text,
    )


def classify_task_description(task_description: str) -> int:
    """Classify a TaskDescription into categories 1-4."""
    matched_category = _match_task_category(task_description)
    if matched_category is not None:
        return matched_category

    # Conservative fallback: if the task description does not clearly match any
    # of the above, treat it as a regular functional case.
    return 1


def classify_file(file_path: str | Path) -> int:
    path = Path(file_path)
    stem = path.stem.lower()

    case = extract_task_description(path)
    task_category = _match_task_category(case.task_description)
    if task_category is not None:
        return task_category

    # File-name level shortcuts are only used when the task description itself
    # does not provide a clear keyword match.
    if re.search(r"(?:wrong|empty|error|false)", stem):
        return 4
    if re.search(
        r"(?:share|send(?:website|.*(?:link|article|news|song|video))|redirect|navigate|launch|view-in|open-in|go-to|check-in)",
        stem,
    ):
        return 3
    if re.search(r"(?:checkelements?|checkelement|same-screen|samescreen|ui-hierarchy|widget-hierarchy|layout-relation)", stem):
        return 2

    return classify_task_description(case.task_description)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Classify a test input file into category 1-4.")
    parser.add_argument("file", help="Path to the input/*.txt file")
    args = parser.parse_args()

    print(classify_file(args.file))


if __name__ == "__main__":
    main()