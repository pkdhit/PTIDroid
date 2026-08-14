from __future__ import annotations

import unittest
from pathlib import Path

from classifier import classify_file


INPUT_DIR = Path("input")


EXPECTED_CATEGORIES: dict[str, int] = {
    "booking-checkelement": 2,
    "booking-searchhotel": 1,
    "brave-checkelement": 2,
    "brave-search-and-addbookmark": 1,
    "calculator-bmi": 1,
    "calculator-error": 4,
    "calendar-addtask": 1,
    "calendar-check-in-googletask": 3,
    "clock-checkelement": 2,
    "clock-setalarm": 1,
    "cnn-checkelements-samescreen": 2,
    "cnn-search-and-set": 1,
    "element-change-darkmode": 1,
    "element-setpin-and-checkelement": 2,
    "element-wrongpin": 4,
    "firefox-change-darkmode": 1,
    "firefox-navigate-to-telephone": 3,
    "firefox-navigate-to-telephone-false": 4,
    "foxnews-checkelements-samescreen": 2,
    "foxnews-sharenews": 3,
    "geek-emptypassword": 4,
    "geek-wrongaccount": 4,
    "googleplay-save-to-wishlist": 1,
    "googleplay-shareapp": 3,
    "googletask-addlist": 1,
    "googletask-addtask": 1,
    "k9mail-checkelements-samescreen": 2,
    "k9mail-deletemail": 1,
    "mail-checkelements-samescreen": 2,
    "mail-sendmail": 1,
    "minimal-addtask": 1,
    "minimal-addtask-and-removetask": 1,
    "settings-battery-percentage": 1,
    "settings-set-airplanemode": 1,
    "signal-checkelement": 2,
    "signal-searchfriend": 1,
    "signal-sendwebsite": 3,
    "spotify-checkelements-samescreen": 2,
    "spotify-playsong": 1,
    "spotify-sharesong": 3,
    "tipcalculator-calculate": 1,
    "tipcalculator-wrongperson": 4,
    "trello-addtask-and-addwidget": 1,
    "trello-checkelements-samescreen": 2,
    "tumblr-checkswitch": 1,
    "tumblr-openblog-in-firefox": 3,
    "tumblr-postblog": 1,
    "wikipedia-sharearticle": 3,
    "wikipedia-wrong-repeatedpassword": 4,
    "wikipedia-wrong-sameusername": 4,
    "yahnac-empty-login": 4,
    "yahnac-emptypassword-login": 4,
    "yahnac-emptyusername-login": 4,
    "yelp-save-restaurant": 1,
    "yelp-share-restaurant": 3,
    "youtube-checkelement": 2,
    "youtube-sharevidio": 3,
    "youtube-subscribe": 1,
}


class ClassifierInputFilesTest(unittest.TestCase):
    def test_all_input_files_match_expected_category(self) -> None:
        actual_files = sorted(INPUT_DIR.glob("*.txt"))
        actual_stems = {path.stem for path in actual_files}
        expected_stems = set(EXPECTED_CATEGORIES)

        self.assertEqual(
            actual_stems,
            expected_stems,
            msg=(
                f"Input file set mismatch. Missing={sorted(expected_stems - actual_stems)}, "
                f"Extra={sorted(actual_stems - expected_stems)}"
            ),
        )

        mismatches: list[str] = []
        for path in actual_files:
            expected = EXPECTED_CATEGORIES[path.stem]
            actual = classify_file(path)
            if actual != expected:
                mismatches.append(
                    f"{path.name}: expected {expected}, got {actual}"
                )

        self.assertEqual(mismatches, [], msg="\n".join(mismatches))


if __name__ == "__main__":
    unittest.main(verbosity=2)