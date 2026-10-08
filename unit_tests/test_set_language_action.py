"""Testovi za action_set_language (flow change_language).

Pokretanje (iz korena Rasa projekta):  python -m unittest discover -s unit_tests -t .
"""

import unittest
from unittest import mock

from rasa_sdk.events import SlotSet

from actions.actions import ActionSetLanguage


def run_action(requested_language):
    tracker = mock.Mock()
    tracker.get_slot.return_value = requested_language
    return ActionSetLanguage().run(mock.Mock(), tracker, {})


class SetLanguageTests(unittest.TestCase):
    def test_sets_language_code_and_clears_request(self) -> None:
        for requested, code in [
            ("serbian_cyrillic", "sr-Cyrl"),
            ("serbian_latin", "sr-Latn"),
            ("english", "en"),
        ]:
            with self.subTest(requested=requested):
                self.assertEqual(
                    run_action(requested),
                    [SlotSet("language", code), SlotSet("requested_language", None)],
                )

    def test_unknown_value_leaves_language_unchanged(self) -> None:
        self.assertEqual(run_action(None), [SlotSet("requested_language", None)])
        self.assertEqual(run_action("german"), [SlotSet("requested_language", None)])


if __name__ == "__main__":
    unittest.main()
