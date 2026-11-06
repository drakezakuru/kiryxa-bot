"""Unit tests for classifier AI verdict parsing and Telegram UTF-16 offsets (no network)."""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.classifier import (
    _extract_entities_info,
    ai_classify_message,
    classify_message,
    utf16_index_to_py,
)


class _FakeAIResponse:
    """Minimal context-manager response carrying an OpenAI-style completion."""

    def __init__(self, content: str) -> None:
        self._body = json.dumps(
            {"choices": [{"message": {"content": content}}]}
        ).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeAIResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


class TestAIVerdictParsing(unittest.TestCase):
    def _ask_ai(self, answer: str) -> bool:
        response = _FakeAIResponse(answer)
        with mock.patch(
            "core.classifier.urllib.request.urlopen", return_value=response
        ) as urlopen:
            result = ai_classify_message(
                text="тестовое сообщение",
                topic_name="Ключи",
                mode="smart",
                api_url="https://api.example.test/v1/chat/completions",
                api_key="test-key",
            )
        self.assertEqual(urlopen.call_count, 1)
        return result

    def test_move_not_keep_is_move(self) -> None:
        # The first token must win even though KEEP appears later.
        self.assertFalse(self._ask_ai("MOVE, not KEEP"))

    def test_keep_is_keep(self) -> None:
        self.assertTrue(self._ask_ai("KEEP"))

    def test_move_is_move(self) -> None:
        self.assertFalse(self._ask_ai("MOVE"))

    def test_unrecognised_answer_defaults_to_keep(self) -> None:
        self.assertTrue(self._ask_ai("не понял"))

    def test_earliest_standalone_verdict_wins_after_unknown_first_token(self) -> None:
        self.assertTrue(self._ask_ai("maybe KEEP it, not MOVE"))
        self.assertFalse(self._ask_ai("maybe MOVE it, not KEEP"))


class TestUtf16IndexConversion(unittest.TestCase):
    def test_offset_after_emoji_maps_to_python_index(self) -> None:
        text = "🔥 ключ"
        self.assertEqual(utf16_index_to_py(text, 0), 0)
        # The emoji occupies two UTF-16 code units but one Python code point.
        self.assertEqual(utf16_index_to_py(text, 2), 1)
        self.assertEqual(utf16_index_to_py(text, 3), 2)
        self.assertEqual(utf16_index_to_py(text, 7), 6)
        self.assertEqual(utf16_index_to_py(text, 999), len(text))

    def test_code_entity_after_emoji_is_sliced_correctly(self) -> None:
        text = "🔥ABCDEFGHIJ"
        message = {
            "text": text,
            "entities": [{"type": "code", "offset": 2, "length": 10}],
        }
        info = _extract_entities_info(message)
        self.assertEqual(info["code_snippets"], ["ABCDEFGHIJ"])


class TestEmojiBeforeKeyClassification(unittest.TestCase):
    def test_key_after_emoji_is_still_kept(self) -> None:
        message = {"text": "🔥 ключ: sk-abcdefghijklmnopqrstuvwx"}
        result = classify_message(message, mode="smart")
        self.assertTrue(result.keep)
        self.assertEqual(result.category, "api_key_or_token")


if __name__ == "__main__":
    unittest.main()