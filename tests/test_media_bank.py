"""Tests for media_bank module and sticker/gif sending."""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.media_bank import MediaBank, normalize_emoji


class TestMediaBank(unittest.TestCase):
    def setUp(self) -> None:
        self.mb = MediaBank()

    def test_parse_sticker_tags(self) -> None:
        text, stk, anim = self.mb.parse_reply("Держи [sticker:think] и думай")
        self.assertEqual(text, "Держи  и думай")
        self.assertIsNotNone(stk)
        self.assertIsNone(anim)

    def test_parse_gif_tags(self) -> None:
        text, stk, anim = self.mb.parse_reply("Хаха [gif]")
        self.assertEqual(text, "Хаха")
        self.assertIsNone(stk)
        self.assertIsNotNone(anim)

    def test_user_ask_detection(self) -> None:
        stk, gif = self.mb.check_user_ask("кирюха скинь стикер")
        self.assertTrue(stk)
        self.assertFalse(gif)

        stk, gif = self.mb.check_user_ask("киря дай гифку угарную")
        self.assertFalse(stk)
        self.assertTrue(gif)

    def test_learn_sticker_and_animation(self) -> None:
        self.mb.learn_sticker("FID_TEST_STICKER", emoji="😎")
        self.assertEqual(self.mb.stickers.get("😎"), "FID_TEST_STICKER")

        # Chat-learned GIFs live apart from the curated library so they survive reloads
        self.mb.learn_animation("FID_TEST_ANIMATION")
        self.assertIn("FID_TEST_ANIMATION", self.mb.learned_animations)
        self.assertNotIn("FID_TEST_ANIMATION", self.mb.animations)


class TestMediaBankPersistence(unittest.TestCase):
    def test_learn_sticker_is_debounced_until_flush(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "media_bank.json"
            mb = MediaBank(storage_path=path)
            self.assertFalse(path.exists())

            mb.learn_sticker("FID_DEBOUNCED", emoji="😎")
            # Learning only marks the bank dirty; the daemon timer writes later.
            self.assertFalse(path.exists())

            mb.flush()
            self.assertTrue(path.exists())
            data = json.loads(path.read_text(encoding="utf-8"))
            self.assertIn("FID_DEBOUNCED", data["learned_stickers"])


class TestMediaBankLimits(unittest.TestCase):
    def test_bucket_limit_evicts_oldest(self) -> None:
        mb = MediaBank()
        emoji = "😂"
        for i in range(305):
            mb.learn_sticker(f"FID_{i}", emoji=emoji)

        key = normalize_emoji(emoji)
        self.assertEqual(len(mb.learned_by_emoji[key]), 300)
        self.assertEqual(len(mb.by_emoji[key]), 300)
        self.assertNotIn("FID_0", mb.learned_by_emoji[key])
        self.assertNotIn("FID_4", mb.learned_by_emoji[key])
        self.assertIn("FID_5", mb.learned_by_emoji[key])
        self.assertIn("FID_304", mb.learned_by_emoji[key])
        self.assertIn("FID_304", mb.by_emoji[key])
        self.assertLessEqual(len(mb.learned_stickers), 2000)


class TestMediaBankMoodSelection(unittest.TestCase):
    def test_explicit_laugh_mood_ignores_other_moods(self) -> None:
        mb = MediaBank()
        mb.learn_sticker("FID_LAUGH", emoji="😂")
        mb.learn_sticker("FID_ANGRY", emoji="😠")

        # Old behaviour first rolled a random learned sticker and could return
        # the angry one for mood="laugh". With deterministic 0.0 it always did.
        with mock.patch("core.media_bank.random.random", return_value=0.0), \
                mock.patch("core.media_bank.random.choice", side_effect=lambda seq: seq[-1]):
            self.assertEqual(mb.get_sticker("laugh"), "FID_LAUGH")


class TestMaybeReactionChance(unittest.TestCase):
    def test_zero_chance_returns_no_media(self) -> None:
        mb = MediaBank()
        with mock.patch("core.media_bank.random.random", return_value=0.0):
            sticker, animation = mb.maybe_reaction(
                "ахахахах ну ты выдал", "ору", chance=0.0
            )
        self.assertIsNone(sticker)
        self.assertIsNone(animation)

    def test_full_chance_still_returns_laugh_gif(self) -> None:
        mb = MediaBank()
        with mock.patch("core.media_bank.random.random", return_value=0.0):
            sticker, animation = mb.maybe_reaction(
                "ахахахах ну ты выдал", "ору", chance=1.0
            )
        self.assertIsNone(sticker)
        self.assertIsNotNone(animation)


if __name__ == "__main__":
    unittest.main()
