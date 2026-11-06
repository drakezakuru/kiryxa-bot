"""Tests for emotion-indexed stickers and GIF triggers."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.media_bank import MediaBank, MOOD_EMOJI


class TestStickerByEmotion(unittest.TestCase):
    def setUp(self) -> None:
        self.mb = MediaBank()

    def test_mood_resolves_to_emoji(self) -> None:
        self.assertEqual(MOOD_EMOJI["laugh"], "😂")
        self.assertEqual(MOOD_EMOJI["angry"], "😠")

    def test_pack_sticker_wins_for_mood(self) -> None:
        self.mb.load_pack([
            {"file_id": "PACK_ANGRY", "emoji": "😠"},
            {"file_id": "PACK_LAUGH", "emoji": "😂"},
        ])
        self.assertEqual(self.mb.get_sticker("angry"), "PACK_ANGRY")
        self.assertEqual(self.mb.get_sticker("😂"), "PACK_LAUGH")

    def test_emoji_tag_in_reply(self) -> None:
        self.mb.load_pack([{"file_id": "PACK_SAD", "emoji": "😢"}])
        text, stk, anim = self.mb.parse_reply("грустно [sticker=😢]")
        self.assertEqual(text, "грустно")
        # Пак дополняет встроенный стикер для 😢 — подходит любой из этой группы
        self.assertIn(stk, {"PACK_SAD", "CAADAgADlUAAAvR_kUj8XVezdrxBRgI"})
        self.assertIsNone(anim)

    def test_falls_back_to_builtin(self) -> None:
        self.assertIsNotNone(self.mb.get_sticker("think"))

    def test_sticker_cooldown_per_chat(self) -> None:
        self.assertIsNotNone(self.mb.get_sticker("think", chat_key="c1"))
        self.assertIsNone(self.mb.get_sticker("think", chat_key="c1"))
        self.assertIsNotNone(self.mb.get_sticker("think", chat_key="c2"))


class TestGifTrigger(unittest.TestCase):
    def setUp(self) -> None:
        self.mb = MediaBank()

    def test_laughter_triggers_gif(self) -> None:
        self.assertIsNotNone(self.mb.wants_reaction_gif("ахахахах ну ты выдал", chat_key="g1"))

    def test_second_laugh_is_rate_limited(self) -> None:
        self.mb.wants_reaction_gif("ору в голос", chat_key="g2")
        self.assertIsNone(self.mb.wants_reaction_gif("ору опять", chat_key="g2"))

    def test_normal_text_does_not_trigger(self) -> None:
        self.assertIsNone(self.mb.wants_reaction_gif("киря как дела", chat_key="g3"))


if __name__ == "__main__":
    unittest.main()
