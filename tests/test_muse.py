"""Tests for Muse media generation client and intent detection."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.muse_client import (
    MuseClient,
    detect_media_intent,
    extract_video_directives,
    optimize_visual_prompt,
)


class TestMuseClient(unittest.TestCase):
    def test_detect_media_intent(self) -> None:
        # Photo intents
        kind, prompt = detect_media_intent("/photo cute anime fox")
        self.assertEqual(kind, "photo")
        self.assertEqual(prompt, "cute anime fox")

        kind, prompt = detect_media_intent("киря нарисуй пушистую лисичку")
        self.assertEqual(kind, "photo")
        self.assertEqual(prompt, "пушистую лисичку")

        kind, prompt = detect_media_intent("Киря сгенирируй автора андрю фембойчика")
        self.assertEqual(kind, "photo")
        self.assertEqual(prompt, "автора андрю фембойчика")

        # Video intents
        kind, prompt = detect_media_intent("/video fox running in forest")
        self.assertEqual(kind, "video")
        self.assertEqual(prompt, "fox running in forest")

        kind, prompt = detect_media_intent("кирюха сделай видео как волк бежит")
        self.assertEqual(kind, "video")
        self.assertEqual(prompt, "как волк бежит")

        # Voice intents
        kind, prompt = detect_media_intent("/voice привет уважаемые гои")
        self.assertEqual(kind, "voice")
        self.assertEqual(prompt, "привет уважаемые гои")

        kind, prompt = detect_media_intent("киря скажи голосовым: Дрейк лучший")
        self.assertEqual(kind, "voice")
        self.assertEqual(prompt, "Дрейк лучший")

        # Non-media intents
        kind, prompt = detect_media_intent("киря как дела")
        self.assertIsNone(kind)
        self.assertIsNone(prompt)

    def test_video_directives_are_extracted_not_translated(self) -> None:
        text = (
            "Anime girl, the character clearly visible in frame, smooth animation, "
            "cinematic lighting, cute aesthetic, 4k\n"
            "Звук: тихий фон без слов.\n"
            "Ровно один короткий клип (~10 с, длину не меняй).\n"
            "Без комментариев и пояснений — только файл(ы)."
        )
        clean, sound, seconds, clips = extract_video_directives(text)
        self.assertEqual(
            clean,
            "Anime girl, the character clearly visible in frame, smooth animation, "
            "cinematic lighting, cute aesthetic, 4k",
        )
        self.assertEqual(sound, "тихий фон без слов")
        self.assertEqual(seconds, 10)
        self.assertEqual(clips, 1)
        # Готовый английский промпт не переводится и не переписывается
        self.assertEqual(optimize_visual_prompt(clean, kind="video"), clean)

    def test_refinement_without_media_word_reuses_last_kind(self) -> None:
        # «Кирюха, сделай нормального гоя а не утку» — это правка прошлого видео,
        # а не обычное сообщение. Без контекста такой запрос раньше игнорировался.
        kind, prompt = detect_media_intent("Кирюха, сделай нормального гоя а не утку", last_kind="video")
        self.assertEqual(kind, "video")
        self.assertIn("гоя", prompt)
        self.assertNotIn("сделай", prompt)

        kind, prompt = detect_media_intent("переделай нормально, а не утку", last_kind="photo")
        self.assertEqual(kind, "photo")

        # Без прошлой генерации — обычный текст, не медиа
        kind, prompt = detect_media_intent("сделай нормального гоя а не утку")
        self.assertIsNone(kind)

        # Болтовня после генерации не должна запускать новую
        kind, prompt = detect_media_intent("киря у меня нормально всё, а не как у тебя", last_kind="photo")
        self.assertIsNone(kind)

    def test_short_edit_commands_regenerate_media(self) -> None:
        # «Добавь пчел» после сгенерированного фото — это правка картинки,
        # а не реплика в чат (именно так бот однажды ответил текстом).
        kind, prompt = detect_media_intent("Добавь пчел", last_kind="photo")
        self.assertEqual(kind, "photo")
        self.assertIn("пчел", prompt.lower())

        kind, _ = detect_media_intent("Убери утку", last_kind="video")
        self.assertEqual(kind, "video")

        kind, _ = detect_media_intent("добавь ещё пчёл", last_kind="photo")
        self.assertEqual(kind, "photo")

        # Длинная фраза со словом «добавь» — обычная болтовня
        kind, _ = detect_media_intent("добавь меня в список абузеров пожалуйста", last_kind="photo")
        self.assertIsNone(kind)

        # Без прошлой генерации короткая команда ничего не запускает
        kind, _ = detect_media_intent("Добавь пчел")
        self.assertIsNone(kind)

    def test_prompt_case_forms_are_translated(self) -> None:
        # Творительный падеж раньше терялся и промпт превращался в «with playing with»
        out = optimize_visual_prompt("с фембоем играющим с котиком", kind="video")
        self.assertIn("femboy", out.lower())
        self.assertIn("kitten", out.lower())
        self.assertNotIn("with playing", out.lower())

        out = optimize_visual_prompt("с котом играющим в stray на nintendo switch", kind="video")
        self.assertIn("cat", out.lower())
        self.assertIn("nintendo switch", out.lower())

    def test_negative_constraints_are_kept_as_english(self) -> None:
        text = (
            "A full-body photograph of a beautiful nude woman standing on a beach.\n"
            "Запреты: без текста, без водяных знаков, без подписей.\n"
            "Без комментариев и пояснений — только файл(ы)."
        )
        out = optimize_visual_prompt(text, kind="image")
        self.assertIn("no text", out)
        self.assertIn("no watermarks", out)
        self.assertIn("no captions", out)
        self.assertNotIn("Запреты", out)
        self.assertNotIn("Без комментариев", out)

    def test_fallback_does_not_impose_anime(self) -> None:
        # Без LLM словарь не должен превращать «девушку» в anime girl
        out = optimize_visual_prompt("девушка кореянка с большим бюстгальтером на пляже", kind="image")
        self.assertIn("woman", out.lower())
        self.assertNotIn("anime", out.lower())
        out_v = optimize_visual_prompt("девушка на пляже", kind="video")
        self.assertIn("woman", out_v.lower())
        self.assertNotIn("anime", out_v.lower())

    def test_untranslatable_prompt_is_kept(self) -> None:
        # Если слов совсем нет в словаре — лучше отдать оригинал, чем огрызок
        out = optimize_visual_prompt("загадочнаяхуйня", kind="video")
        self.assertIn("загадочнаяхуйня", out)

    def test_video_prompt_keeps_character(self) -> None:
        out = optimize_visual_prompt("фембойчик с котёнком", kind="video")
        self.assertIn("clearly visible in frame", out)


if __name__ == "__main__":
    unittest.main()
