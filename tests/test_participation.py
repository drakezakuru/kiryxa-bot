"""Tests for participation scoring and inbound coalescing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.coalesce import Coalescer, CoalescePolicy
from core.participation import SCORE_THRESHOLD, score_message, should_join


class TestParticipation(unittest.TestCase):
    def test_addressed_always_replies(self) -> None:
        verdict = should_join("ну и бред", addressed=True)
        self.assertTrue(verdict.reply)
        self.assertEqual(verdict.reason, "addressed")

    def test_closing_lines_are_ignored(self) -> None:
        for text in ("ок", "спс", "ага", "да", "понял"):
            self.assertLess(score_message(text), SCORE_THRESHOLD, text)

    def test_open_question_scores_high(self) -> None:
        text = "гои, кто шарит за клод, у кого ключ живой?"
        self.assertGreaterEqual(score_message(text), SCORE_THRESHOLD)
        self.assertTrue(should_join(text).reply)

    def test_reply_to_other_person_is_left_alone(self) -> None:
        text = "ну и что дальше"
        with_other = score_message(text, replies_to_other=True)
        without = score_message(text)
        self.assertLess(with_other, without)

    def test_private_thread_left_alone(self) -> None:
        verdict = should_join(
            "ну да, прикольно",
            from_user=1,
            last_speaker=2,
            recent_speakers=[1, 2, 1, 2, 1],
        )
        self.assertFalse(verdict.reply)

    def test_busy_chat_stays_quiet(self) -> None:
        times = [100.0 + i * 0.5 for i in range(30)]  # 30 сообщений за 15 с
        verdict = should_join(
            "гои кто шарит?",
            recent_messages=times,
            now=115.0,
        )
        self.assertFalse(verdict.reply)
        self.assertEqual(verdict.reason, "chat is busy")


class TestCoalescing(unittest.TestCase):
    def test_merges_parts_after_quiet_window(self) -> None:
        c = Coalescer(CoalescePolicy(text_quiet_sec=2.5))
        c.add("k", "короче", 1, now=0.0)
        c.add("k", "я нашёл ключ", 1, now=0.5)
        c.add("k", "рабочий", 1, now=1.0)

        self.assertEqual(c.flush_due(2.0), [])
        due = c.flush_due(4.0)
        self.assertEqual(due, [("k", 1, "короче я нашёл ключ рабочий")])
        self.assertEqual(c.flush_due(5.0), [])

    def test_force_flush(self) -> None:
        c = Coalescer()
        c.add("k", "один", 1, now=0.0)
        self.assertEqual(c.flush("k", 1), "один")

    def test_max_batch_window(self) -> None:
        c = Coalescer(CoalescePolicy(text_quiet_sec=2.5, max_batch_sec=8.0))
        c.add("k", "долгая мысль", 1, now=0.0)
        # Каждую секунду дописывают — тишины нет, но лимит батча срабатывает
        for i in range(1, 9):
            c.add("k", f"часть{i}", 1, now=float(i))
        self.assertTrue(c.flush_due(8.1))

    def test_dedups_repeated_parts(self) -> None:
        c = Coalescer()
        c.add("k", "спс", 1, now=0.0)
        c.add("k", "спс", 1, now=1.0)
        self.assertEqual(c.flush("k", 1), "спс")

    def test_two_authors_do_not_mix(self) -> None:
        c = Coalescer()
        c.add("k", "я про ключи", 1, now=0.0)
        c.add("k", "а я про игры", 2, now=0.1)
        due = c.flush_due(10.0)
        self.assertIn(("k", 1, "я про ключи"), due)
        self.assertIn(("k", 2, "а я про игры"), due)


if __name__ == "__main__":
    unittest.main()
