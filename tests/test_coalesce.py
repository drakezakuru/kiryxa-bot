"""Tests for per-author inbound coalescing."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.coalesce import Coalescer, CoalescePolicy


class TestCoalescerAuthors(unittest.TestCase):
    def test_authors_on_same_key_do_not_mix(self) -> None:
        c = Coalescer()
        key = (-100, 7)
        c.add(key, "я про ключи", user_id=1, now=0.0)
        c.add(key, "а я про рыбалку", user_id=2, now=0.0)

        due = c.flush_due(10.0)
        self.assertEqual(len(due), 2)
        by_user = {user_id: text for _, user_id, text in due}
        self.assertEqual(by_user[1], "я про ключи")
        self.assertEqual(by_user[2], "а я про рыбалку")

    def test_quiet_timer_is_per_author(self) -> None:
        c = Coalescer(CoalescePolicy(text_quiet_sec=2.5))
        key = ("chat", None)
        c.add(key, "первый", user_id=1, now=0.0)
        c.add(key, "второй", user_id=2, now=2.0)

        # Первый уже отмолчался, второй ещё нет.
        self.assertEqual(c.flush_due(2.6), [(key, 1, "первый")])
        self.assertEqual(c.parts_count(key, 1), 0)
        self.assertEqual(c.parts_count(key, 2), 1)

        self.assertEqual(c.flush_due(4.5), [(key, 2, "второй")])

    def test_flush_only_one_author(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "мой текст", user_id=1, now=0.0)
        c.add(key, "чужой текст", user_id=2, now=0.0)

        self.assertEqual(c.flush(key, 1), "мой текст")
        self.assertIsNone(c.flush(key, 1))
        self.assertEqual(c.flush_due(10.0), [(key, 2, "чужой текст")])

    def test_user_id_defaults_to_none(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "без автора", now=0.0)
        self.assertEqual(c.parts_count(key), 1)
        self.assertEqual(c.flush(key), "без автора")


class TestCoalescerTiming(unittest.TestCase):
    def test_flush_due_returns_only_ready_batches(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "рано", user_id=1, now=0.0)

        self.assertEqual(c.flush_due(1.0), [])
        self.assertEqual(c.parts_count(key, 1), 1)

        self.assertEqual(c.flush_due(10.0), [(key, 1, "рано")])
        self.assertEqual(c.flush_due(11.0), [])

    def test_max_parts_flushes_without_quiet(self) -> None:
        c = Coalescer(
            CoalescePolicy(text_quiet_sec=2.5, max_batch_sec=100.0, max_parts=3)
        )
        key = ("chat", None)
        for i in range(3):
            c.add(key, f"часть{i}", user_id=1, now=0.0)

        self.assertEqual(c.parts_count(key, 1), 3)
        self.assertEqual(c.flush_due(0.0), [(key, 1, "часть0 часть1 часть2")])

    def test_max_batch_window(self) -> None:
        c = Coalescer(
            CoalescePolicy(text_quiet_sec=2.5, max_batch_sec=8.0, max_parts=100)
        )
        key = ("chat", None)
        for i in range(7):  # каждую секунду дописывают — тишины нет
            c.add(key, f"ч{i}", user_id=1, now=float(i))

        self.assertEqual(c.flush_due(7.9), [])
        self.assertEqual(
            c.flush_due(8.1),
            [(key, 1, "ч0 ч1 ч2 ч3 ч4 ч5 ч6")],
        )

    def test_sticker_uses_longer_quiet_window(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "подпись к стикеру", user_id=1, now=0.0, is_sticker=True)

        self.assertEqual(c.flush_due(3.0), [])
        self.assertEqual(c.flush_due(4.5), [(key, 1, "подпись к стикеру")])

    def test_max_parts_is_per_author(self) -> None:
        c = Coalescer(
            CoalescePolicy(text_quiet_sec=2.5, max_batch_sec=100.0, max_parts=3)
        )
        key = ("chat", None)
        c.add(key, "a1", user_id=1, now=0.0)
        c.add(key, "a2", user_id=1, now=0.0)
        c.add(key, "b1", user_id=2, now=0.0)

        # Ни у кого нет трёх частей — никто не готов.
        self.assertEqual(c.flush_due(0.0), [])


class TestCoalescerMerging(unittest.TestCase):
    def test_merges_parts_after_quiet_window(self) -> None:
        c = Coalescer(CoalescePolicy(text_quiet_sec=2.5))
        key = ("chat", None)
        c.add(key, "короче", user_id=1, now=0.0)
        c.add(key, "я нашёл ключ", user_id=1, now=0.5)
        c.add(key, "рабочий", user_id=1, now=1.0)

        self.assertEqual(c.flush_due(2.0), [])
        self.assertEqual(c.flush_due(4.0), [(key, 1, "короче я нашёл ключ рабочий")])
        self.assertEqual(c.flush_due(5.0), [])

    def test_dedups_repeated_parts(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "спс", user_id=1, now=0.0)
        c.add(key, "спс", user_id=1, now=1.0)

        self.assertEqual(c.flush_due(10.0), [(key, 1, "спс")])

    def test_drop_clears_pending_batch(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "забыть", user_id=1, now=0.0)

        self.assertEqual(c.parts_count(key, 1), 1)
        c.drop(key, 1)
        self.assertEqual(c.parts_count(key, 1), 0)
        self.assertEqual(c.flush_due(10.0), [])
        self.assertIsNone(c.flush(key, 1))

    def test_empty_parts_are_dropped_not_returned(self) -> None:
        c = Coalescer()
        key = ("chat", None)
        c.add(key, "   ", user_id=1, now=0.0)

        self.assertEqual(c.flush_due(10.0), [])
        self.assertEqual(c.parts_count(key, 1), 0)


if __name__ == "__main__":
    unittest.main()