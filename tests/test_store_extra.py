"""Reliability regression tests for core.store (P2-2).

Covers:
  * atomic compare-and-set for folded notes (no lost updates between fold workers);
  * clear RuntimeError instead of sqlite3.ProgrammingError after close();
  * chat_memory pruning that never drops not-yet-folded turns;
  * move_logs retention.
"""

from __future__ import annotations

import sqlite3
import sys
import tempfile
import threading
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.store import Store


class StoreTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "store.db"
        self.store = Store(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.tmp.cleanup()

    def memory_ids(self, chat_id: int, thread_id: int | None) -> list[int]:
        rows = self.store.memory_rows_after(chat_id, thread_id, 0, limit=10_000)
        return [int(r["id"]) for r in rows]


class TestSetNotesCas(StoreTestCase):
    def test_stale_folded_upto_does_not_overwrite_fresher_notes(self) -> None:
        chat_id, thread_id = -100500, 7
        self.assertTrue(self.store.set_notes(chat_id, thread_id, "fresh", "new", 20))

        # A worker that started before the fold above tries to commit an older
        # snapshot: it must not clobber the fresher notes.
        self.assertFalse(self.store.set_notes(chat_id, thread_id, "stale", "old", 10))

        state = self.store.get_notes(chat_id, thread_id)
        self.assertEqual(state["notes"], "fresh")
        self.assertEqual(state["style"], "new")
        self.assertEqual(state["folded_upto"], 20)

    def test_same_folded_upto_is_atomic_and_complete(self) -> None:
        """Two workers at the same folding boundary cannot interleave writes.

        Both calls use folded_upto=10; whichever commits last leaves its whole
        payload (never a mix/partial row), and a still-older boundary is then
        rejected instead of overwriting newer data.
        """
        chat_id, thread_id = -100500, 8
        self.store.set_notes(chat_id, thread_id, "base", "", 10)

        barrier = threading.Barrier(2)
        results: list[bool] = []
        lock = threading.Lock()

        def worker(payload: str) -> None:
            barrier.wait()
            ok = self.store.set_notes(chat_id, thread_id, payload, payload, 10)
            with lock:
                results.append(ok)

        threads = [
            threading.Thread(target=worker, args=("worker-a",)),
            threading.Thread(target=worker, args=("worker-b",)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)
            self.assertFalse(t.is_alive())

        state = self.store.get_notes(chat_id, thread_id)
        self.assertIn(state["notes"], {"worker-a", "worker-b"})
        self.assertEqual(state["style"], state["notes"])
        self.assertEqual(state["folded_upto"], 10)
        self.assertEqual(len(results), 2)
        self.assertTrue(all(results))

        # The loser of a later race with an older boundary must not overwrite.
        self.assertFalse(self.store.set_notes(chat_id, thread_id, "stale", "", 5))
        self.assertEqual(self.store.get_notes(chat_id, thread_id)["folded_upto"], 10)


class TestClosedStore(StoreTestCase):
    def test_public_methods_raise_runtime_error_after_close(self) -> None:
        self.store.close()

        with self.assertRaises(RuntimeError) as ctx:
            self.store.get_chat(-100500)
        self.assertIn("closed", str(ctx.exception).lower())

        with self.assertRaises(RuntimeError):
            self.store.add_memory(-100500, None, "user", "user", "hello")

        # close() is idempotent and must not raise on a second call.
        self.store.close()

    def test_closed_store_error_is_not_sqlite_programming_error(self) -> None:
        self.store.close()
        try:
            self.store.add_memory(-100500, None, "user", "user", "hello")
        except Exception as exc:  # noqa: BLE001 - the type is the assertion
            self.assertIsInstance(exc, RuntimeError)
            self.assertNotIsInstance(exc, sqlite3.ProgrammingError)
        else:
            self.fail("add_memory on a closed store did not raise")


class TestPruneMemory(StoreTestCase):
    def test_keeps_last_200_until_first_fold(self) -> None:
        chat_id = -100500
        for i in range(250):
            self.store.add_memory(chat_id, 42, "user", "user", f"m{i}")

        ids = self.memory_ids(chat_id, 42)
        self.assertEqual(len(ids), 200)
        self.assertEqual(ids, list(range(ids[0], ids[0] + 200)))

    def test_prune_removes_only_folded_rows(self) -> None:
        chat_id, thread_id = -100500, 42
        for i in range(250):
            self.store.add_memory(chat_id, thread_id, "user", "user", f"m{i}")

        ids = self.memory_ids(chat_id, thread_id)
        folded_upto = ids[19]
        self.store.set_notes(chat_id, thread_id, "summary", "", folded_upto)

        deleted = self.store.prune(chat_id, thread_id)
        expected_deleted = len([i for i in ids if i <= folded_upto])
        self.assertEqual(deleted, expected_deleted)

        remaining = self.memory_ids(chat_id, thread_id)
        self.assertTrue(remaining)
        self.assertTrue(all(i > folded_upto for i in remaining))
        self.assertEqual(remaining, [i for i in ids if i > folded_upto])

    def test_unfolded_messages_survive_add_memory_and_prune(self) -> None:
        chat_id, thread_id = -100500, 42
        for i in range(60):
            self.store.add_memory(chat_id, thread_id, "user", "user", f"m{i}")

        ids = self.memory_ids(chat_id, thread_id)
        folded_upto = ids[9]
        self.store.set_notes(chat_id, thread_id, "summary", "", folded_upto)

        # add_memory runs retention itself; a fresh turn is above folded_upto
        # and must never be deleted.
        self.store.add_memory(chat_id, thread_id, "user", "user", "not folded yet")
        contents = [
            r["content"]
            for r in self.store.memory_rows_after(chat_id, thread_id, folded_upto, 10_000)
        ]
        self.assertIn("not folded yet", contents)

        remaining = self.memory_ids(chat_id, thread_id)
        self.assertTrue(all(i > folded_upto for i in remaining))

    def test_prune_is_scoped_to_one_thread(self) -> None:
        chat_id = -100500
        for i in range(20):
            self.store.add_memory(chat_id, 1, "user", "user", f"a{i}")
            self.store.add_memory(chat_id, 2, "user", "user", f"b{i}")

        thread1_ids = self.memory_ids(chat_id, 1)
        thread2_ids = self.memory_ids(chat_id, 2)
        self.store.set_notes(chat_id, 1, "summary", "", thread1_ids[4])
        self.store.prune(chat_id, 1)

        self.assertTrue(self.memory_ids(chat_id, 1))
        self.assertEqual(self.memory_ids(chat_id, 2), thread2_ids)
        self.assertEqual(len(self.memory_ids(chat_id, 2)), 20)


class TestMoveLogsRetention(StoreTestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.db_path = Path(self.tmp.name) / "store.db"
        self.store = Store(
            self.db_path,
            move_logs_limit=50,
            move_logs_cleanup_every=10,
        )

    def _record(self, chat_id: int, n: int) -> None:
        for i in range(n):
            self.store.record_moved(
                chat_id, 10, 20, 1000 + i, "user", "reason", f"snippet {i}"
            )

    def _count(self, chat_id: int) -> int:
        row = self.store._conn.execute(
            "SELECT COUNT(*) AS n FROM move_logs WHERE chat_id = ?", (chat_id,)
        ).fetchone()
        return int(row["n"])

    def test_retention_keeps_latest_rows_per_chat(self) -> None:
        chat_id = -100500
        self._record(chat_id, 120)
        self.assertEqual(self._count(chat_id), 50)

        kept = self.store._conn.execute(
            "SELECT id FROM move_logs WHERE chat_id = ? ORDER BY id ASC", (chat_id,)
        ).fetchall()
        kept_ids = [int(r["id"]) for r in kept]
        self.assertEqual(kept_ids, list(range(kept_ids[0], kept_ids[0] + 50)))

    def test_retention_is_per_chat(self) -> None:
        self._record(-100500, 60)
        self._record(-100501, 15)
        self.assertEqual(self._count(-100500), 50)
        self.assertEqual(self._count(-100501), 15)

    def test_default_limit_is_2000(self) -> None:
        self.assertEqual(Store._MOVE_LOGS_LIMIT, 2000)
        other = Store(self.db_path)
        try:
            self.assertEqual(other.move_logs_limit, 2000)
        finally:
            other.close()

    def test_default_retention_after_2100_inserts(self) -> None:
        db = Path(self.tmp.name) / "default.db"
        store = Store(db)
        try:
            for i in range(2100):
                store.record_moved(-100500, 10, 20, 1000 + i, "user", "reason", "s")
            row = store._conn.execute(
                "SELECT COUNT(*) AS n FROM move_logs WHERE chat_id = ?", (-100500,)
            ).fetchone()
            self.assertLessEqual(int(row["n"]), 2000)
        finally:
            store.close()


if __name__ == "__main__":
    unittest.main()