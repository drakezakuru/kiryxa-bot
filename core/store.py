"""SQLite storage for per-chat settings, forum topic rules, whitelists, and moved message stats."""

from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


class Store:
    """Thread-safe SQLite store for TopicGuard Bot."""

    _MOVE_LOGS_LIMIT = 2000
    _MOVE_LOGS_CLEANUP_EVERY = 50
    _MEMORY_KEEP_UNFOLDED = 200

    def __init__(
        self,
        db_path: str | Path,
        default_auto_guard: bool = True,
        default_admin_bypass: bool = False,
        default_notify_sec: int = 0,
        move_logs_limit: int = _MOVE_LOGS_LIMIT,
        move_logs_cleanup_every: int = _MOVE_LOGS_CLEANUP_EVERY,
    ) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.default_auto_guard = 1 if default_auto_guard else 0
        self.default_admin_bypass = 1 if default_admin_bypass else 0
        self.default_notify_sec = int(default_notify_sec)
        self.move_logs_limit = max(1, int(move_logs_limit))
        self.move_logs_cleanup_every = max(1, int(move_logs_cleanup_every))
        self._move_inserts: Dict[int, int] = {}
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    def _ensure_open(self) -> None:
        """Fail fast when the store was closed while a worker was still running."""
        if self._closed:
            raise RuntimeError("store is closed")

    @contextmanager
    def _locked(self, write: bool = False) -> Iterator[None]:
        """Hold the store lock, check liveness, and optionally open a transaction.

        The open check happens while the lock is held, so ``close()`` (which
        takes the same lock) can never close the connection between the check
        and the SQL statements that follow.
        """
        with self._lock:
            self._ensure_open()
            if write:
                with self._conn:
                    yield
            else:
                yield

    def _init_schema(self) -> None:
        with self._locked(write=True):
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    chat_id INTEGER PRIMARY KEY,
                    title TEXT NOT NULL DEFAULT '',
                    general_thread_id INTEGER DEFAULT NULL,
                    auto_guard INTEGER NOT NULL DEFAULT 1,
                    admin_bypass INTEGER NOT NULL DEFAULT 0,
                    notify_sec INTEGER NOT NULL DEFAULT 0,
                    moved_count INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS topics (
                    chat_id INTEGER NOT NULL,
                    thread_id INTEGER NOT NULL,
                    topic_name TEXT NOT NULL DEFAULT '',
                    mode TEXT NOT NULL DEFAULT 'smart',
                    is_manual INTEGER NOT NULL DEFAULT 0,
                    moved_count INTEGER NOT NULL DEFAULT 0,
                    kept_count INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, thread_id)
                );

                CREATE TABLE IF NOT EXISTS whitelist (
                    chat_id INTEGER NOT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL DEFAULT '',
                    added_at INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, user_id)
                );

                CREATE TABLE IF NOT EXISTS move_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    from_thread_id INTEGER NOT NULL,
                    to_thread_id INTEGER DEFAULT NULL,
                    user_id INTEGER NOT NULL,
                    username TEXT NOT NULL DEFAULT '',
                    reason TEXT NOT NULL DEFAULT '',
                    snippet TEXT NOT NULL DEFAULT '',
                    created_at INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_move_logs_chat ON move_logs(chat_id, id);

                CREATE TABLE IF NOT EXISTS chat_memory (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id INTEGER NOT NULL,
                    thread_id INTEGER,
                    role TEXT NOT NULL,
                    author_name TEXT NOT NULL DEFAULT '',
                    content TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT 0
                );
                CREATE INDEX IF NOT EXISTS idx_chat_mem ON chat_memory(chat_id, thread_id, id);
                CREATE TABLE IF NOT EXISTS chat_notes (
                    chat_id INTEGER NOT NULL,
                    thread_id INTEGER,
                    notes TEXT NOT NULL DEFAULT '',
                    style TEXT NOT NULL DEFAULT '',
                    folded_upto INTEGER NOT NULL DEFAULT 0,
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, thread_id)
                );
                CREATE TABLE IF NOT EXISTS person_notes (
                    chat_id INTEGER NOT NULL,
                    name_key TEXT NOT NULL,
                    display_name TEXT NOT NULL DEFAULT '',
                    note TEXT NOT NULL DEFAULT '',
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (chat_id, name_key)
                );
                CREATE TABLE IF NOT EXISTS settings (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL DEFAULT '',
                    updated_at INTEGER NOT NULL DEFAULT 0
                );
                """
            )

    def ensure_chat(self, chat_id: int, title: str = "") -> Dict[str, Any]:
        now = int(time.time())
        with self._locked(write=True):
            row = self._conn.execute(
                "SELECT * FROM chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            if row is None:
                self._conn.execute(
                    """
                    INSERT INTO chats (chat_id, title, general_thread_id, auto_guard, admin_bypass, notify_sec, moved_count, updated_at)
                    VALUES (?, ?, NULL, ?, ?, ?, 0, ?)
                    """,
                    (
                        chat_id,
                        title or "",
                        self.default_auto_guard,
                        self.default_admin_bypass,
                        self.default_notify_sec,
                        now,
                    ),
                )
            elif title and title != row["title"]:
                self._conn.execute(
                    "UPDATE chats SET title = ?, updated_at = ? WHERE chat_id = ?",
                    (title, now, chat_id),
                )
            row = self._conn.execute(
                "SELECT * FROM chats WHERE chat_id = ?", (chat_id,)
            ).fetchone()
            return dict(row)

    def get_chat(self, chat_id: int) -> Dict[str, Any]:
        return self.ensure_chat(chat_id)

    def list_chats(self) -> List[Dict[str, Any]]:
        with self._locked():
            rows = self._conn.execute(
                "SELECT * FROM chats ORDER BY updated_at DESC"
            ).fetchall()
            return [dict(r) for r in rows]

    # --- generic key/value settings (голос, режимы и прочие переключатели) ---
    def get_setting(self, key: str, default: str = "") -> str:
        with self._locked():
            row = self._conn.execute(
                "SELECT value FROM settings WHERE key = ?", (key,)
            ).fetchone()
            return str(row["value"]) if row is not None else default

    def set_setting(self, key: str, value: str) -> None:
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                """
                INSERT INTO settings (key, value, updated_at) VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET value = excluded.value, updated_at = excluded.updated_at
                """,
                (key, str(value), now),
            )

    def set_general_thread(self, chat_id: int, thread_id: Optional[int]) -> None:
        self.ensure_chat(chat_id)
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                "UPDATE chats SET general_thread_id = ?, updated_at = ? WHERE chat_id = ?",
                (thread_id, now, chat_id),
            )

    def set_auto_guard(self, chat_id: int, enabled: bool) -> None:
        self.ensure_chat(chat_id)
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                "UPDATE chats SET auto_guard = ?, updated_at = ? WHERE chat_id = ?",
                (1 if enabled else 0, now, chat_id),
            )

    def set_admin_bypass(self, chat_id: int, enabled: bool) -> None:
        self.ensure_chat(chat_id)
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                "UPDATE chats SET admin_bypass = ?, updated_at = ? WHERE chat_id = ?",
                (1 if enabled else 0, now, chat_id),
            )

    def set_notify_sec(self, chat_id: int, seconds: int) -> None:
        self.ensure_chat(chat_id)
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                "UPDATE chats SET notify_sec = ?, updated_at = ? WHERE chat_id = ?",
                (max(0, int(seconds)), now, chat_id),
            )

    def get_topic(self, chat_id: int, thread_id: int) -> Optional[Dict[str, Any]]:
        with self._locked():
            row = self._conn.execute(
                "SELECT * FROM topics WHERE chat_id = ? AND thread_id = ?",
                (chat_id, thread_id),
            ).fetchone()
            return dict(row) if row else None

    def upsert_topic(
        self,
        chat_id: int,
        thread_id: int,
        topic_name: Optional[str] = None,
        mode: Optional[str] = None,
        is_manual: Optional[bool] = None,
    ) -> Dict[str, Any]:
        now = int(time.time())
        with self._locked(write=True):
            existing = self.get_topic(chat_id, thread_id)
            if existing is None:
                final_name = topic_name if topic_name is not None else f"Ветка #{thread_id}"
                final_mode = mode if mode is not None else "smart"
                final_manual = 1 if is_manual else 0
                self._conn.execute(
                    """
                    INSERT INTO topics (chat_id, thread_id, topic_name, mode, is_manual, moved_count, kept_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, 0, 0, ?)
                    """,
                    (chat_id, thread_id, final_name, final_mode, final_manual, now),
                )
            else:
                final_name = (
                    topic_name
                    if (topic_name is not None and topic_name.strip())
                    else existing["topic_name"]
                )
                # Only overwrite mode if explicitly manual OR existing wasn't manually locked
                if mode is not None and (is_manual or not existing["is_manual"]):
                    final_mode = mode
                else:
                    final_mode = existing["mode"]
                final_manual = (
                    (1 if is_manual else 0)
                    if is_manual is not None
                    else existing["is_manual"]
                )
                self._conn.execute(
                    """
                    UPDATE topics
                    SET topic_name = ?, mode = ?, is_manual = ?, updated_at = ?
                    WHERE chat_id = ? AND thread_id = ?
                    """,
                    (final_name, final_mode, final_manual, now, chat_id, thread_id),
                )
            row = self._conn.execute(
                "SELECT * FROM topics WHERE chat_id = ? AND thread_id = ?",
                (chat_id, thread_id),
            ).fetchone()
            return dict(row)

    def list_topics(self, chat_id: int) -> List[Dict[str, Any]]:
        with self._locked():
            rows = self._conn.execute(
                "SELECT * FROM topics WHERE chat_id = ? ORDER BY thread_id ASC",
                (chat_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def record_kept(self, chat_id: int, thread_id: int) -> None:
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                """
                UPDATE topics
                SET kept_count = kept_count + 1, updated_at = ?
                WHERE chat_id = ? AND thread_id = ?
                """,
                (now, chat_id, thread_id),
            )

    def record_moved(
        self,
        chat_id: int,
        from_thread_id: int,
        to_thread_id: Optional[int],
        user_id: int,
        username: str,
        reason: str,
        snippet: str,
    ) -> None:
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                "UPDATE chats SET moved_count = moved_count + 1, updated_at = ? WHERE chat_id = ?",
                (now, chat_id),
            )
            self._conn.execute(
                """
                UPDATE topics
                SET moved_count = moved_count + 1, updated_at = ?
                WHERE chat_id = ? AND thread_id = ?
                """,
                (now, chat_id, from_thread_id),
            )
            self._conn.execute(
                """
                INSERT INTO move_logs (chat_id, from_thread_id, to_thread_id, user_id, username, reason, snippet, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    from_thread_id,
                    to_thread_id,
                    user_id,
                    username or "",
                    reason,
                    (snippet or "")[:300],
                    now,
                ),
            )
            # Keep move_logs bounded per chat.  Doing this on every call would
            # be wasteful, so run it once per `move_logs_cleanup_every` inserts
            # into this particular chat.
            inserts = self._move_inserts.get(chat_id, 0) + 1
            self._move_inserts[chat_id] = inserts
            if inserts % self.move_logs_cleanup_every == 0:
                self._conn.execute(
                    """
                    DELETE FROM move_logs
                    WHERE chat_id = ?
                      AND id NOT IN (
                          SELECT id FROM move_logs
                          WHERE chat_id = ?
                          ORDER BY id DESC
                          LIMIT ?
                      )
                    """,
                    (chat_id, chat_id, self.move_logs_limit),
                )

    def add_whitelist(self, chat_id: int, user_id: int, username: str = "") -> None:
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                """
                INSERT INTO whitelist (chat_id, user_id, username, added_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(chat_id, user_id) DO UPDATE SET username = excluded.username
                """,
                (chat_id, user_id, username or "", now),
            )

    def remove_whitelist(self, chat_id: int, user_id: int) -> bool:
        with self._locked(write=True):
            cur = self._conn.execute(
                "DELETE FROM whitelist WHERE chat_id = ? AND user_id = ?",
                (chat_id, user_id),
            )
            return cur.rowcount > 0

    def is_whitelisted(self, chat_id: int, user_id: int) -> bool:
        with self._locked():
            row = self._conn.execute(
                "SELECT 1 FROM whitelist WHERE chat_id = ? AND user_id = ?",
                (chat_id, user_id),
            ).fetchone()
            return row is not None

    def list_whitelist(self, chat_id: int) -> List[Dict[str, Any]]:
        with self._locked():
            rows = self._conn.execute(
                "SELECT * FROM whitelist WHERE chat_id = ? ORDER BY added_at DESC",
                (chat_id,),
            ).fetchall()
            return [dict(r) for r in rows]

    def add_memory(
        self,
        chat_id: int,
        thread_id: Optional[int],
        role: str,
        author_name: str,
        content: str,
    ) -> None:
        now = int(time.time())
        with self._locked(write=True):
            self._conn.execute(
                """
                INSERT INTO chat_memory (chat_id, thread_id, role, author_name, content, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, thread_id, role, author_name or "", content[:4000], now),
            )
            self._prune_memory(chat_id, thread_id)

    def _prune_memory(self, chat_id: int, thread_id: Optional[int]) -> int:
        """Retention for chat_memory; caller must already hold the lock.

        Only turns already folded into ``chat_notes`` (``id <= folded_upto``)
        may be dropped.  Until the first fold, keep the last
        ``_MEMORY_KEEP_UNFOLDED`` turns instead of the old hard limit of 50.
        """
        tid = self._note_thread(thread_id)
        row = self._conn.execute(
            "SELECT folded_upto FROM chat_notes WHERE chat_id = ? AND thread_id = ?",
            (chat_id, tid),
        ).fetchone()
        folded_upto = int(row["folded_upto"] or 0) if row is not None else 0
        if folded_upto > 0:
            cur = self._conn.execute(
                """
                DELETE FROM chat_memory
                WHERE chat_id = ?
                  AND (thread_id = ? OR (thread_id IS NULL AND ? IS NULL))
                  AND id <= ?
                """,
                (chat_id, thread_id, thread_id, folded_upto),
            )
        else:
            cur = self._conn.execute(
                """
                DELETE FROM chat_memory
                WHERE chat_id = ?
                  AND (thread_id = ? OR (thread_id IS NULL AND ? IS NULL))
                  AND id NOT IN (
                      SELECT id FROM chat_memory
                      WHERE chat_id = ?
                        AND (thread_id = ? OR (thread_id IS NULL AND ? IS NULL))
                      ORDER BY id DESC LIMIT ?
                  )
                """,
                (
                    chat_id,
                    thread_id,
                    thread_id,
                    chat_id,
                    thread_id,
                    thread_id,
                    self._MEMORY_KEEP_UNFOLDED,
                ),
            )
        return max(0, int(cur.rowcount))

    def prune(self, chat_id: int, thread_id: Optional[int]) -> int:
        """Drop memory rows that are already covered by the notes summary."""
        with self._locked(write=True):
            return self._prune_memory(chat_id, thread_id)

    def get_memory(
        self, chat_id: int, thread_id: Optional[int], limit: int = 24
    ) -> List[Dict[str, str]]:
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT role, author_name, content FROM chat_memory
                WHERE chat_id = ? AND (thread_id = ? OR (thread_id IS NULL AND ? IS NULL))
                ORDER BY id DESC LIMIT ?
                """,
                (chat_id, thread_id, thread_id, limit),
            ).fetchall()
            out: List[Dict[str, str]] = []
            for r in reversed(rows):
                name = r["author_name"]
                cnt = r["content"]
                out.append({
                    "role": r["role"],
                    "content": f"{name}: {cnt}" if r["role"] == "user" and name else cnt,
                })
            return out

    # ---------- long-term notes ----------

    @staticmethod
    def _note_thread(thread_id: Optional[int]) -> int:
        """NULL (General без ветки) храним как -1: SQLite не умеет NULL в PK."""
        return -1 if thread_id is None else int(thread_id)

    def get_notes(self, chat_id: int, thread_id: Optional[int]) -> Dict[str, Any]:
        """Долгосрочные заметки по чату: сводка, стиль и до какого id свернули."""
        tid = self._note_thread(thread_id)
        with self._locked():
            row = self._conn.execute(
                """
                SELECT notes, style, folded_upto FROM chat_notes
                WHERE chat_id = ? AND thread_id = ?
                """,
                (chat_id, tid),
            ).fetchone()
        if row is None:
            return {"notes": "", "style": "", "folded_upto": 0}
        return {"notes": row["notes"] or "", "style": row["style"] or "",
                "folded_upto": int(row["folded_upto"] or 0)}

    def set_notes(
        self,
        chat_id: int,
        thread_id: Optional[int],
        notes: str,
        style: str = "",
        folded_upto: Optional[int] = None,
    ) -> bool:
        """Atomically save notes, rejecting writers that are behind on folding.

        Read-check-write happens under a single lock/transaction: the incoming
        ``folded_upto`` must be >= the stored one (CAS), otherwise a stale fold
        worker must not overwrite fresher notes.  Returns True when applied.
        """
        tid = self._note_thread(thread_id)
        now = int(time.time())
        with self._locked(write=True):
            row = self._conn.execute(
                "SELECT folded_upto FROM chat_notes WHERE chat_id = ? AND thread_id = ?",
                (chat_id, tid),
            ).fetchone()
            saved_upto = int(row["folded_upto"] or 0) if row is not None else 0
            upto = saved_upto if folded_upto is None else int(folded_upto)
            if upto < saved_upto:
                return False
            self._conn.execute(
                """
                INSERT INTO chat_notes (chat_id, thread_id, notes, style, folded_upto, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, thread_id) DO UPDATE SET
                    notes = excluded.notes,
                    style = excluded.style,
                    folded_upto = excluded.folded_upto,
                    updated_at = excluded.updated_at
                """,
                (chat_id, tid, notes[:2000], style[:300], upto, now),
            )
            return True

    def memory_rows_after(self, chat_id: int, thread_id: Optional[int], after_id: int,
                          limit: int = 40) -> List[Dict[str, Any]]:
        """Записи новее `after_id` — то, что ещё не свёрнуто в заметки."""
        with self._locked():
            rows = self._conn.execute(
                """
                SELECT id, role, author_name, content FROM chat_memory
                WHERE chat_id = ? AND (thread_id = ? OR (thread_id IS NULL AND ? IS NULL))
                  AND id > ?
                ORDER BY id ASC LIMIT ?
                """,
                (chat_id, thread_id, thread_id, after_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def set_person_note(self, chat_id: int, name: str, note: str) -> None:
        with self._locked(write=True):
            key = " ".join((name or "").split()).casefold()[:32]
            if not key or not note.strip():
                return
            self._conn.execute(
                """
                INSERT INTO person_notes (chat_id, name_key, display_name, note, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(chat_id, name_key) DO UPDATE SET
                    note = excluded.note,
                    display_name = excluded.display_name,
                    updated_at = excluded.updated_at
                """,
                (chat_id, key, name[:64], note[:200], int(time.time())),
            )

    def get_person_notes(self, chat_id: int, names: List[str]) -> Dict[str, str]:
        """Заметки только про тех, кто сейчас в диалоге — не про всех подряд."""
        with self._locked():
            keys = {" ".join((n or "").split()).casefold()[:32] for n in names if n}
            if not keys:
                return {}
            rows = self._conn.execute(
                f"SELECT display_name, note FROM person_notes WHERE chat_id = ? "
                f"AND name_key IN ({','.join('?' * len(keys))})",
                (chat_id, *keys),
            ).fetchall()
        return {r["display_name"]: r["note"] for r in rows if r["note"]}

    def close(self) -> None:
        """Close the connection; safe to call twice and from any thread."""
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._conn.close()
