"""Inbound coalescing: answer a burst of messages once instead of three times.

People split one thought across several short messages, or send a sticker and
then the text that goes with it. Replying to each part produces three replies
that each miss the point.

Batches are keyed by ``(key, user_id)``: ``key`` identifies the conversation
(``(chat_id, thread_id)`` in the bot) and ``user_id`` identifies the author.
Two people writing in the same topic therefore never share a batch or a timer,
and a merged reply can never be attributed to the wrong person.

This is pure policy with no I/O and no clock of its own: the caller feeds parts
and asks when to flush. Defaults: flush 2.5 s after a text part, 4.5 s after a
sticker (a sticker is often followed by its caption), and never later than 8 s
after the first part of a batch.
"""

from __future__ import annotations

from collections.abc import Hashable
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

BatchKey = Tuple[Hashable, Optional[int]]


@dataclass(frozen=True)
class CoalescePolicy:
    text_quiet_sec: float = 2.5
    sticker_quiet_sec: float = 4.5
    max_batch_sec: float = 8.0
    max_parts: int = 6


@dataclass
class Batch:
    """Parts of one logical turn from one author."""

    parts: List[str] = field(default_factory=list)
    sticker_seen: bool = False
    first_at: float = 0.0
    last_at: float = 0.0

    def text(self) -> str:
        seen: List[str] = []
        for part in self.parts:
            cleaned = " ".join(part.split())
            if cleaned and cleaned not in seen:
                seen.append(cleaned)
        return " ".join(seen).strip()

    def empty(self) -> bool:
        return not any(p.strip() for p in self.parts)


class Coalescer:
    """Per-conversation, per-author coalescer.

    Feed parts with :meth:`add`, then either ask :meth:`flush_due` after a
    timer tick or force one author's batch with :meth:`flush`.
    """

    def __init__(self, policy: Optional[CoalescePolicy] = None) -> None:
        self.policy = policy or CoalescePolicy()
        self._batches: Dict[BatchKey, Batch] = {}

    def add(
        self,
        key: Hashable,
        text: str,
        user_id: Optional[int] = None,
        *,
        now: float,
        is_sticker: bool = False,
    ) -> None:
        """Append a part to this author's batch, starting it if needed."""
        batch_key = (key, user_id)
        batch = self._batches.get(batch_key)
        if batch is None or batch.empty():
            batch = Batch(first_at=now)
            self._batches[batch_key] = batch
        batch.parts.append(text or "")
        batch.last_at = now
        if is_sticker:
            batch.sticker_seen = True

    def parts_count(self, key: Hashable, user_id: Optional[int] = None) -> int:
        """How many raw parts this author has queued."""
        batch = self._batches.get((key, user_id))
        return len(batch.parts) if batch is not None else 0

    def flush(self, key: Hashable, user_id: Optional[int] = None) -> Optional[str]:
        """Force-flush one author's batch now and return the merged turn.

        Returns ``None`` when nothing is queued. Timers are handled by
        :meth:`flush_due`; this method is the explicit "answer now" path.
        """
        batch = self._batches.pop((key, user_id), None)
        if batch is None or batch.empty():
            return None
        return batch.text()

    def flush_due(self, now: float) -> List[Tuple[Hashable, Optional[int], str]]:
        """Return every batch whose quiet window or batch limit has expired.

        Each item is ``(key, user_id, merged_text)`` and is removed from the
        pending set. Batches that are not ready stay queued.
        """
        due: List[Tuple[Hashable, Optional[int], str]] = []
        for batch_key, batch in list(self._batches.items()):
            if batch.empty():
                del self._batches[batch_key]
                continue
            if not self._ready(batch, now):
                continue
            del self._batches[batch_key]
            merged = batch.text()
            if merged:
                key, user_id = batch_key
                due.append((key, user_id, merged))
        return due

    def drop(self, key: Hashable, user_id: Optional[int] = None) -> None:
        """Discard an author's pending batch without answering it."""
        self._batches.pop((key, user_id), None)

    def ready(self, key: Hashable, user_id: Optional[int] = None, now: float = 0.0) -> bool:
        """Готов ли батч автора к ответу прямо сейчас (тишина или лимит)."""
        batch = self._batches.get((key, user_id))
        return bool(batch is not None and not batch.empty() and self._ready(batch, now))

    def _ready(self, batch: Batch, now: float) -> bool:
        quiet_for = now - batch.last_at
        open_for = now - batch.first_at
        quiet_needed = (
            self.policy.sticker_quiet_sec if batch.sticker_seen
            else self.policy.text_quiet_sec
        )
        return (
            quiet_for >= quiet_needed
            or open_for >= self.policy.max_batch_sec
            or len(batch.parts) >= self.policy.max_parts
        )