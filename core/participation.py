"""Should Kiryxa join this conversation at all?

The old rule was a timer: whoever spoke to the bot got 20 seconds of undivided
attention, and everyone else was ignored. That misses the case where two people
are talking to each other and the bot barging in is just noise.

This module scores the message locally: no model call, no tokens. The score
decides whether an *unaddressed* message deserves a reply; a message that names
the bot or replies to it always does.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

# A chat moving faster than this is having its own conversation.
BUSY_PER_MINUTE = 12.0
# How many recent timestamps we keep to estimate the pace.
PACE_WINDOW_SEC = 60.0
# If the same two people talk back and forth this many times, it is their thread.
PRIVATE_THREAD_TURNS = 4

_QUESTION_RE = re.compile(r"[?？]\s*$")
_OPEN_FLOOR_RE = re.compile(
    r"(?i)\b(?:ребят\w*|народ|гои|все|кто\s+(?:знает|может|тут)|есть\s+кто|"
    r"подскажите|помогите|вопрос|кто\s+шарит|кому\s+надо|никто\s+не|"
    r"кто-нибудь|кто\s+нибудь)\b"
)
_CLOSING_RE = re.compile(
    r"(?i)^\s*(?:ага|угу|ок|окей|да|нет|понял|поняла|ясно|спс|спасибо|"
    r"благодарю|принял|приняла|lol|lmao|ахах\w*|хаха\w*|ok|okay|yep|nope)\W*$"
)
_INTEREST_RE = re.compile(
    r"(?i)\b(?:ключ\w*|кред\w*|нейросет\w*|нейронк\w*|модел\w*|апи|api|gpt|claude|gemini|"
    r"deepseek|qwen|kimi|лимит\w*|подписк\w*|триал\w*|халяв\w*|бесплатн\w*|free|"
    r"сервер\w*|скрипт\w*|код\b|софт\w*|игр\w*|катк\w*|стим|steam|фурри|фембой\w*"
    r"|лис\w*|котик\w*|кошк\w*|аниме|мем\w*|смешн\w*|прикол\w*)\b"
)
_EXCITED_RE = re.compile(r"(?:!{2,}|\?{2,}|😂|🤣|🔥|💀)")

# Score for a message that names the bot / replies to it: always high enough.
SCORE_ADDRESSED = 100.0
# Minimum score for an unaddressed message to earn a reply.
SCORE_THRESHOLD = 2.5


@dataclass(frozen=True)
class Participation:
    """Why the bot did (or did not) answer."""

    reply: bool
    reason: str
    score: float = 0.0


def score_message(
    text: str,
    *,
    from_user: int = 0,
    last_speaker: Optional[int] = None,
    recent_speakers: Optional[Iterable[int]] = None,
    replies_to_other: bool = False,
) -> float:
    """Local importance score; nothing here costs an API call."""
    body = (text or "").strip()
    if not body:
        return 0.0

    score = 0.0
    if _OPEN_FLOOR_RE.search(body):
        score += 2.0
    if _QUESTION_RE.search(body):
        score += 1.5
    if _INTEREST_RE.search(body):
        score += 1.5
    if _EXCITED_RE.search(body):
        score += 0.5
    if _CLOSING_RE.match(body):
        score -= 3.0
    if len(body) < 4:
        score -= 2.0
    if len(body.split()) >= 8:
        score += 0.5

    # A reply to somebody else is usually a thread we are not part of.
    if replies_to_other:
        score -= 2.0
    # Same person talking to themselves (a burst of messages) is not a dialogue.
    if last_speaker and from_user and last_speaker == from_user:
        score -= 1.0

    speakers = list(recent_speakers or [])
    if len(speakers) >= PRIVATE_THREAD_TURNS and len(set(speakers[-PRIVATE_THREAD_TURNS:])) <= 2:
        score -= 1.5

    return score


def should_join(
    text: str,
    *,
    addressed: bool = False,
    from_user: int = 0,
    last_speaker: Optional[int] = None,
    recent_speakers: Optional[Iterable[int]] = None,
    replies_to_other: bool = False,
    recent_messages: Optional[Deque[float]] = None,
    now: float = 0.0,
    busy_per_minute: float = BUSY_PER_MINUTE,
) -> Participation:
    """Decide whether to reply to an unaddressed message."""
    if addressed:
        return Participation(True, "addressed", SCORE_ADDRESSED)

    if recent_messages is not None and now:
        pace = sum(1 for ts in recent_messages if now - ts <= PACE_WINDOW_SEC)
        if pace > busy_per_minute:
            return Participation(False, "chat is busy", 0.0)

    score = score_message(
        text,
        from_user=from_user,
        last_speaker=last_speaker,
        recent_speakers=recent_speakers,
        replies_to_other=replies_to_other,
    )
    if score >= SCORE_THRESHOLD:
        return Participation(True, "interesting enough", score)
    return Participation(False, "not worth jumping in", score)
