"""Media bank for Kiryxa: stickers by emotion, GIFs, and auto-learning from chat.

Three sources of stickers, in priority order:
1. Sticker packs indexed by emoji (`getStickerSet`) — the model asks for a mood,
   the bank returns a sticker whose own emoji matches, not a random one.
2. Stickers learned from the chat itself (someone sent one — we remember it).
3. A small built-in fallback set, used until packs are loaded.

Replies may carry `[sticker:mood]`, `[sticker=😠]` or `[gif]`; the tag is stripped
from the visible text and turned into real Telegram media.
"""

from __future__ import annotations

import atexit
import json
import logging
import random
import re
import threading
import time
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.gif_library import ALL_GIFS, GIFS_BY_MOOD

logger = logging.getLogger("topic_guard.media_bank")

# Mood -> emoji. A mood is what the model can say; the emoji is how the sticker
# pack is indexed, so one mapping serves both directions.
MOOD_EMOJI: Dict[str, str] = {
    "think": "🤔",
    "hmm": "🤔",
    "sleep": "😴",
    "tired": "😴",
    "cry": "😭",
    "sad": "😢",
    "hello": "👋",
    "hi": "👋",
    "shock": "😱",
    "wow": "😱",
    "laugh": "😂",
    "lol": "😂",
    "love": "❤️",
    "heart": "❤️",
    "angry": "😠",
    "mad": "😠",
    "star": "⭐",
    "cool": "😎",
    "open": "😮",
    "scared": "😨",
    "facepalm": "🤦",
    "ok": "👍",
    "thumbsup": "👍",
    "fire": "🔥",
    "party": "🎉",
    "cat": "🐱",
    "fox": "🦊",
}

# Built-in fallback stickers (verified file_ids). Used only when a mood has no
# sticker from packs or chat learning.
DEFAULT_STICKERS: Dict[str, str] = {
    "think": "BAADAgADcE4AAj-8sUqxWwc_80Rd3wI",
    "sleep": "BAADAgAD51AAAr3vsUrc904QM3vbYQI",
    "cry": "CAADAgAD7xUAAkvlIUmNlsDeh4L9DwI",
    "hello": "CAADAgADDAEAAiI3jgR7D1jAYFgdrgI",
    "shock": "CAADAgADLD0AAmBOiUiWGgRWVfZwEgI",
    "sad": "CAADAgADlUAAAvR_kUj8XVezdrxBRgI",
    "star": "BAADAgAD6jQAAi1oaUvTdmpt3K_dRgI",
    "open": "CAADAgADaD4AAviGCErvYQRu1XH99QI",
}

# Curated GIF library (Tenor, uploaded to Telegram) grouped by mood.
DEFAULT_ANIMATIONS: List[str] = list(ALL_GIFS)

STICKER_TAG_RE = re.compile(
    r"\[(?:sticker|стикер)\s*[:=]\s*([^\]\s]+)\s*\]",
    re.IGNORECASE,
)
BARE_STICKER_TAG_RE = re.compile(r"\[(?:sticker|стикер)\]", re.IGNORECASE)
GIF_TAG_RE = re.compile(
    r"\[(?:gif|гифка|гиф)\s*(?::\s*([^\]]*))?\]",
    re.IGNORECASE,
)

USER_ASK_STICKER_RE = re.compile(
    r"(?i)\b(?:скинь|кинь|дай|отправь|покажи)\b.{0,20}\b(?:стикер|стик)\b",
)
USER_ASK_GIF_RE = re.compile(
    r"(?i)\b(?:скинь|кинь|дай|отправь|покажи)\b.{0,20}\b(?:гифку|гиф|анимацию)\b",
)

# Laughter and hype in the chat: a GIF fits better than words, if we have any.
GIF_TRIGGER_RE = re.compile(
    r"(?i)(?:\b(?:ахах\w*|хаха\w*|ору\b|ржу\b|угар\w*|ору\s+в\s+голос|lmao|lol\b|rofl)"
    r"|😂{2,}|🤣{2,})",
)
HYPE_RE = re.compile(
    r"(?i)(?:\b(?:погнали|погнали\b|го\b|поехали|ура|наконец-то|заработал\w*|живой|жит|збс)"
    r"|🔥{2,}|🎉{2,})"
)
# A GIF must not land more often than this in one chat.
GIF_COOLDOWN_SEC = 90.0
STICKER_COOLDOWN_SEC = 25.0
# Как часто вместо библиотеки отдавать то, что бот «стащил» из чата
LEARNED_STICKER_CHANCE = 0.5
LEARNED_GIF_CHANCE = 0.55
# Доля ответов без эмоции, к которым всё равно цепляется гифка
RANDOM_GIF_CHANCE = 0.15

# Ограничения на память и файл: старые записи вытесняются первыми.
MAX_BUCKET_SIZE = 300      # записей на один эмодзи-бакет
MAX_TOTAL_ITEMS = 2000     # суммарно записей в by_emoji / learned_by_emoji
MAX_LEARNED_ITEMS = 2000   # суммарно выученных стикеров / гифок
MAX_STICKER_KEYS = 2500    # ключей в self.stickers (включая built-in)
SAVE_DEBOUNCE_SEC = 2.0    # как часто максимум пишем файл на диск

# Быстрые подсказки эмоции по тексту (модель теги ставит не всегда)
_MOOD_PATTERNS: List[Tuple[str, "re.Pattern[str]"]] = [
    ("laugh", re.compile(r"(?i)(?:ахах\w*|хаха\w*|\bору\b|\bржу\b|угар\w*|\blol\b|\blmao\b|😂|🤣)")),
    ("hype", re.compile(r"(?i)(?:\bпогнали\b|\bпоехали\b|\bура\b|наконец-?то|заработал\w*|\bживой\b|\bзбс\b|🔥|🎉)")),
    ("angry", re.compile(r"(?i)(?:\bбесит\b|\bзадолбал\w*|\bдостал\w*|\bтупой\b|\bеблан\w*|\bхуйня\b|\bненавижу\b|😠|🤬)")),
    ("sad", re.compile(r"(?i)(?:\bгрустн\w*|\bпечаль\w*|\bжаль\b|\bобидно\b|\bувы\b|😢|😭)")),
    ("love", re.compile(r"(?i)(?:\bлюблю\b|\bобожаю\b|\bняшн\w*|\bмилый\b|😍|❤️|🥰)")),
    ("shock", re.compile(r"(?i)(?:\bохуел\w*|\bнихрена\b|\bсерь[её]зно\b|\bда\s+ладно\b|😱|🤯)")),
    ("think", re.compile(r"(?i)(?:\bдумаю\b|\bхм+\b|\bнаверное\b|\bможет\s+быть\b|🤔)")),
    ("party", re.compile(r"(?i)(?:\bднюх\w*|\bпраздник\w*|\bпоздравля\w*|🎂|🥳)")),
]

VARIATION_SELECTOR = "\ufe0f"
ZWJ = "\u200d"


def normalize_emoji(value: str) -> str:
    """Telegram reports emoji with/without variation selectors — compare them equal."""
    return (value or "").replace(VARIATION_SELECTOR, "").replace(ZWJ, "").strip()


def _mood_emoji(mood: str) -> str:
    """Mood word, or a raw emoji, into the emoji that indexes the sticker packs."""
    raw = (mood or "").strip()
    if not raw:
        return ""
    # The model may pass the emoji itself: `[sticker=😠]`
    norm_raw = normalize_emoji(raw)
    if norm_raw in {normalize_emoji(e) for e in MOOD_EMOJI.values()}:
        return raw
    key = raw.lower()
    if key in MOOD_EMOJI:
        return MOOD_EMOJI[key]
    for name, emoji in MOOD_EMOJI.items():
        if name in key:
            return emoji
    return ""


class MediaBank:
    """Sticker/GIF library with emoji indexing, pack loading and chat learning."""

    # Limits are class attributes so tests/subclasses can tune them.
    MAX_BUCKET_SIZE = MAX_BUCKET_SIZE
    MAX_TOTAL_ITEMS = MAX_TOTAL_ITEMS
    MAX_LEARNED_ITEMS = MAX_LEARNED_ITEMS
    MAX_STICKER_KEYS = MAX_STICKER_KEYS
    SAVE_DEBOUNCE_SEC = SAVE_DEBOUNCE_SEC

    def __init__(
        self,
        storage_path: Optional[str | Path] = None,
        sticker_sets: Optional[List[str]] = None,
    ) -> None:
        # The bot learns stickers from worker threads while the polling thread
        # reads them.  RLock because public methods may call each other
        # (maybe_reaction -> get_sticker/get_animation, parse_reply -> get_*).
        self._lock = threading.RLock()
        self.stickers: Dict[str, str] = dict(DEFAULT_STICKERS)
        # normalized emoji -> [file_id, ...] from packs and chat learning
        self.by_emoji: Dict[str, List[str]] = {}
        # Curated GIF library (moods) and GIFs learned from the chat
        self.animations: List[str] = list(DEFAULT_ANIMATIONS)
        self.learned_animations: List[str] = []
        self.learned_stickers: List[str] = []
        # emoji -> sticker ids that came from the chat itself (not from a pack)
        self.learned_by_emoji: Dict[str, List[str]] = {}
        # Global insertion order per emoji bucket, used to evict the oldest
        # ids once a total limit is exceeded.
        self._by_emoji_order: "OrderedDict[Tuple[str, str], None]" = OrderedDict()
        self._learned_by_emoji_order: "OrderedDict[Tuple[str, str], None]" = OrderedDict()
        self.storage_path = Path(storage_path) if storage_path else None
        self.sticker_sets = list(sticker_sets or [])
        self._last_sticker_at: Dict[str, float] = {}
        self._last_gif_at: Dict[str, float] = {}
        # Debounced persistence: mutations only mark the bank dirty; a daemon
        # timer writes the file at most once per SAVE_DEBOUNCE_SEC.
        self._dirty = False
        self._save_timer: Optional[threading.Timer] = None
        self.load()
        self._seed_builtin_emoji()
        if self.storage_path:
            # Normal interpreter shutdown (including SIGTERM handled by main.py)
            # must not lose the last debounce window.
            atexit.register(self._flush_at_exit)

    def _seed_builtin_emoji(self) -> None:
        """Mood -> emoji -> built-in sticker, so moods resolve before any pack loads."""
        with self._lock:
            for mood, fid in DEFAULT_STICKERS.items():
                emoji = MOOD_EMOJI.get(mood)
                if not emoji or not fid:
                    continue
                self._remember_emoji(
                    self.by_emoji,
                    self._by_emoji_order,
                    normalize_emoji(emoji),
                    fid,
                    self.MAX_BUCKET_SIZE,
                    self.MAX_TOTAL_ITEMS,
                )

    # ---------- bounded collections ----------

    @staticmethod
    def _remember_ordered(items: List[str], file_id: str, limit: int) -> bool:
        """Append id, evicting the oldest ones. Returns True if it was new."""
        if file_id in items:
            return False
        items.append(file_id)
        if len(items) > limit:
            del items[: len(items) - limit]
        return True

    @staticmethod
    def _remember_emoji(
        mapping: Dict[str, List[str]],
        order: "OrderedDict[Tuple[str, str], None]",
        emoji: str,
        file_id: str,
        per_bucket: int,
        total: int,
    ) -> bool:
        """Add id to an emoji bucket, enforcing per-bucket and global caps.

        ``order`` keeps insertion order across all buckets, so once ``total``
        is exceeded the oldest id is evicted instead of the whole bucket.
        Returns True if the id was not present in the bucket before.
        """
        bucket = mapping.setdefault(emoji, [])
        is_new = file_id not in bucket
        if not is_new:
            bucket.remove(file_id)
        bucket.append(file_id)
        # Re-adding an id moves it to the newest position.
        order[(emoji, file_id)] = None
        while len(bucket) > per_bucket:
            old = bucket.pop(0)
            order.pop((emoji, old), None)
        while len(order) > total:
            (old_emoji, old_file_id), _ = order.popitem(last=False)
            old_bucket = mapping.get(old_emoji)
            if not old_bucket:
                continue
            try:
                old_bucket.remove(old_file_id)
            except ValueError:
                continue
            if not old_bucket:
                mapping.pop(old_emoji, None)
        return is_new

    def _trim_stickers_locked(self) -> None:
        """Cap the sticker dict; built-in mood keys are never evicted."""
        excess = len(self.stickers) - self.MAX_STICKER_KEYS
        if excess <= 0:
            return
        for key in list(self.stickers):
            if excess <= 0:
                break
            if key in DEFAULT_STICKERS:
                continue
            self.stickers.pop(key, None)
            excess -= 1

    # ---------- persistence ----------

    def _mark_dirty_locked(self) -> None:
        """Schedule a debounced write; caller must hold the lock."""
        if not self.storage_path:
            return
        self._dirty = True
        timer = self._save_timer
        if timer is None or not timer.is_alive():
            timer = threading.Timer(self.SAVE_DEBOUNCE_SEC, self._save_timer_fired)
            timer.daemon = True
            timer.name = "media-bank-save"
            self._save_timer = timer
            timer.start()

    def _save_timer_fired(self) -> None:
        with self._lock:
            # A flush() or a newer timer may have superseded this one while it
            # was waiting for the lock: never write stale state.
            if self._save_timer is not threading.current_thread():
                return
            self._save_timer = None
            if not self._dirty:
                return
            if self._write_locked():
                self._dirty = False

    def _write_locked(self) -> bool:
        """Write a bounded snapshot; caller must hold the lock."""
        if not self.storage_path:
            return False
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "stickers": dict(self.stickers),
                "by_emoji": {
                    k: list(v[-self.MAX_BUCKET_SIZE:])
                    for k, v in self.by_emoji.items()
                },
                "learned_animations": list(self.learned_animations[-self.MAX_LEARNED_ITEMS:]),
                "learned_stickers": list(self.learned_stickers[-self.MAX_LEARNED_ITEMS:]),
                "learned_by_emoji": {
                    k: list(v[-self.MAX_BUCKET_SIZE:])
                    for k, v in self.learned_by_emoji.items()
                },
            }
            # Atomic replace: a crash mid-write must not leave a broken JSON.
            tmp_path = self.storage_path.with_name(self.storage_path.name + ".tmp")
            tmp_path.write_text(
                json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            tmp_path.replace(self.storage_path)
            return True
        except Exception as e:
            logger.debug("Failed to save media bank: %s", e)
            return False

    def _flush_at_exit(self) -> None:
        try:
            with self._lock:
                # Do not recreate a deleted temp directory or write unchanged
                # state: only pending changes need saving at shutdown.
                if self._dirty:
                    self.flush()
        except Exception:
            pass

    def load(self) -> None:
        with self._lock:
            if not self.storage_path or not self.storage_path.is_file():
                return
            changed = False
            try:
                data = json.loads(self.storage_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    return
                legacy_learned: List[str] = []
                for k, v in (data.get("stickers") or {}).items():
                    if k and v:
                        self.stickers[str(k).lower()] = str(v)
                        # Старые версии писали выученные стикеры как learned_N без эмодзи
                        if str(k).lower().startswith("learned_"):
                            legacy_learned.append(str(v))
                self._trim_stickers_locked()
                for emoji, ids in (data.get("by_emoji") or {}).items():
                    if emoji and isinstance(ids, list):
                        norm = normalize_emoji(str(emoji))
                        for fid in ids:
                            if fid:
                                changed |= self._remember_emoji(
                                    self.by_emoji,
                                    self._by_emoji_order,
                                    norm,
                                    str(fid),
                                    self.MAX_BUCKET_SIZE,
                                    self.MAX_TOTAL_ITEMS,
                                )
                # Only chat-learned GIFs are persisted; the curated library lives in code
                builtin = set(DEFAULT_ANIMATIONS)
                for anim in (data.get("learned_animations") or []):
                    # Старые версии складывали сюда же встроенные гифки — выбрасываем
                    if anim and str(anim) not in builtin:
                        changed |= self._remember_ordered(
                            self.learned_animations, str(anim), self.MAX_LEARNED_ITEMS
                        )
                for fid in (data.get("learned_stickers") or []) + legacy_learned:
                    if fid:
                        changed |= self._remember_ordered(
                            self.learned_stickers, str(fid), self.MAX_LEARNED_ITEMS
                        )
                # Миграция: эмодзи-ключ, чей file_id выучен из чата, становится «своим»
                for emoji_key, fid in list(self.stickers.items()):
                    if emoji_key.startswith("learned_"):
                        continue
                    if fid in self.learned_stickers and len(emoji_key) <= 4:
                        changed |= self._remember_emoji(
                            self.learned_by_emoji,
                            self._learned_by_emoji_order,
                            normalize_emoji(emoji_key),
                            fid,
                            self.MAX_BUCKET_SIZE,
                            self.MAX_TOTAL_ITEMS,
                        )
                for emoji, ids in (data.get("learned_by_emoji") or {}).items():
                    if emoji and isinstance(ids, list):
                        norm = normalize_emoji(str(emoji))
                        for fid in ids:
                            if fid:
                                changed |= self._remember_emoji(
                                    self.learned_by_emoji,
                                    self._learned_by_emoji_order,
                                    norm,
                                    str(fid),
                                    self.MAX_BUCKET_SIZE,
                                    self.MAX_TOTAL_ITEMS,
                                )
            except Exception as e:
                logger.debug("Failed to load media bank: %s", e)
            if changed:
                # Persist the trimmed/migrated version on the normal debounce.
                self._mark_dirty_locked()

    def flush(self) -> None:
        """Force pending changes to disk immediately (explicit save/shutdown)."""
        with self._lock:
            timer = self._save_timer
            self._save_timer = None
            if timer is not None:
                timer.cancel()
            if not self.storage_path:
                self._dirty = False
                return
            if self._write_locked():
                self._dirty = False

    def save(self) -> None:
        """Explicit save request: write now, bypassing the debounce timer."""
        self.flush()

    # ---------- learning ----------

    def learn_sticker(self, file_id: str, emoji: Optional[str] = None) -> None:
        """Remember a sticker from the chat and index it by emoji when known."""
        if not file_id:
            return
        with self._lock:
            changed = self._remember_ordered(
                self.learned_stickers, file_id, self.MAX_LEARNED_ITEMS
            )
            key = (emoji or "").strip()
            if key:
                if self.stickers.get(key) != file_id:
                    self.stickers[key] = file_id
                    changed = True
                norm = normalize_emoji(key)
                changed |= self._remember_emoji(
                    self.by_emoji,
                    self._by_emoji_order,
                    norm,
                    file_id,
                    self.MAX_BUCKET_SIZE,
                    self.MAX_TOTAL_ITEMS,
                )
                changed |= self._remember_emoji(
                    self.learned_by_emoji,
                    self._learned_by_emoji_order,
                    norm,
                    file_id,
                    self.MAX_BUCKET_SIZE,
                    self.MAX_TOTAL_ITEMS,
                )
            # ВАЖНО: ключ learned_N добавляем только если у стикера нет эмодзи.
            # Иначе тот же file_id живёт под двумя ключами и выпадает вдвое чаще.
            elif file_id not in self.stickers.values():
                idx = len(self.stickers)
                learned_key = f"learned_{idx}"
                while learned_key in self.stickers:
                    idx += 1
                    learned_key = f"learned_{idx}"
                self.stickers[learned_key] = file_id
                changed = True
            self._trim_stickers_locked()
            if changed:
                self._mark_dirty_locked()

    def learn_animation(self, file_id: str) -> None:
        """Remember a GIF someone posted in the chat; it joins the random pool."""
        if not file_id:
            return
        with self._lock:
            if file_id in self.animations or file_id in self.learned_animations:
                return
            if self._remember_ordered(
                self.learned_animations, file_id, self.MAX_LEARNED_ITEMS
            ):
                self._mark_dirty_locked()

    def load_pack(self, stickers: List[Dict[str, Any]]) -> int:
        """Index a sticker set (`getStickerSet().stickers`) by emoji. Returns added count."""
        added = 0
        with self._lock:
            for stk in stickers or []:
                fid = str((stk or {}).get("file_id") or "")
                emoji = str((stk or {}).get("emoji") or "")
                if not fid or not emoji:
                    continue
                if self._remember_emoji(
                    self.by_emoji,
                    self._by_emoji_order,
                    normalize_emoji(emoji),
                    fid,
                    self.MAX_BUCKET_SIZE,
                    self.MAX_TOTAL_ITEMS,
                ):
                    added += 1
            if added:
                self._mark_dirty_locked()
        return added

    # ---------- selection ----------

    def _pick_from_emoji(self, emoji: str) -> Optional[str]:
        """Pick a pack/built-in sticker for emoji, not a chat-learned one."""
        with self._lock:
            norm = normalize_emoji(emoji)
            bucket = self.by_emoji.get(norm)
            if not bucket:
                return None
            learned = set(self.learned_by_emoji.get(norm, ()))
            pack_only = [fid for fid in bucket if fid not in learned]
            if pack_only:
                return random.choice(pack_only)
            return None

    @staticmethod
    def _gif_pool_for_mood(mood: str) -> List[str]:
        """Curated GIFs whose mood matches the requested word or emoji."""
        key = (mood or "").strip().lower()
        if not key:
            return []
        if key in GIFS_BY_MOOD:
            return list(GIFS_BY_MOOD[key])
        for name, ids in GIFS_BY_MOOD.items():
            if name in key or key in name:
                return list(ids)
        # The model may pass the mood emoji itself (`[gif:😂]`).
        emoji = _mood_emoji(key)
        if emoji:
            norm = normalize_emoji(emoji)
            for name, ids in GIFS_BY_MOOD.items():
                mapped = MOOD_EMOJI.get(name)
                if mapped and normalize_emoji(mapped) == norm:
                    return list(ids)
        return []

    def get_sticker(self, mood: Optional[str] = None, chat_key: str = "") -> Optional[str]:
        """Sticker for a mood/emoji, with a per-chat cooldown so it is not spammed."""
        with self._lock:
            now = time.time()
            if chat_key and now - self._last_sticker_at.get(chat_key, 0.0) < STICKER_COOLDOWN_SEC:
                return None

            fid: Optional[str] = None
            raw_mood = (mood or "").strip()
            emoji = _mood_emoji(raw_mood)
            if raw_mood:
                # Explicit mood: never mix in a random learned sticker of
                # another mood. Pack first, then chat-learned for this emoji.
                if emoji:
                    # 1. Sticker pack (or built-in) indexed by this emoji.
                    fid = self._pick_from_emoji(emoji)
                    # 2. Chat-learned sticker with the same emoji.
                    if not fid:
                        norm = normalize_emoji(emoji)
                        kin = self.learned_by_emoji.get(norm)
                        if kin:
                            fid = random.choice(kin)
                        else:
                            bucket = self.by_emoji.get(norm)
                            if bucket:
                                fid = random.choice(bucket)
                # Mood word stored directly in self.stickers (legacy/builtin).
                if not fid:
                    low = raw_mood.lower()
                    if low in self.stickers:
                        fid = self.stickers[low]
                    else:
                        for k, v in self.stickers.items():
                            if low and (low in k or k in low):
                                fid = v
                                break
                # Only when nothing mood-specific exists: any learned sticker.
                if not fid and self.learned_stickers:
                    fid = random.choice(self.learned_stickers)
            else:
                # No mood: preserve the old "chat-learned often" behaviour.
                if self.learned_stickers and random.random() < LEARNED_STICKER_CHANCE:
                    kin_pool = [v for ids in self.learned_by_emoji.values() for v in ids]
                    fid = random.choice(kin_pool or self.learned_stickers)
                if not fid:
                    if self.learned_stickers and random.random() < 0.55:
                        fid = random.choice(self.learned_stickers)
                    else:
                        preferred = [
                            self.stickers[k]
                            for k in ("think", "sleep", "shock")
                            if k in self.stickers
                        ]
                        if preferred:
                            fid = random.choice(preferred)
                        elif self.stickers:
                            fid = random.choice(list(self.stickers.values()))

            if fid and chat_key:
                self._last_sticker_at[chat_key] = now
            return fid

    def get_animation(self, mood: Optional[str] = None, chat_key: str = "") -> Optional[str]:
        """GIF for a mood when the library has one, else a learned/library GIF."""
        with self._lock:
            now = time.time()
            if chat_key and now - self._last_gif_at.get(chat_key, 0.0) < GIF_COOLDOWN_SEC:
                return None

            pool: List[str] = []
            raw_mood = (mood or "").strip()
            if raw_mood:
                pool = self._gif_pool_for_mood(raw_mood)
                if not pool:
                    # No curated GIF for this mood: learned GIFs are the only
                    # fallback. A real mood pool is never replaced by them.
                    pool = list(self.learned_animations)
                    if not pool:
                        pool = list(self.animations or ALL_GIFS)
            else:
                # Без настроения чаще кидаем ту гифку, которую притащили из чата
                curated = self.animations or list(ALL_GIFS)
                pool = list(curated) + list(self.learned_animations)
                # Гифка из чата перебивает библиотеку примерно в половине случаев:
                # иначе бот кидает только скачанное нами и игнорирует «своё».
                if self.learned_animations and random.random() < LEARNED_GIF_CHANCE:
                    pool = list(self.learned_animations)
            if not pool:
                return None
            if chat_key:
                self._last_gif_at[chat_key] = now
            return random.choice(pool)

    # ---------- parsing ----------

    def parse_reply(
        self, reply_text: str, chat_key: str = ""
    ) -> Tuple[str, Optional[str], Optional[str]]:
        """Extract `[sticker:mood]` / `[sticker=😠]` / `[gif]` and resolve real file ids."""
        if not reply_text:
            return "", None, None

        sticker_fid: Optional[str] = None
        anim_fid: Optional[str] = None

        match = STICKER_TAG_RE.search(reply_text)
        if match:
            sticker_fid = self.get_sticker(match.group(1), chat_key=chat_key)
            reply_text = STICKER_TAG_RE.sub("", reply_text)
        elif BARE_STICKER_TAG_RE.search(reply_text):
            sticker_fid = self.get_sticker(None, chat_key=chat_key)
            reply_text = BARE_STICKER_TAG_RE.sub("", reply_text)

        gif_match = GIF_TAG_RE.search(reply_text)
        if gif_match:
            if not sticker_fid:
                mood = (gif_match.group(1) or "").strip()
                anim_fid = self.get_animation(mood=mood, chat_key=chat_key)
            reply_text = GIF_TAG_RE.sub("", reply_text)

        return reply_text.strip(), sticker_fid, anim_fid

    def check_user_ask(self, text: str) -> Tuple[bool, bool]:
        """Did the user explicitly ask for a sticker or GIF? (sticker, gif)"""
        return (
            bool(USER_ASK_STICKER_RE.search(text or "")),
            bool(USER_ASK_GIF_RE.search(text or "")),
        )

    def guess_mood(self, text: str) -> str:
        """Эмоция по тексту, когда модель не поставила тег."""
        low = (text or "").lower()
        for mood, pattern in _MOOD_PATTERNS:
            if pattern.search(low):
                return mood
        return ""

    def maybe_reaction(
        self, user_text: str, reply_text: str, chat_key: str = "", chance: float = 0.45
    ) -> Tuple[Optional[str], Optional[str]]:
        """Реакция без тега от модели.

        Приоритет: эмоция -> гифка (laugh/hype), иначе 50/50 стикер или гифка.
        Совсем без эмоции тоже иногда кидаем случайную гифку — 15% ответов,
        чтобы бот не молчал картинками в скучных темах.

        ``chance`` is the probability of reacting at all when a mood is found.
        It used to be ignored for laugh/hype, which made auto-GIFs fire almost
        always; now every mood branch honours it.
        """
        with self._lock:
            try:
                chance_value = float(chance)
            except (TypeError, ValueError):
                chance_value = 0.45
            if chance_value <= 0.0:
                return None, None
            chance_value = min(chance_value, 1.0)

            mood = self.guess_mood(user_text) or self.guess_mood(reply_text)

            if mood:
                if random.random() >= chance_value:
                    return None, None
                if mood in ("laugh", "hype"):
                    gif = self.get_animation(mood=mood, chat_key=chat_key)
                    if gif:
                        return None, gif
                    # гифка на кулдауне — можно добить стикером
                    return self.get_sticker(mood, chat_key=chat_key), None
                if random.random() < 0.5:
                    gif = self.get_animation(mood=mood, chat_key=chat_key)
                    if gif:
                        return None, gif
                return self.get_sticker(mood, chat_key=chat_key), None

            # Настроения нет — иногда просто рандомная гифка «в тему разговора»
            if random.random() < RANDOM_GIF_CHANCE:
                return None, self.get_animation(mood=None, chat_key=chat_key)
            return None, None

    def wants_reaction_gif(self, text: str, chat_key: str = "") -> Optional[str]:
        """Unprompted GIF for laughter/hype — rate-limited per chat."""
        if not text:
            return None
        mood = ""
        if GIF_TRIGGER_RE.search(text):
            mood = "laugh"
        elif HYPE_RE.search(text):
            mood = "hype"
        if not mood:
            return None
        return self.get_animation(mood=mood, chat_key=chat_key)
