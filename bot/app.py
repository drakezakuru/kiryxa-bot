"""Main TopicGuard Bot handler: forum topic moderation, forwarding off-topic to General, and admin UI."""

from __future__ import annotations

import html
import logging
import random
import subprocess
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from core.classifier import classify_message, classify_topic_name
from core.coalesce import Coalescer
from core.participation import should_join
from core.persona import (
    RateLimiter,
    _sanitize_reply,
    generate_reply,
    is_addressed,
    recent_replies,
    screen_level,
    screen_request,
)
from core.elevenlabs import (
    DEFAULT_MODEL,
    DEFAULT_VOICE_ID,
    VOICE_BY_ID,
    VOICE_BY_NAME,
    VOICE_CATALOG,
    ElevenLabsClient,
)
from core.media_bank import MediaBank
from core.muse_client import MuseClient, detect_media_intent
from core.notes import (
    FOLD_EVERY,
    FOLD_BATCH,
    deterministic_profile,
    fold_notes,
    merge_notes,
    notes_block,
)
from core.vision_client import VisionClient
from core.config import Config
from core.store import Store

from .tg_api import (
    MockTelegramAPI,
    TelegramAPI,
    TelegramError,
    inline_btn,
    inline_keyboard,
)

logger = logging.getLogger("topic_guard.bot")

# Что сказать, если модель промолчала, а ответ нужен
SILENT_FALLBACKS = (
    "чёт подвис, повтори ещё раз",
    "у меня тут мысль потерялась, скажи ещё раз?",
    "не, я не пропал, просто затупил. повтори",
)

MODE_LABELS: Dict[str, str] = {
    "smart": "🧠 Умный (Ключи + Триал инфа)",
    "keys": "🔑 Только ключи / коды / раздачи",
    "trials": "🎁 Фри триалы / гайды / ссылки",
    "admin": "👑 Только посты админов",
    "off": "💬 Без фильтра (чат / флуд)",
    "general": "🏠 General (сюда летит оффтоп)",
}


def chat_id_to_c_id(chat_id: int) -> Optional[str]:
    """Convert supergroup chat_id (-1001234567890) to t.me/c/1234567890 internal ID."""
    s = str(chat_id)
    if s.startswith("-100") and len(s) > 4:
        return s[4:]
    if s.startswith("-") and len(s) > 1:
        return s[1:]
    return None


def build_message_link(
    chat_id: int,
    message_id: int,
    thread_id: Optional[int] = None,
    chat_username: Optional[str] = None,
) -> Optional[str]:
    """Build a deep link to a specific message inside a supergroup / forum topic."""
    if chat_username:
        if thread_id:
            return f"https://t.me/{chat_username}/{thread_id}/{message_id}"
        return f"https://t.me/{chat_username}/{message_id}"
    cid = chat_id_to_c_id(chat_id)
    if not cid:
        return None
    if thread_id:
        return f"https://t.me/c/{cid}/{thread_id}/{message_id}"
    return f"https://t.me/c/{cid}/{message_id}"


def build_topic_link(
    chat_id: int,
    thread_id: Optional[int],
    chat_username: Optional[str] = None,
) -> Optional[str]:
    """Build a deep link to a forum topic."""
    if not thread_id:
        return None
    if chat_username:
        return f"https://t.me/{chat_username}/{thread_id}"
    cid = chat_id_to_c_id(chat_id)
    if not cid:
        return None
    return f"https://t.me/c/{cid}/{thread_id}"


def format_user_mention(user: Dict[str, Any]) -> str:
    uid = user.get("id", 0)
    username = user.get("username")
    first = str(user.get("first_name") or "").strip()
    last = str(user.get("last_name") or "").strip()
    full_name = f"{first} {last}".strip() or f"User {uid}"
    escaped_name = html.escape(full_name)
    if username:
        return f'<a href="https://t.me/{html.escape(username)}">{escaped_name}</a> (@{html.escape(username)})'
    return f'<a href="tg://user?id={uid}">{escaped_name}</a>'


class TopicGuardBot:
    """Forum topic moderation & General-chat router bot."""

    def __init__(
        self,
        api: TelegramAPI | MockTelegramAPI,
        store: Store,
        config: Config,
    ) -> None:
        self.api = api
        self.store = store
        self.config = config
        self.bot_id: int = 0
        self.bot_username: str = ""
        # Cache of (chat_id) -> (timestamp, set_of_admin_user_ids)
        self._admin_cache: Dict[int, Tuple[float, Set[int]]] = {}
        self._lock = threading.RLock()
        self._chat_limiter = RateLimiter(per_user_sec=2.0, global_per_min=25)
        self._quip_limiter = RateLimiter(per_user_sec=15.0, global_per_min=8)
        # (chat_id, thread_id) -> list of {"role": str, "content": str}
        self._dialogue_history: Dict[Tuple[int, Optional[int]], List[Dict[str, str]]] = {}
        # (chat_id, thread_id) -> (user_id, last_active_timestamp)
        self._active_dialogues: Dict[Tuple[int, Optional[int]], Tuple[int, float]] = {}
        # Participation: who spoke last, recent speakers and message pace per chat
        self._last_speaker: Dict[Tuple[int, Optional[int]], int] = {}
        self._recent_speakers: Dict[Tuple[int, Optional[int]], deque] = {}
        self._recent_message_times: Dict[Tuple[int, Optional[int]], deque] = {}
        # Coalescing: merge a burst of short messages into one turn
        self._coalescers: Dict[Tuple[int, Optional[int]], Coalescer] = {}
        # Anti-spam & auto-mute tracking
        # (chat_id, user_id) -> list of (timestamp, message_id)
        self._user_spam_events: Dict[Tuple[int, int], List[Tuple[float, int]]] = {}
        # (chat_id, user_id) -> offense count
        self._user_mute_counts: Dict[Tuple[int, int], int] = {}
        # (chat_id, user_id) -> until_timestamp
        self._muted_until: Dict[Tuple[int, int], float] = {}
        # Cache of message_ids already mirrored from General to keys topic
        # (chat_id, message_id): одинаковые message_id бывают в разных чатах
        self._mirrored_from_general: Set[Tuple[int, int]] = set()
        # Последние vision-пары (эмодзи, коммент) по чатам: бот не должен
        # реагировать на разные гифки одной и той же фразой.
        self._recent_vision: Dict[str, deque] = {}
        # chat_key -> (тип последней генерации, timestamp, промпт): нужен, чтобы
        # понимать правки вроде «сделай нормального гоя а не утку».
        self._last_media: Dict[str, Tuple[str, float, str]] = {}
        # Медленные вещи (LLM, поиск, TTS) уходят в пул, чтобы polling не стоял.
        # Порядок внутри одного чата держит lock на (chat_id, thread_id).
        self._persona_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="persona")
        self._persona_locks: Dict[Tuple[int, Optional[int]], threading.Lock] = {}
        self._persona_locks_guard = threading.Lock()
        self._msg_seq = 0
        self._last_muse_restart = 0.0
        self.media_bank = MediaBank(storage_path=Path(self.config.data_dir) / "media_bank.json")
        self.elevenlabs: Optional[ElevenLabsClient] = None
        if self.config.elevenlabs_enabled and self.config.elevenlabs_keys:
            self.elevenlabs = ElevenLabsClient(
                api_keys=self.config.elevenlabs_keys,
                voice_id=self.config.elevenlabs_voice_id or DEFAULT_VOICE_ID,
                model=self.config.elevenlabs_model or DEFAULT_MODEL,
            )
            logger.info(
                "ElevenLabs: %d ключей, голос %s",
                len(self.config.elevenlabs_keys),
                self.elevenlabs.voice_id,
            )
        self.muse: Optional[MuseClient] = None
        if self.config.muse_enabled:
            self.muse = MuseClient(
                base_url=self.config.muse_api_url,
                default_voice=self.config.muse_voice,
                llm_api_url=self.config.ai_api_url,
                llm_api_key=self.config.ai_api_key,
                llm_model=self.config.ai_model,
                elevenlabs=self.elevenlabs,
            )
        self.vision: Optional[VisionClient] = None
        if self.config.vision_enabled:
            self.vision = VisionClient(
                base_url=self.config.vision_api_url,
                api_key=self.config.vision_api_key,
                model=self.config.vision_model,
            )

    def init_bot_info(self) -> None:
        try:
            me = self.api.get_me()
            self.bot_id = int(me.get("id", 0))
            self.bot_username = str(me.get("username") or "")
            logger.info("Connected as @%s (id=%s)", self.bot_username, self.bot_id)
        except Exception as e:
            logger.warning("Failed to get bot info: %s", e)

        try:
            self.api.set_my_commands(
                [
                    {"command": "panel", "description": "Панель управления ветками"},
                    {"command": "guard", "description": "Режим текущей ветки (smart/keys/trials/admin/off)"},
                    {"command": "setgeneral", "description": "Назначить эту ветку как General для оффтопа"},
                    {"command": "photo", "description": "Сгенерировать фото через Muse"},
                    {"command": "video", "description": "Сгенерировать видео через Muse"},
                    {"command": "voice", "description": "Озвучить текст / ГС через Muse"},
                    {"command": "voices", "description": "Выбрать голос бота (в ЛС)"},
                    {"command": "voicemode", "description": "Отвечать голосом: on/off"},
                    {"command": "status", "description": "Статус и статистика веток"},
                    {"command": "wl", "description": "Добавить/убрать юзера в белый список (реплаем)"},
                    {"command": "help", "description": "Справка по командам"},
                ]
            )
        except Exception as e:
            logger.debug("Could not set commands: %s", e)

        self._load_sticker_packs()

    def _author_own_text(self, chat_id: int, mem_tid: Optional[int], author_name: str) -> str:
        """Что этот человек писал сам — чтобы бот не обвинял его в чужих действиях."""
        try:
            rows = self.store.get_memory(chat_id, mem_tid, limit=80)
        except Exception:
            return ""
        prefix = f"{author_name}:"
        return "\n".join(
            r["content"] for r in rows
            if r.get("role") == "user" and str(r.get("content") or "").startswith(prefix)
        )

    def _memory_block(self, chat_id: int, mem_tid: Optional[int], author_name: str) -> str:
        """Что бот помнит о чате и о собеседнике — отдельным блоком в промпт."""
        try:
            notes = self.store.get_notes(chat_id, mem_tid)
            people = self.store.get_person_notes(chat_id, [author_name])
            # Манера человека считается по его же репликам, без вызова модели
            rows = self.store.get_memory(chat_id, mem_tid, limit=60)
            profile = deterministic_profile(rows)
            if profile and author_name not in people:
                people[author_name] = profile
            return notes_block(notes.get("notes", ""), notes.get("style", ""), people)
        except Exception as e:
            logger.debug("memory block failed: %s", e)
            return ""

    def _maybe_fold_notes(self, chat_id: int, mem_tid: Optional[int]) -> None:
        """Раз в N ходов свернуть старые реплики в заметки (в фоне)."""
        if not (self.config.ai_enabled and self.config.ai_api_key):
            return
        threading.Thread(
            target=self._fold_notes_worker,
            args=(chat_id, mem_tid),
            daemon=True,
        ).start()

    def _fold_notes_worker(self, chat_id: int, mem_tid: Optional[int]) -> None:
        try:
            state = self.store.get_notes(chat_id, mem_tid)
            rows = self.store.memory_rows_after(
                chat_id, mem_tid, int(state.get("folded_upto") or 0), limit=FOLD_BATCH
            )
            if len(rows) < FOLD_EVERY:
                return
            result = fold_notes(
                old_notes=str(state.get("notes") or ""),
                turns=rows,
                api_url=self.config.ai_api_url,
                api_key=self.config.ai_api_key,
                model=self.config.ai_model,
            )
            if not result:
                return
            last_id = int(rows[-1].get("id") or 0)
            self.store.set_notes(
                chat_id, mem_tid,
                notes=merge_notes(str(state.get("notes") or ""), result.get("notes", "")),
                style=result.get("style", "") or str(state.get("style") or ""),
                folded_upto=last_id,
            )
            for name, note in (result.get("people") or {}).items():
                self.store.set_person_note(chat_id, name, note)
            logger.info("Folded %s turns into notes for %s/%s", len(rows), chat_id, mem_tid)
        except Exception as e:
            logger.debug("notes fold worker failed: %s", e)

    def _load_sticker_packs(self) -> None:
        """Index sticker packs by emoji so `[sticker:😠]` returns a matching sticker."""
        for name in self.config.sticker_sets:
            try:
                res = self.api.call("getStickerSet", {"name": name})
                added = self.media_bank.load_pack((res or {}).get("stickers") or [])
                logger.info("Sticker pack %s: %s stickers indexed", name, added)
            except Exception as e:
                logger.debug("Could not load sticker pack %s: %s", name, e)

    def is_user_admin(self, chat_id: int, user_id: int) -> bool:
        if user_id in self.config.admin_tg_ids:
            return True
        # Telegram anonymous admin GroupAnonymousBot (1087968824)
        if user_id == 1087968824:
            return True

        now = time.time()
        with self._lock:
            cached = self._admin_cache.get(chat_id)
            if cached and (now - cached[0] < 120):
                return user_id in cached[1]

        try:
            admins = self.api.get_chat_administrators(chat_id)
            admin_ids: Set[int] = set(self.config.admin_tg_ids)
            for item in admins or []:
                u = item.get("user") or {}
                uid = u.get("id")
                if uid:
                    admin_ids.add(int(uid))
            with self._lock:
                self._admin_cache[chat_id] = (now, admin_ids)
            return user_id in admin_ids
        except Exception as e:
            logger.debug("getChatAdministrators failed for %s: %s", chat_id, e)
            return False

    def _forget_reply(self, chat_key: str, text: str) -> None:
        """Откатить анти-повтор, если ответ так и не дошёл до чата."""
        try:
            from core.persona import forget_reply
            forget_reply(chat_key, text)
        except Exception:
            pass

    def _sweep_caches(self, now: Optional[float] = None) -> None:
        """Чистим накопившиеся тайминги: диалоги, медиа, мьюты, спам-события."""
        now = now or time.time()
        with self._lock:
            for key, (_, ts) in list(self._active_dialogues.items()):
                if now - ts > 600:
                    self._active_dialogues.pop(key, None)
            for key, (_kind, ts, _prompt) in list(self._last_media.items()):
                if now - ts > 3600:
                    self._last_media.pop(key, None)
            for key, until in list(self._muted_until.items()):
                if until < now:
                    self._muted_until.pop(key, None)
            for key, stamps in list(self._user_spam_events.items()):
                fresh = [t for t in stamps if now - t < 3600]
                if fresh:
                    self._user_spam_events[key] = fresh
                else:
                    self._user_spam_events.pop(key, None)
        # Внутренние кэши persona (повторы, мотивы, лимитер) тоже чистим
        try:
            from core import persona as persona_mod
            sweep = getattr(persona_mod, "sweep_state", None)
            if callable(sweep):
                sweep()
        except Exception:
            pass

    def _is_chat_admin(self, chat_id: int, user_id: int) -> bool:
        """Админ именно этого чата, а не просто глобальный админ из конфига."""
        try:
            member = self.api.get_chat_member(chat_id, user_id) or {}
            return str(member.get("status") or "") in ("creator", "administrator")
        except Exception:
            return False

    def _is_owner(self, user_id: int, username: str = "") -> bool:
        """Владелец бота: id из owner_tg_ids или совпавший owner_username."""
        if user_id in self.config.owner_tg_ids:
            return True
        configured = (self.config.owner_username or "").strip().lstrip("@").lower()
        candidate = (username or "").strip().lstrip("@").lower()
        return bool(configured and candidate and candidate == configured)

    def _discover_topic_metadata(self, message: Dict[str, Any]) -> Tuple[Optional[int], Optional[str]]:
        """Extract (thread_id, topic_name) from a forum message if available."""
        is_topic = bool(message.get("is_topic_message"))
        thread_id = message.get("message_thread_id")

        if thread_id is None and not is_topic:
            return None, None

        tid = int(thread_id) if thread_id is not None else 1
        topic_name: Optional[str] = None

        # Check service events or reply_to_message.forum_topic_created
        ft_created = message.get("forum_topic_created")
        if isinstance(ft_created, dict) and ft_created.get("name"):
            topic_name = str(ft_created["name"]).strip()

        ft_edited = message.get("forum_topic_edited")
        if isinstance(ft_edited, dict) and ft_edited.get("name"):
            topic_name = str(ft_edited["name"]).strip()

        reply = message.get("reply_to_message")
        if isinstance(reply, dict) and not topic_name:
            r_ftc = reply.get("forum_topic_created")
            if isinstance(r_ftc, dict) and r_ftc.get("name"):
                topic_name = str(r_ftc["name"]).strip()
            r_fte = reply.get("forum_topic_edited")
            if isinstance(r_fte, dict) and r_fte.get("name"):
                topic_name = str(r_fte["name"]).strip()

        return tid, topic_name

    def handle_update(self, update: Dict[str, Any]) -> None:
        if "callback_query" in update:
            self.handle_callback_query(update["callback_query"])
            return

        message = update.get("message")
        if not message:
            return

        self.handle_message(message)

    def handle_message(self, message: Dict[str, Any]) -> None:
        chat = message.get("chat") or {}
        chat_id = int(chat.get("id", 0))
        chat_type = str(chat.get("type") or "")
        chat_title = str(chat.get("title") or "")
        chat_username = chat.get("username")

        user = message.get("from") or {}
        user_id = int(user.get("id", 0))
        is_bot_sender = bool(user.get("is_bot"))

        # Ignore messages sent by this bot itself
        if self.bot_id and user_id == self.bot_id:
            return

        self._msg_seq += 1
        if self._msg_seq % 200 == 0:
            self._sweep_caches()

        # Private chat with the bot
        if chat_type == "private":
            self._handle_private_message(message)
            return

        # Ensure group/supergroup is registered
        chat_cfg = self.store.ensure_chat(chat_id, title=chat_title)
        tid, discovered_name = self._discover_topic_metadata(message)

        # Register or update topic metadata if inside a topic
        if tid is not None:
            existing_topic = self.store.get_topic(chat_id, tid)
            if existing_topic is None:
                auto_mode = classify_topic_name(discovered_name or "")
                # Ifauto_guard is off on the chat, default new topics to 'off'
                if not chat_cfg.get("auto_guard", 1):
                    auto_mode = "off"
                self.store.upsert_topic(
                    chat_id,
                    tid,
                    topic_name=discovered_name or f"Ветка #{tid}",
                    mode=auto_mode,
                    is_manual=False,
                )
            elif discovered_name and discovered_name != existing_topic.get("topic_name"):
                new_mode = None
                if not existing_topic.get("is_manual"):
                    new_mode = classify_topic_name(discovered_name)
                self.store.upsert_topic(
                    chat_id,
                    tid,
                    topic_name=discovered_name,
                    mode=new_mode,
                )

        # Ignore pure service messages (forum_topic_created, pinned_message, new_chat_members, etc.)
        if any(
            k in message
            for k in (
                "forum_topic_created",
                "forum_topic_edited",
                "forum_topic_closed",
                "forum_topic_reopened",
                "general_forum_topic_hidden",
                "general_forum_topic_unhidden",
                "new_chat_members",
                "left_chat_member",
                "pinned_message",
            )
        ):
            return

        text = str(message.get("text") or message.get("caption") or "").strip()

        # Check for bot commands
        if text.startswith("/"):
            cmd_handled = self._handle_group_command(message, chat_cfg, tid)
            if cmd_handled:
                return

        # Ignore other bots unless needed
        if is_bot_sender:
            return

        # Learn any sticker or animation posted in the chat
        stk = message.get("sticker")
        if isinstance(stk, dict):
            self.media_bank.learn_sticker(str(stk.get("file_id") or ""), stk.get("emoji"))
        anim = message.get("animation")
        if isinstance(anim, dict):
            self.media_bank.learn_animation(str(anim.get("file_id") or ""))

        # Check if user is currently auto-muted
        mute_key = (chat_id, user_id)
        now = time.time()
        if now < self._muted_until.get(mute_key, 0.0):
            try:
                self.api.delete_message(chat_id, int(message["message_id"]))
            except Exception:
                pass
            return

        # Только по chat_id: название чата может поменять кто угодно,
        # и модерация бы молча отключилась (или включилась в чужом чате).
        is_private_chat = bool(chat_id == self.config.private_chat_id)

        # In external/public groups: Kiryxa is 100% a chat bot!
        # NEVER moderate, delete, or move messages to General in outside chats!
        if not is_private_chat:
            self._dispatch_persona(
                message, chat_id, tid, chat_cfg=chat_cfg, is_private_chat=False
            )
            return

        # Determine if this message is in the General topic
        general_tid = chat_cfg.get("general_thread_id")
        if tid is None or tid == 1 or (general_tid is not None and tid == int(general_tid)):
            # In General topic: if a key, trial, or abuse drop appears, mirror to Abuse & Keys topic!
            if is_private_chat:
                self._submit(self._maybe_mirror_drop_from_general, message, chat_cfg, chat_username)
            # In General topic: respond in-character if replied to or mentioned
            self._dispatch_persona(
                message, chat_id, tid, chat_cfg=chat_cfg, is_private_chat=True
            )
            return

        topic_cfg = self.store.get_topic(chat_id, tid)
        mode = (topic_cfg or {}).get("mode", "smart")
        if mode in ("off", "general"):
            self._dispatch_persona(message, chat_id, tid, chat_cfg=chat_cfg, is_private_chat=True)
            return

        # Check admin / sender_chat (channel/anonymous group admin)
        is_sender_chat_admin = bool(message.get("sender_chat"))
        is_admin = is_sender_chat_admin or self.is_user_admin(chat_id, user_id)
        is_wl = self.store.is_whitelisted(chat_id, user_id)
        admin_bypass = bool(chat_cfg.get("admin_bypass", self.config.default_admin_bypass))

        verdict = classify_message(
            message,
            mode=mode,
            is_admin=is_admin,
            is_whitelisted=is_wl,
            admin_bypass=admin_bypass,
            topic_name=str((topic_cfg or {}).get("topic_name") or f"Ветка #{tid}"),
            ai_enabled=bool(self.config.ai_enabled),
            ai_api_url=self.config.ai_api_url,
            ai_api_key=self.config.ai_api_key,
            ai_model=self.config.ai_model,
        )

        if verdict.keep:
            self.store.record_kept(chat_id, tid)
            return

        # Check for spam/flood in protected topic (non-admins)
        if not is_admin:
            msg_id = int(message["message_id"])
            is_spam, mute_sec, dur_label = self._check_and_register_spam(chat_id, user_id, msg_id)
            if is_spam:
                self._handle_user_spam_punishment(
                    message=message,
                    chat_cfg=chat_cfg,
                    topic_cfg=topic_cfg or {"thread_id": tid, "topic_name": f"Ветка #{tid}", "mode": mode},
                    mute_sec=mute_sec,
                    dur_label=dur_label,
                    chat_username=chat_username,
                )
                return

        # CSAM и вредонос из ветки НЕЛЬЗЯ пересылать в General: раньше такой текст
        # уезжал туда с подколом и оставался висеть в чате. Удаляем и не зеркалим.
        screen_text = str(message.get("text") or message.get("caption") or "")
        level = screen_level(screen_text) if screen_text else None
        if level in ("csam", "harm"):
            try:
                self.api.delete_message(chat_id, int(message["message_id"]))
            except Exception:
                pass
            logger.warning(
                "Blocked %s request in chat %s thread %s; not mirrored to General",
                level, chat_id, tid,
            )
            if level == "harm":
                try:
                    self.api.send_message(
                        chat_id=chat_id,
                        message_thread_id=tid,
                        text="<i>такое в ветке не оставляю</i>",
                        parse_mode="HTML",
                    )
                except Exception:
                    pass
            return

        # Message is off-topic / chatter -> Move to General chat and delete original
        self._move_message_to_general(
            message=message,
            chat_cfg=chat_cfg,
            topic_cfg=topic_cfg or {"thread_id": tid, "topic_name": f"Ветка #{tid}", "mode": mode},
            reason=verdict.reason,
            chat_username=chat_username,
        )

    def _check_and_register_spam(
        self, chat_id: int, user_id: int, msg_id: int
    ) -> Tuple[bool, int, str]:
        now = time.time()
        key = (chat_id, user_id)
        events = self._user_spam_events.setdefault(key, [])
        events.append((now, msg_id))
        # Keep only events within last 12 seconds
        self._user_spam_events[key] = [(t, m) for t, m in events if now - t < 12.0]
        recent = self._user_spam_events[key]

        # Threshold: 3 or more off-topic messages within 12 seconds is SPAM
        if len(recent) >= 3:
            count = self._user_mute_counts.get(key, 0) + 1
            self._user_mute_counts[key] = count
            # Raid / extreme flood: 6+ messages in 12s -> 30 days
            if len(recent) >= 6:
                return True, 86400 * 30, "30 дней (за спам/рейд)"
            if count == 1:
                return True, 600, "10 минут"
            elif count == 2:
                return True, 3600, "1 час"
            elif count == 3:
                return True, 86400, "24 часа"
            else:
                return True, 86400 * 30, "30 дней"

        return False, 0, ""

    def _handle_user_spam_punishment(
        self,
        message: Dict[str, Any],
        chat_cfg: Dict[str, Any],
        topic_cfg: Dict[str, Any],
        mute_sec: int,
        dur_label: str,
        chat_username: Optional[str] = None,
    ) -> None:
        chat_id = int(chat_cfg["chat_id"])
        user = message.get("from") or {}
        user_id = int(user.get("id", 0))
        now = time.time()
        until_date = int(now + mute_sec)
        self._muted_until[(chat_id, user_id)] = until_date

        # 1. Apply Telegram restriction (mute)
        try:
            self.api.restrict_chat_member(chat_id=chat_id, user_id=user_id, until_date=until_date)
            logger.info("Muted user %s in %s for %s sec (%s)", user_id, chat_id, mute_sec, dur_label)
        except Exception as e:
            logger.warning("Could not mute user %s: %s", user_id, e)

        # 2. Batch-delete all recent spam messages from this user in that topic
        recent_mids = [m for _, m in self._user_spam_events.get((chat_id, user_id), [])]
        self._user_spam_events[(chat_id, user_id)] = []
        if recent_mids:
            try:
                self.api.delete_messages(chat_id, recent_mids)
            except Exception:
                for mid in recent_mids:
                    try:
                        self.api.delete_message(chat_id, mid)
                    except Exception:
                        pass
        try:
            self.api.delete_message(chat_id, int(message["message_id"]))
        except Exception:
            pass

        # 3. Post ONE clean announcement in General (NOT copying the spam!)
        target_general_tid = (
            int(chat_cfg["general_thread_id"])
            if chat_cfg.get("general_thread_id") is not None
            else None
        )
        from_tid = int(topic_cfg["thread_id"])
        topic_name = str(topic_cfg.get("topic_name") or f"Ветка #{from_tid}")
        topic_url = build_topic_link(chat_id, from_tid, chat_username=chat_username)
        escaped_topic = html.escape(topic_name)
        topic_ref = f'<a href="{topic_url}">#{escaped_topic}</a>' if topic_url else f"<b>#{escaped_topic}</b>"
        user_mention = format_user_mention(user)

        mute_text = (
            f"🔇 {user_mention} отправлен в мут на <b>{dur_label}</b> за спам/флуд в ветке {topic_ref}.\n\n"
            f"💬 <i>Хватит спамить стикерами и мусором, гой. Посиди подумай 🧂</i>"
        )
        try:
            self._send_to_general(chat_id, target_general_tid, mute_text)
        except Exception as e:
            logger.warning("Failed to send mute notice to general: %s", e)

    def _restart_muse_if_needed(self, force: bool = False) -> None:
        """Muse иногда зависает на одном аккаунте: перезапуск моста лечит."""
        now = time.time()
        min_gap = 120.0 if force else 600.0
        if now - self._last_muse_restart < min_gap:
            return
        self._last_muse_restart = now
        try:
            subprocess.run(
                ["sudo", "systemctl", "restart", "muse2api.service"],
                timeout=40, check=False,
            )
            logger.warning("Restarted muse2api after media generation failure")
        except Exception as e:
            logger.warning("Could not restart muse2api: %s", e)

    def _persona_lock(self, key: Tuple[int, Optional[int]]) -> threading.Lock:
        with self._persona_locks_guard:
            lock = self._persona_locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._persona_locks[key] = lock
            return lock

    def _submit(self, fn, *args) -> None:
        """Выполнить что-то медленное в пуле, не блокируя polling."""
        def worker() -> None:
            try:
                fn(*args)
            except Exception as e:
                logger.warning("Background job %s failed: %s", getattr(fn, "__name__", fn), e)

        try:
            self._persona_pool.submit(worker)
        except Exception as e:
            logger.warning("Background pool rejected job: %s", e)

    def _dispatch_persona(
        self,
        message: Dict[str, Any],
        chat_id: int,
        tid: Optional[int],
        chat_cfg: Optional[Dict[str, Any]],
        is_private_chat: bool,
    ) -> None:
        """Ответ персоны — в фоне, с сохранением порядка внутри ветки."""
        if getattr(self, "_inline_persona", False):
            # Тестовый режим: без потоков, иначе assertions гоняются с воркерами
            self._maybe_reply_in_general(
                message, chat_id, tid, chat_cfg=chat_cfg, is_private_chat=is_private_chat
            )
            return
        lock = self._persona_lock((chat_id, tid))

        def worker() -> None:
            with lock:
                try:
                    self._maybe_reply_in_general(
                        message, chat_id, tid, chat_cfg=chat_cfg, is_private_chat=is_private_chat
                    )
                except Exception as e:
                    logger.warning("Persona worker failed: %s", e)

        try:
            self._persona_pool.submit(worker)
        except Exception as e:
            logger.warning("Persona pool rejected job, running inline: %s", e)
            worker()

    @staticmethod
    def _media_from_message(msg: Dict[str, Any]) -> Tuple[Optional[str], str, str, Optional[str]]:
        """Достать из сообщения медиа для Vision.

        Возвращает (file_id, mime, label, fallback_id). Для видео и гифок отдаём
        сам файл — Gemini умеет смотреть видео целиком; thumbnail остаётся
        запасным вариантом, если файл слишком большой или не скачался.
        """
        photos = msg.get("photo")
        if isinstance(photos, list) and photos:
            return photos[-1].get("file_id"), "image/jpeg", "фото", None
        anim = msg.get("animation")
        if isinstance(anim, dict):
            thumb = anim.get("thumbnail")
            thumb_id = thumb.get("file_id") if isinstance(thumb, dict) else None
            fid = anim.get("file_id") or thumb_id
            mime = "video/mp4" if anim.get("file_id") else "image/jpeg"
            return fid, mime, "гифку", thumb_id
        video = msg.get("video") or msg.get("video_note")
        if isinstance(video, dict):
            thumb = video.get("thumbnail")
            thumb_id = thumb.get("file_id") if isinstance(thumb, dict) else None
            fid = video.get("file_id") or thumb_id
            mime = str(video.get("mime_type") or "video/mp4").lower()
            return fid, mime, "видео", thumb_id
        doc = msg.get("document")
        if isinstance(doc, dict):
            mime = str(doc.get("mime_type") or "").lower()
            if mime.startswith("image/"):
                return doc.get("file_id"), mime, "картинку", None
            if mime.startswith("video/"):
                thumb = doc.get("thumbnail")
                thumb_id = thumb.get("file_id") if isinstance(thumb, dict) else None
                return doc.get("file_id"), mime, "видео", thumb_id
        return None, "image/jpeg", "", None

    def _maybe_reply_in_general(
        self,
        message: Dict[str, Any],
        chat_id: int,
        tid: Optional[int],
        chat_cfg: Optional[Dict[str, Any]] = None,
        is_private_chat: Optional[bool] = None,
    ) -> None:
        """Reply in-character as Kiryxa in General when replied to, called by name, or in active dialogue."""
        if not (self.config.ai_enabled and self.config.ai_api_key):
            return
        text = str(message.get("text") or message.get("caption") or "").strip()

        user = message.get("from") or {}
        user_id = int(user.get("id", 0))
        username = str(user.get("username") or "")
        # Владелец — конкретный человек из конфига, а не любой админ.
        is_drake = self._is_owner(user_id, username)
        author_name = (
            self.config.owner_name
            if is_drake
            else str(user.get("first_name") or username or "гой").strip()
        )

        if is_private_chat is None:
            chat_title = str((chat_cfg or {}).get("title") or (message.get("chat") or {}).get("title") or "")
            # Приватку определяем по chat_id, а не по названию: название может
            # поменять кто угодно, и модерация бы просто отключилась.
            is_private_chat = bool(
                chat_id == self.config.private_chat_id
                or (
                    not self.config.private_chat_id
                    and ("ключики от ваших сердец" in chat_title.lower() or "приватка" in chat_title.lower())
                )
            )

        # General in forums can arrive as None, thread_id=1 or the configured general id.
        # Keep one memory key for all of them, otherwise the bot loses its own conversation.
        mem_tid = tid
        if is_private_chat:
            gtid = (chat_cfg or {}).get("general_thread_id")
            gtid = int(gtid) if gtid is not None else None
            if tid is None or tid == 1 or (gtid is not None and tid == gtid):
                mem_tid = gtid

        reply_msg = message.get("reply_to_message") or {}
        reply_from = reply_msg.get("from") or {}
        is_reply_to_bot = bool(
            self.bot_id and int(reply_from.get("id", 0)) == self.bot_id
        )
        called_by_name = is_addressed(text, self.bot_username)

        now = time.time()
        diag_key = (chat_id, mem_tid)
        # Активный диалог: тот же человек пишет вдогонку в течение 20 секунд
        last_dialogue = self._active_dialogues.get(diag_key)
        is_active_dialogue = bool(
            last_dialogue
            and (now - last_dialogue[1] < 10.0)
            and (last_dialogue[0] == user_id)
        )
        called_or_reply = bool(is_reply_to_bot or called_by_name or is_active_dialogue)

        # Участие: во внешних чатах Кирюха может сам влезть в интересную тему,
        # но не в быстрый чат и не в чужой диалог двух людей.
        times = self._recent_message_times.setdefault(diag_key, deque(maxlen=60))
        times.append(now)
        speakers = self._recent_speakers.setdefault(diag_key, deque(maxlen=8))
        part = should_join(
            text,
            addressed=called_or_reply,
            from_user=user_id,
            last_speaker=self._last_speaker.get(diag_key),
            recent_speakers=speakers,
            replies_to_other=bool(reply_msg and not is_reply_to_bot),
            recent_messages=times,
            now=now,
        ) if text else None
        allow_unprompted = not is_private_chat
        is_addressed_or_reply = called_or_reply or bool(
            text and allow_unprompted and part and part.reply
        )
        if text:
            self._last_speaker[diag_key] = user_id
            speakers.append(user_id)
        msg_id = int(message.get("message_id", 0)) or None

        # В сообщении может быть фото, гифка, видео или документ-картинка
        file_id_to_see, media_mime, media_label, media_fallback = self._media_from_message(message)

        # Картинка в replied-сообщении. Если у текущего сообщения есть текст —
        # это вопрос по картинке: отвечаем текстом, а не комментируем её.
        reply_file_id, reply_media_mime, _reply_label, _reply_fallback = self._media_from_message(reply_msg)
        # Без текста (просто ответ гифкой/фото/видео) — можно комментировать кадр
        if not file_id_to_see and reply_file_id and not text:
            file_id_to_see, media_mime = reply_file_id, reply_media_mime

        # If image/GIF is present and vision enabled:
        if file_id_to_see and self.vision:
            # Входящее фото пишем в память сразу: раньше запись появлялась только
            # если Vision успешно отправил комментарий, и картинка терялась из истории.
            self.store.add_memory(
                chat_id, mem_tid, "user", author_name,
                f"[прислал {media_label or 'картинку'}: {text or 'без текста'}]",
            )
            threading.Thread(
                target=self._handle_vision_reaction_and_reply,
                args=(
                    message,
                    chat_id,
                    tid,
                    mem_tid,
                    msg_id,
                    file_id_to_see,
                    text,
                    media_mime,
                    media_fallback,
                    author_name,
                    diag_key,
                    user_id,
                    now,
                    is_addressed_or_reply,
                    is_drake,
                ),
                daemon=True,
            ).start()
            return

        if not text:
            # Медиа без подписи и без кадра (видео без thumbnail) — хотя бы в память
            if media_label:
                self.store.add_memory(
                    chat_id, mem_tid, "user", author_name, f"[прислал {media_label}]"
                )
            return

        # Extract reply quote context if message is a reply to another post
        reply_context = ""
        if isinstance(reply_msg, dict) and not reply_msg.get("forum_topic_created"):
            r_text = str(reply_msg.get("text") or reply_msg.get("caption") or "").strip()
            r_user = reply_msg.get("from") or {}
            r_uid = int(r_user.get("id", 0))
            r_uname = str(r_user.get("username") or "")
            if r_uid == self.bot_id:
                r_author = "Kiryxa"
            elif self._is_owner(r_uid, r_uname):
                r_author = self.config.owner_name
            else:
                r_author = str(r_user.get("first_name") or r_uname or "собеседник").strip()
            if r_text:
                reply_context = f'[В ответ на {r_author}: "{r_text[:400]}"]'

        # Fetch previous conversation turns from SQLite (before current user message)
        hist = self.store.get_memory(chat_id, mem_tid, limit=50)

        # Record incoming message in chat memory so Kiryxa never misses context
        mem_entry = f"{reply_context} {text}".strip() if reply_context else text
        if media_label:
            # Видео без кадра (или выключенный Vision): хотя бы фиксируем факт медиа
            mem_entry = f"[прислал {media_label}] {mem_entry}".strip()
        self.store.add_memory(chat_id, mem_tid, "user", author_name, mem_entry)

        if not is_addressed_or_reply:
            # Смех в чате — повод ответить одной гифкой, без текста и без модели.
            # Кулдаун на чат не даёт превратить это в спам.
            if text and allow_unprompted:
                gif_only = self.media_bank.wants_reaction_gif(
                    text, chat_key=f"laugh:{chat_id}:{mem_tid}"
                )
                if gif_only:
                    try:
                        self.api.send_animation(
                            chat_id=chat_id,
                            animation=gif_only,
                            message_thread_id=tid,
                            reply_to_message_id=msg_id,
                        )
                    except Exception as e:
                        logger.debug("Failed to send laugh GIF: %s", e)
            return

        # Склейка: если человек пишет мысль в три сообщения подряд, отвечаем один раз.
        # Собираем ДО лимитера, иначе вторая короткая реплика не доходила до coalescer.
        # Адресованные сообщения отвечаются сразу — задержка тут только раздражает.
        if not called_or_reply and allow_unprompted:
            merged = self._coalesce(
                chat_id, mem_tid, text, user_id, author_name, tid, msg_id, reply_context,
                is_private_chat=is_private_chat,
            )
            if merged is None:
                return
            text = merged

        if not self._chat_limiter.allow(user_id, ignore_user_limit=called_or_reply):
            # Дёрнули по имени/реплаем, а лимит съел сообщение — это выглядит
            # как «бот иногда вообще не отвечает». Логируем, чтобы было видно.
            logger.info("Rate limit skipped reply to %s in %s (addressed=%s)", user_id, chat_id, called_or_reply)
            return

        # Check for media generation request via Muse (photo, video, voice/ГС)
        if self.muse:
            media_key = f"{chat_id}:{mem_tid}"
            last_kind = None
            last_prompt = ""
            last = self._last_media.get(media_key)
            if last and now - last[1] < 900:
                last_kind, last_prompt = last[0], last[2]
            media_kind, media_prompt = detect_media_intent(text)
            if not media_kind and last_kind and (called_or_reply or is_drake):
                # Правка прошлой генерации: «сделай нормального гоя а не утку».
                # Старый промпт сохраняем, иначе теряется сцена из первого запроса.
                media_kind, media_prompt = detect_media_intent(text, last_kind=last_kind)
                if media_kind and media_prompt and last_prompt:
                    media_prompt = f"{last_prompt}; {media_prompt}"
            if media_kind and media_prompt:
                self._last_media[media_key] = (media_kind, now, media_prompt)
                # Тот же фильтр, что и для чата: генерация не должна обходить проверки
                refusal = screen_request(text, is_drake=is_drake)
                if refusal:
                    try:
                        self.api.send_message(
                            chat_id=chat_id,
                            message_thread_id=tid,
                            text=html.escape(refusal),
                            parse_mode="HTML",
                            reply_to_message_id=msg_id,
                        )
                    except Exception as e:
                        logger.warning("Failed to send media refusal: %s", e)
                    return
                # Run in detached background thread so polling NEVER freezes!
                threading.Thread(
                    target=self._handle_muse_media_generation,
                    args=(chat_id, tid, msg_id, media_kind, media_prompt),
                    daemon=True,
                ).start()
                return

        prompt_with_quote = f"{reply_context}\n{text}".strip() if reply_context else text
        # Ответ на картинку с вопросом: даём модели словесное описание кадра,
        # чтобы она отвечала на вопрос, а не комментировала пикчу вслепую.
        if reply_file_id and self.vision:
            try:
                img_bytes = self.api.get_file_bytes(reply_file_id)
                if img_bytes:
                    ctx_timeout = 45.0 if str(reply_media_mime).startswith("video") else 20.0
                    # Именно описание, а не шутка: модель должна отвечать по факту
                    desc = self.vision.describe(
                        img_bytes, mime=reply_media_mime, timeout=ctx_timeout
                    )
                    if desc:
                        prompt_with_quote = f"[что на картинке: {desc}]\n{prompt_with_quote}"
            except Exception as e:
                logger.debug("Vision context for reply failed: %s", e)

        reply_text = generate_reply(
            text=prompt_with_quote,
            author_name=author_name,
            api_url=self.config.ai_api_url,
            api_key=self.config.ai_api_key,
            model=self.config.ai_model,
            history=hist,
            exa_api_key=self.config.exa_api_key,
            is_private_chat=is_private_chat,
            is_drake=is_drake,
            chat_key=f"{chat_id}:{mem_tid}",
            fallback=(
                (self.config.fallback_api_url, self.config.fallback_api_key, self.config.fallback_model)
                if self.config.fallback_api_url and self.config.fallback_api_key and self.config.fallback_model
                else None
            ),
            memory_block=self._memory_block(chat_id, mem_tid, author_name),
            author_own_text=self._author_own_text(chat_id, mem_tid, author_name),
            owner_name=self.config.owner_name,
            owner_username=self.config.owner_username,
        )
        if not reply_text and called_or_reply:
            # Обратились по имени, а модель промолчала — пробуем ещё раз,
            # но с пустым контекстом: иногда именно история валит генерацию.
            logger.info("Empty reply for addressed msg %s, retrying without history", msg_id)
            try:
                reply_text = generate_reply(
                    text=text,
                    author_name=author_name,
                    api_url=self.config.ai_api_url,
                    api_key=self.config.ai_api_key,
                    model=self.config.ai_model,
                    history=[],
                    exa_api_key="",
                    is_private_chat=is_private_chat,
                    is_drake=is_drake,
                    chat_key=f"{chat_id}:{mem_tid}:retry",
                    fallback=(
                        (self.config.fallback_api_url, self.config.fallback_api_key, self.config.fallback_model)
                        if self.config.fallback_api_url and self.config.fallback_api_key and self.config.fallback_model
                        else None
                    ),
                    owner_name=self.config.owner_name,
                    owner_username=self.config.owner_username,
                )
            except Exception as e:
                logger.warning("Retry reply failed: %s", e)
            if not reply_text and called_or_reply:
                # Обратились лично — молчать нельзя, даже если обе модели упали.
                # Раньше живая отмазка была только для владельца, остальные получали тишину.
                reply_text = random.choice(SILENT_FALLBACKS)

        if reply_text:
            # Check if reply contains sticker/gif tags or if user asked for sticker/gif
            clean_text, tag_stk, tag_anim = self.media_bank.parse_reply(
                reply_text, chat_key=f"{chat_id}:{mem_tid}"
            )
            wants_stk, wants_gif = self.media_bank.check_user_ask(text)
            # Смех/восторг в чате — повод кинуть гифку, если давно не кидали
            laugh_gif = self.media_bank.wants_reaction_gif(text, chat_key=f"laugh:{chat_id}:{mem_tid}")
            # Модель часто не ставит тег — тогда выбираем реакцию сами
            if not tag_stk and not tag_anim and not laugh_gif:
                auto_stk, auto_anim = self.media_bank.maybe_reaction(
                    text, clean_text, chat_key=f"auto:{chat_id}:{mem_tid}"
                )
                tag_stk = tag_stk or auto_stk
                tag_anim = tag_anim or auto_anim
            msg_id = int(message.get("message_id", 0)) or None

            sent_media = False
            delivered = False
            # 1. Send Animation / GIF if tag or user requested
            if tag_anim or wants_gif or laugh_gif:
                anim_fid = tag_anim or laugh_gif or self.media_bank.get_animation(
                    chat_key=f"{chat_id}:{mem_tid}"
                )
                if anim_fid:
                    try:
                        self.api.send_animation(
                            chat_id=chat_id,
                            animation=anim_fid,
                            message_thread_id=tid,
                            caption=html.escape(clean_text) if clean_text else None,
                            parse_mode="HTML" if clean_text else None,
                            reply_to_message_id=msg_id,
                        )
                        sent_media = True
                        delivered = True
                    except Exception as e:
                        logger.debug("Failed to send animation: %s", e)

            # 2. Send Sticker if tag or user requested
            if not sent_media and (tag_stk or wants_stk):
                stk_fid = tag_stk or self.media_bank.get_sticker(chat_key=f"{chat_id}:{mem_tid}")
                if stk_fid:
                    try:
                        self.api.send_sticker(
                            chat_id=chat_id,
                            sticker=stk_fid,
                            message_thread_id=tid,
                            reply_to_message_id=msg_id,
                        )
                        sent_media = True
                        delivered = True
                    except Exception as e:
                        logger.debug("Failed to send sticker: %s", e)

            # 2.5. Голосовой режим: вместо текста отвечаем голосовым
            if (
                clean_text
                and not sent_media
                and len(clean_text) <= 600
                and self._voice_mode_on(chat_id)
            ):
                audio = self._speak(clean_text)
                if audio:
                    try:
                        self.api.send_voice(
                            chat_id=chat_id,
                            voice=audio,
                            message_thread_id=tid,
                            reply_to_message_id=msg_id,
                        )
                        sent_media = True
                        delivered = True
                    except Exception as e:
                        logger.debug("Failed to send voice reply: %s", e)

            # 3. If text remains and wasn't sent as animation caption, send message
            if clean_text and (not sent_media or tag_stk or wants_stk):
                try:
                    self.api.send_message(
                        chat_id=chat_id,
                        message_thread_id=tid,
                        text=html.escape(clean_text),
                        parse_mode="HTML",
                        reply_to_message_id=msg_id if not sent_media else None,
                    )
                    delivered = True
                except Exception as e:
                    logger.warning("Failed to send reply to %s in %s: %s", msg_id, chat_id, e)

            # Store in memory only what actually reached the chat
            if delivered:
                mem_text = (clean_text or reply_text).strip()
                if not clean_text:
                    mem_text = "[отправил гифку]" if tag_anim else "[отправил стикер]"
                elif tag_anim:
                    mem_text = f"[гифка] {mem_text}"
                elif tag_stk:
                    mem_text = f"[стикер] {mem_text}"
                self.store.add_memory(chat_id, mem_tid, "assistant", "Kiryxa", mem_text)
                self._active_dialogues[diag_key] = (user_id, now)
                self._maybe_fold_notes(chat_id, mem_tid)
            else:
                logger.warning("Reply for msg %s in %s was not delivered, skipping memory write", msg_id, chat_id)
                self._forget_reply(f"{chat_id}:{mem_tid}", reply_text)

    def _coalesce(
        self,
        chat_id: int,
        mem_tid: Optional[int],
        text: str,
        user_id: int,
        author_name: str,
        tid: Optional[int],
        msg_id: Optional[int],
        reply_context: str,
        is_private_chat: bool = False,
    ) -> Optional[str]:
        """Merge a burst of unaddressed messages into one turn.

        Returns the merged text when it is time to answer, or None while we are
        still collecting parts. The waiting happens in a daemon thread so polling
        never blocks; a late part simply moves the flush deadline.
        """
        key = (chat_id, mem_tid)
        with self._lock:
            coalescer = self._coalescers.get(key)
            if coalescer is None:
                coalescer = Coalescer()
                self._coalescers[key] = coalescer
            # Группируем по автору: реплики двух людей не должны склеиваться
            coalescer.add(text, user_id=user_id, now=time.time())
            parts = coalescer.parts_count(key, user_id)
            needs_wait = parts < coalescer.policy.max_parts

        if needs_wait:
            threading.Thread(
                target=self._flush_after_quiet,
                args=(key, chat_id, mem_tid, user_id, author_name, tid, msg_id, reply_context, is_private_chat),
                daemon=True,
            ).start()
            return None

        with self._lock:
            merged = self._coalescers[key].flush(key, user_id)
        return merged

    def _flush_after_quiet(
        self,
        key: Tuple[int, Optional[int]],
        chat_id: int,
        mem_tid: Optional[int],
        user_id: int,
        author_name: str,
        tid: Optional[int],
        msg_id: Optional[int],
        reply_context: str,
        is_private_chat: bool = False,
    ) -> None:
        """Wait out the quiet window, then answer the merged turn if nobody added more."""
        # Ждём именно тишину: если человек дописывает, окно сдвигается.
        deadline = time.time() + 12.0
        merged: Optional[str] = None
        while time.time() < deadline:
            time.sleep(0.6)
            with self._lock:
                coalescer = self._coalescers.get(key)
                if coalescer is None:
                    return
                if not coalescer.ready(key, user_id, time.time()):
                    continue
                merged = coalescer.flush(key, user_id)
            break
        if not merged:
            return
        # Лимитер применяем один раз — на готовый батч, а не на каждую часть
        if not self._chat_limiter.allow(user_id):
            logger.info("Rate limit skipped merged reply to %s in %s", user_id, chat_id)
            return
        self._reply_to_merged(
            chat_id, mem_tid, merged, author_name, tid, msg_id, reply_context, is_private_chat
        )

    def _reply_to_merged(
        self,
        chat_id: int,
        mem_tid: Optional[int],
        text: str,
        author_name: str,
        tid: Optional[int],
        msg_id: Optional[int],
        reply_context: str,
        is_private_chat: bool = False,
    ) -> None:
        """Answer a merged burst; mirrors the tail of `_maybe_reply_in_general`."""
        if not (self.config.ai_enabled and self.config.ai_api_key):
            return
        hist = self.store.get_memory(chat_id, mem_tid, limit=50)
        prompt_with_quote = f"{reply_context}\n{text}".strip() if reply_context else text
        try:
            reply_text = generate_reply(
                text=prompt_with_quote,
                author_name=author_name,
                api_url=self.config.ai_api_url,
                api_key=self.config.ai_api_key,
                model=self.config.ai_model,
                history=hist,
                exa_api_key=self.config.exa_api_key,
                is_private_chat=is_private_chat,
                is_drake=author_name == self.config.owner_name,
                chat_key=f"{chat_id}:{mem_tid}",
                fallback=(
                    (self.config.fallback_api_url, self.config.fallback_api_key, self.config.fallback_model)
                    if self.config.fallback_api_url and self.config.fallback_api_key and self.config.fallback_model
                    else None
                ),
                owner_name=self.config.owner_name,
                owner_username=self.config.owner_username,
            )
        except Exception as e:
            logger.warning("Coalesced reply failed: %s", e)
            return
        if not reply_text:
            return

        clean_text, tag_stk, tag_anim = self.media_bank.parse_reply(
            reply_text, chat_key=f"{chat_id}:{mem_tid}"
        )
        try:
            if tag_stk:
                self.api.send_sticker(
                    chat_id=chat_id, sticker=tag_stk,
                    message_thread_id=tid, reply_to_message_id=msg_id,
                )
            if tag_anim:
                self.api.send_animation(
                    chat_id=chat_id, animation=tag_anim,
                    message_thread_id=tid, reply_to_message_id=msg_id,
                )
            if clean_text:
                self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=tid,
                    text=html.escape(clean_text),
                    parse_mode="HTML",
                    reply_to_message_id=msg_id,
                )
                self.store.add_memory(chat_id, mem_tid, "assistant", "Kiryxa", clean_text)
        except Exception as e:
            logger.warning("Failed to send coalesced reply: %s", e)
            self._forget_reply(f"{chat_id}:{mem_tid}", reply_text)

    def _handle_muse_media_generation(
        self,
        chat_id: int,
        tid: Optional[int],
        msg_id: Optional[int],
        kind: str,
        prompt: str,
    ) -> None:
        if not self.muse:
            return
        if kind == "video" and not self.config.muse_video_enabled:
            try:
                self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=tid,
                    text="🎬 <i>Muse сейчас не отдаёт видео — их рендер виснет и блокирует даже фото. Могу сделать картинку.</i>",
                    parse_mode="HTML",
                    reply_to_message_id=msg_id,
                )
            except Exception:
                pass
            return
        if kind == "video":
            # После недавнего зависания не блокируем аккаунт снова сразу
            try:
                retry_after = float(self.store.get_setting("muse.video_retry_after", "0"))
            except Exception:
                retry_after = 0.0
            left = retry_after - time.time()
            if left > 0:
                try:
                    self.api.send_message(
                        chat_id=chat_id,
                        message_thread_id=tid,
                        text=f"🎬 <i>Muse недавно завис на видео — попробую снова через ~{int(left // 60) + 1} мин. Пока могу сделать фото.</i>",
                        parse_mode="HTML",
                        reply_to_message_id=msg_id,
                    )
                except Exception:
                    pass
                return

        if kind == "photo":
            placeholder = None
            try:
                placeholder = self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=tid,
                    text="🎨 <i>Рисую фото через Muse, это может занять до пары минут...</i>",
                    parse_mode="HTML",
                    reply_to_message_id=msg_id,
                )
            except Exception:
                pass
            photo_bytes = self.muse.generate_image(prompt)
            if photo_bytes:
                sent = False
                try:
                    self.api.send_photo(
                        chat_id=chat_id,
                        photo=photo_bytes,
                        caption=f"🎨 <i>{html.escape(prompt)}</i>\n\n🐾 <i>фурр~ держи артик!</i>",
                        message_thread_id=tid,
                        reply_to_message_id=msg_id,
                    )
                    sent = True
                    self.store.add_memory(
                        chat_id, tid, "assistant", "Kiryxa", f"[отправил фото: {prompt}]"
                    )
                except Exception as e:
                    logger.warning("Failed to send photo: %s", e)
                # Плейсхолдер убираем только после успешной отправки, иначе
                # пользователь не увидит ни фото, ни ошибку.
                if placeholder and sent:
                    try:
                        self.api.delete_message(chat_id, int(placeholder["message_id"]))
                    except Exception:
                        pass
                elif placeholder and not sent:
                    try:
                        self.api.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(placeholder["message_id"]),
                            text="😿 <i>фото сгенерилось, но не отправилось — попробуй ещё раз</i>",
                        )
                    except Exception:
                        pass
            else:
                if placeholder:
                    try:
                        self.api.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(placeholder["message_id"]),
                            text="😿 <i>Muse не отдал фото — попробуй ещё раз чуть позже</i>",
                        )
                    except Exception:
                        pass
                self._restart_muse_if_needed()

        elif kind == "video":
            placeholder = None
            try:
                placeholder = self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=tid,
                    text="🎬 <i>Генерирую видео через Muse, обычно 1-3 минуты — не уходи...</i>",
                    parse_mode="HTML",
                    reply_to_message_id=msg_id,
                )
            except Exception:
                pass
            video_bytes = self.muse.generate_video(prompt, timeout=180.0)
            if video_bytes:
                self.store.set_setting("muse.video_retry_after", "0")
                sent = False
                try:
                    self.api.send_video(
                        chat_id=chat_id,
                        video=video_bytes,
                        caption=f"🎬 <i>{html.escape(prompt)}</i>\n\n🐾 <i>фурр-фурр, клип готов!</i>",
                        message_thread_id=tid,
                        reply_to_message_id=msg_id,
                    )
                    sent = True
                    self.store.add_memory(
                        chat_id, tid, "assistant", "Kiryxa", f"[отправил видео: {prompt}]"
                    )
                except Exception as e:
                    logger.warning("Failed to send video: %s", e)
                if placeholder and sent:
                    try:
                        self.api.delete_message(chat_id, int(placeholder["message_id"]))
                    except Exception:
                        pass
                elif placeholder and not sent:
                    try:
                        self.api.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(placeholder["message_id"]),
                            text="😿 <i>видео готово, но не отправилось — попробуй ещё раз</i>",
                        )
                    except Exception:
                        pass
            else:
                if placeholder:
                    try:
                        self.api.edit_message_text(
                            chat_id=chat_id,
                            message_id=int(placeholder["message_id"]),
                            text="😿 <i>Muse не отдал видео — попробуй ещё раз чуть позже</i>",
                        )
                    except Exception:
                        pass
                self.store.set_setting("muse.video_retry_after", str(int(time.time() + 1800)))
                self._restart_muse_if_needed(force=True)

        elif kind == "voice":
            voice_bytes = self.muse.generate_voice(prompt, voice=self._active_voice_id())
            if voice_bytes:
                try:
                    self.api.send_voice(
                        chat_id=chat_id,
                        voice=voice_bytes,
                        caption=None,
                        message_thread_id=tid,
                        reply_to_message_id=msg_id,
                    )
                    self.store.add_memory(
                        chat_id, tid, "assistant", "Kiryxa", f"[сказал голосом: {prompt[:200]}]"
                    )
                except Exception as e:
                    logger.warning("Failed to send voice: %s", e)
            else:
                try:
                    self.api.send_message(
                        chat_id=chat_id,
                        message_thread_id=tid,
                        text="😿 <i>фырк, не вышло наговорить голосовое, горло першит ня :3</i>",
                        reply_to_message_id=msg_id,
                    )
                except Exception:
                    pass

    def _handle_vision_reaction_and_reply(
        self,
        message: Dict[str, Any],
        chat_id: int,
        tid: Optional[int],
        mem_tid: Optional[int],
        msg_id: Optional[int],
        file_id: str,
        caption: str,
        mime: str,
        fallback_file_id: Optional[str],
        author_name: str,
        diag_key: Tuple[int, Optional[int]],
        user_id: int,
        now: float,
        is_addressed_or_reply: bool,
        is_drake: bool = False,
    ) -> None:
        if not self.vision or not msg_id:
            return

        try:
            logger.info("Vision processing image for chat %s msg %s...", chat_id, msg_id)
            img_bytes = self.api.get_file_bytes(file_id)
            # Видео больше ~18 МБ не тащим в Vision — берём кадр-превью
            if img_bytes and str(mime).startswith("video") and len(img_bytes) > 18 * 1024 * 1024:
                logger.info("Video %s too large for vision (%s bytes), using thumbnail", file_id, len(img_bytes))
                img_bytes = None
            if not img_bytes and fallback_file_id:
                img_bytes = self.api.get_file_bytes(fallback_file_id)
                mime = "image/jpeg"
            if not img_bytes:
                logger.warning("Could not download media bytes for %s in %s", file_id, chat_id)
                return

            vkey = f"{diag_key[0]}:{diag_key[1]}"
            past = self._recent_vision.setdefault(vkey, deque(maxlen=5))
            seen = [f"{e} {c}" for e, c in past]

            # До двух попыток: если модель повторила недавний комментарий,
            # просим заново, показав ей, что именно повторять нельзя.
            emoji_reaction, comment = "", ""
            # Видео на холодном мосте может думать дольше минуты
            is_video = str(mime).startswith("video")
            for attempt in range(2):
                result = self.vision.analyze_image(
                    img_bytes,
                    caption=caption,
                    mime=mime,
                    timeout=(60.0 if attempt == 0 else 90.0) if is_video else (18.0 if attempt == 0 else 25.0),
                    recent=seen + recent_replies(vkey, 3),
                )
                if not result:
                    if attempt == 0:
                        logger.info("Vision failed for %s, retrying", msg_id)
                        continue
                    logger.warning("Vision analysis returned None for %s", msg_id)
                    # Обратились с картинкой, а Vision не смог — не оставляем
                    # человека без ответа вообще.
                    if is_addressed_or_reply or is_drake:
                        try:
                            self.api.send_message(
                                chat_id=chat_id,
                                message_thread_id=tid,
                                text="😿 <i>картинку не разглядел — подпиши текстом, что там?</i>",
                                parse_mode="HTML",
                                reply_to_message_id=msg_id,
                            )
                        except Exception:
                            pass
                    return
                emoji_reaction, raw_comment = result
                comment = _sanitize_reply(
                    raw_comment,
                    chat_key=vkey,
                    user_text=caption,
                    author_name=author_name,
                )
                if comment:
                    break
                logger.info(
                    "Vision comment for %s repeated a recent reply (attempt %d): %r",
                    msg_id, attempt + 1, raw_comment,
                )
                seen = seen + [f"{emoji_reaction} {raw_comment}"]

            if not comment:
                # Ничего нового не придумал — ставим только реакцию, без воды.
                comment = ""

            logger.info("Vision analyzed msg %s: reaction=%s, comment=%r", msg_id, emoji_reaction, comment)
            if comment:
                past.append((emoji_reaction, comment))

            # 1. Set emoji reaction on the photo/GIF in Telegram
            if emoji_reaction:
                try:
                    self.api.set_message_reaction(chat_id, msg_id, reaction=emoji_reaction)
                except Exception as e:
                    logger.debug("Failed to set reaction: %s", e)

            # 2. If user specifically addressed Kiryxa, is the owner, or in active dialogue, also comment in chat!
            should_comment = is_addressed_or_reply or is_drake
            if should_comment and comment:
                if is_drake or self._chat_limiter.allow(user_id):
                    try:
                        self.api.send_message(
                            chat_id=chat_id,
                            message_thread_id=tid,
                            text=html.escape(comment),
                            parse_mode="HTML",
                            reply_to_message_id=msg_id,
                        )
                        # В память идёт ТОЛЬКО ответ бота: входящее фото уже записано
                        # до запуска потока (иначе при сбое Vision картинка терялась).
                        self.store.add_memory(chat_id, mem_tid, "assistant", "Kiryxa", comment)
                        self._active_dialogues[diag_key] = (user_id, now)
                    except Exception as e:
                        logger.warning("Failed to send vision comment: %s", e)
                        self._forget_reply(vkey, comment)
        except Exception as e:
            logger.warning("Vision handler error: %s", e)

    def _find_keys_and_abuse_topic_id(self, chat_id: int) -> int:
        """Найти ветку для ключей и абузов (например #3 'Abuse and Keys')."""
        try:
            topics = self.store.list_topics(chat_id)
            for t in topics:
                name = str(t.get("topic_name") or "").lower()
                if any(w in name for w in ("abuse", "keys", "ключ", "абуз", "раздач", "халяв")):
                    return int(t["thread_id"])
        except Exception as e:
            logger.debug("Failed to find keys topic in chat %s: %s", chat_id, e)
        return 3

    def _maybe_mirror_drop_from_general(
        self,
        message: Dict[str, Any],
        chat_cfg: Dict[str, Any],
        chat_username: Optional[str] = None,
    ) -> None:
        """Если в General скинули ключи, промокод, триал или гайд — дублируем
        в ветку Abuse and Keys, но из General НЕ удаляем!"""
        chat_id = int(chat_cfg.get("chat_id") or 0)
        msg_id = int(message.get("message_id", 0))
        if not chat_id or not msg_id:
            return

        mirror_key = (chat_id, msg_id)
        with self._lock:
            if mirror_key in self._mirrored_from_general:
                return

        # Проверяем, содержит ли сообщение ценный контент (ключи, триалы, доки)
        verdict = classify_message(
            message,
            mode="smart",
            is_admin=False,
            is_whitelisted=False,
            admin_bypass=False,
            ai_enabled=self.config.ai_enabled,
            ai_api_url=self.config.ai_api_url,
            ai_api_key=self.config.ai_api_key,
            ai_model=self.config.ai_model,
        )
        if not verdict.keep:
            return

        # Игнорируем пустые теги #инфо / !keep без реального содержания
        if verdict.reason == "explicit_keep_tag":
            text = str(message.get("text") or message.get("caption") or "").strip()
            if len(text.split()) < 3:
                return

        target_tid = self._find_keys_and_abuse_topic_id(chat_id)

        logger.info(
            "Mirroring valuable drop msg %s from General to keys/abuse topic #%s (reason: %s)",
            msg_id, target_tid, verdict.reason
        )

        # 1. Сначала пробуем переслать (forward_message) — сохраняется ссылка на автора
        mirrored = False
        try:
            self.api.forward_message(
                chat_id=chat_id,
                from_chat_id=chat_id,
                message_id=msg_id,
                message_thread_id=target_tid,
            )
            mirrored = True
        except Exception as e:
            logger.debug("Forwarding drop failed, trying copyMessage: %s", e)

        # 2. Если пересылка заблокирована приватностью автора — копируем
        if not mirrored:
            try:
                self.api.copy_message(
                    chat_id=chat_id,
                    from_chat_id=chat_id,
                    message_id=msg_id,
                    message_thread_id=target_tid,
                )
                mirrored = True
            except Exception as e2:
                logger.warning("Failed to copy drop msg %s to topic #%s: %s", msg_id, target_tid, e2)

        if mirrored:
            # Запоминаем только после успешной пересылки: при сбое будет повтор
            with self._lock:
                self._mirrored_from_general.add(mirror_key)
            # Ставим реакцию 🔥 на сообщение в General, чтобы видно было, что бот сохранил в ключи
            try:
                self.api.set_message_reaction(chat_id, msg_id, reaction="🔥")
            except Exception:
                pass

    def _move_message_to_general(
        self,
        message: Dict[str, Any],
        chat_cfg: Dict[str, Any],
        topic_cfg: Dict[str, Any],
        reason: str,
        chat_username: Optional[str] = None,
    ) -> None:
        chat_id = int(chat_cfg["chat_id"])
        msg_id = int(message["message_id"])
        from_tid = int(topic_cfg["thread_id"])
        topic_name = str(topic_cfg.get("topic_name") or f"Ветка #{from_tid}")
        target_general_tid: Optional[int] = (
            int(chat_cfg["general_thread_id"])
            if chat_cfg.get("general_thread_id") is not None
            else None
        )

        user = message.get("from") or {}
        user_id = int(user.get("id", 0))
        username = str(user.get("username") or "")
        user_mention = format_user_mention(user)

        topic_url = build_topic_link(chat_id, from_tid, chat_username=chat_username)
        escaped_topic_name = html.escape(topic_name)
        if topic_url:
            topic_ref = f'<a href="{topic_url}">#{escaped_topic_name}</a>'
        else:
            topic_ref = f"<b>#{escaped_topic_name}</b>"

        # Check if user was replying to a specific post inside the topic (not just the root topic header)
        reply_info_line = ""
        reply_msg = message.get("reply_to_message")
        if isinstance(reply_msg, dict):
            reply_msg_id = int(reply_msg.get("message_id", 0))
            is_root_topic_msg = bool(reply_msg.get("forum_topic_created")) or (reply_msg_id == from_tid)
            if reply_msg_id and not is_root_topic_msg:
                reply_link = build_message_link(
                    chat_id, reply_msg_id, thread_id=from_tid, chat_username=chat_username
                )
                reply_text_raw = str(reply_msg.get("text") or reply_msg.get("caption") or "").strip()
                reply_snippet = reply_text_raw.replace("\n", " ")[:55]
                if len(reply_text_raw) > 55:
                    reply_snippet += "…"
                escaped_snip = html.escape(reply_snippet or "сообщение")
                if reply_link:
                    reply_info_line = f"\n↩️ В ответ на: <a href=\"{reply_link}\">{escaped_snip}</a>"
                else:
                    reply_info_line = f"\n↩️ В ответ на: <i>{escaped_snip}</i>"

        text = str(message.get("text") or "").strip()
        caption = str(message.get("caption") or "").strip()

        header = (
            f"↪️ {user_mention} <i>(из ветки {topic_ref})</i>:"
            f"{reply_info_line}"
        )

        sent_ok = False
        try:
            if text:
                # Подкол генерим отдельно и в фоне: раньше LLM держала перенос
                # в основном потоке и тормозила polling.
                full_text = f"{header}\n\n{html.escape(text)}"
                self._send_to_general(chat_id, target_general_tid, full_text)
                sent_ok = True
                self._submit(
                    self._send_quip,
                    chat_id,
                    target_general_tid,
                    user,
                    topic_name,
                    text,
                    user_id,
                )
            else:
                # Media message (photo, sticker, voice, video, etc.): send header + copy message
                self._send_to_general(chat_id, target_general_tid, header)
                try:
                    self.api.copy_message(
                        chat_id=chat_id,
                        from_chat_id=chat_id,
                        message_id=msg_id,
                        message_thread_id=target_general_tid,
                    )
                except TelegramError:
                    if target_general_tid is not None:
                        self.api.copy_message(
                            chat_id=chat_id,
                            from_chat_id=chat_id,
                            message_id=msg_id,
                            message_thread_id=None,
                        )
                sent_ok = True
        except Exception as e:
            logger.error("Failed to forward message %s to general in %s: %s", msg_id, chat_id, e)

        if not sent_ok:
            # Перенос не удался — оригинал НЕ трогаем. Раньше сообщение удалялось
            # из ветки, даже если отправить его в General не получилось: прямая потеря.
            logger.error(
                "Message %s in chat %s was NOT moved to General; keeping original",
                msg_id, chat_id,
            )
            try:
                self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=from_tid,
                    text="⚠️ <i>не смог перенести это в General, оставил тут — попробуй ещё раз позже</i>",
                    parse_mode="HTML",
                )
            except Exception:
                pass
            return

        # Delete the original only after the copy in General is confirmed
        try:
            self.api.delete_message(chat_id=chat_id, message_id=msg_id)
        except Exception as e:
            logger.warning("Could not delete message %s in chat %s: %s", msg_id, chat_id, e)

        self.store.record_moved(
            chat_id=chat_id,
            from_thread_id=from_tid,
            to_thread_id=target_general_tid,
            user_id=user_id,
            username=username,
            reason=reason,
            snippet=text or caption or "[media]",
        )

        # Optional self-destruct notification inside the topic if notify_sec > 0
        notify_sec = int(chat_cfg.get("notify_sec") or 0)
        if notify_sec > 0:
            self._send_temp_notice(chat_id, from_tid, user_mention, notify_sec)

    def _send_quip(
        self,
        chat_id: int,
        general_tid: Optional[int],
        user: Dict[str, Any],
        topic_name: str,
        text: str,
        user_id: int,
    ) -> None:
        """Короткий подкол к перенесённому оффтопу — отдельным сообщением, в фоне."""
        if not (self.config.ai_enabled and self.config.ai_api_key):
            return
        if not self._quip_limiter.allow(user_id):
            return
        author_first = str(user.get("first_name") or user.get("username") or "гой").strip()
        quip = generate_reply(
            text=text,
            author_name=author_first,
            api_url=self.config.ai_api_url,
            api_key=self.config.ai_api_key,
            model=self.config.ai_model,
            context_note=f"Этот человек написал оффтоп в ветке #{topic_name}, ты перенёс его в General. Подколи его одной короткой фразой.",
            owner_name=self.config.owner_name,
            owner_username=self.config.owner_username,
        )
        if not quip:
            return
        try:
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=general_tid,
                text=f"💬 <i>{html.escape(quip)}</i>",
                parse_mode="HTML",
            )
        except Exception as e:
            logger.debug("Failed to send quip: %s", e)

    def _send_to_general(
        self,
        chat_id: int,
        general_tid: Optional[int],
        html_text: str,
    ) -> Dict[str, Any]:
        """Send message to General topic, falling back between thread_id=None and thread_id=1 if needed."""
        try:
            return self.api.send_message(
                chat_id=chat_id,
                text=html_text,
                message_thread_id=general_tid,
                parse_mode="HTML",
                disable_web_page_preview=True,
            )
        except TelegramError as e:
            # If general_tid was None and failed, or was set and failed, fallback to None
            if general_tid is not None:
                return self.api.send_message(
                    chat_id=chat_id,
                    text=html_text,
                    message_thread_id=None,
                    parse_mode="HTML",
                    disable_web_page_preview=True,
                )
            raise e

    def _send_temp_notice(
        self, chat_id: int, thread_id: int, user_mention: str, delay_sec: int
    ) -> None:
        try:
            notice = self.api.send_message(
                chat_id=chat_id,
                message_thread_id=thread_id,
                text=f"🧹 {user_mention}, обсуждения и вопросы перенесены в <b>General</b> чат.",
                parse_mode="HTML",
            )
            notice_id = int(notice.get("message_id", 0))
            if notice_id and delay_sec > 0:
                def _cleanup() -> None:
                    time.sleep(delay_sec)
                    try:
                        self.api.delete_message(chat_id, notice_id)
                    except Exception:
                        pass

                threading.Thread(target=_cleanup, daemon=True).start()
        except Exception:
            pass

    def _handle_private_message(self, message: Dict[str, Any]) -> None:
        chat_id = int(message["chat"]["id"])
        user = message.get("from") or {}
        user_id = int(user.get("id", 0))
        text = str(message.get("text") or "").strip()

        low = text.lower()
        if low.startswith("/voices") or low.startswith("/voice ") or low == "/voice" or low.startswith("/voicemode"):
            self._handle_voice_switcher(chat_id, user_id, text)
            return

        # Support deleting a message by t.me/c/<cid>/... link sent in DM by an admin
        import re
        link_match = re.search(r"t\.me/c/(\d+)/(?:(\d+)/)?(\d+)", text)
        if link_match:
            c_num = link_match.group(1)
            target_chat_id = int(f"-100{c_num}")
            target_msg_id = int(link_match.group(3))
            if self._is_chat_admin(target_chat_id, user_id):
                try:
                    self.api.delete_message(target_chat_id, target_msg_id)
                    self.api.send_message(
                        chat_id=chat_id,
                        text=f"🗑 Сообщение <code>{target_msg_id}</code> в чате <code>{target_chat_id}</code> удалено.",
                        parse_mode="HTML",
                    )
                except Exception as e:
                    self.api.send_message(
                        chat_id=chat_id,
                        text=f"⚠️ Не удалось удалить сообщение <code>{target_msg_id}</code>: {html.escape(str(e))}",
                        parse_mode="HTML",
                    )
                return

        # Обычный текст в ЛС — живой диалог с Кирюхой для своих, без рекламной справки.
        if (
            text
            and not text.startswith("/")
            and (
                user_id in self.config.admin_tg_ids
                or self._is_owner(user_id, str(user.get("username") or ""))
            )
        ):
            # Генерация медиа работает и в ЛС (раньше здесь был только текст)
            if self.muse:
                media_kind, media_prompt = detect_media_intent(text)
                if media_kind and media_prompt:
                    refusal = screen_request(
                        text,
                        is_drake=self._is_owner(user_id, str(user.get("username") or "")),
                    )
                    if refusal:
                        self.api.send_message(chat_id=chat_id, text=html.escape(refusal), parse_mode="HTML")
                        return
                    if not self._chat_limiter.allow(user_id, ignore_user_limit=True):
                        self.api.send_message(chat_id=chat_id, text="🐾 Не так быстро, пару секунд")
                        return
                    self._submit(
                        self._handle_muse_media_generation,
                        chat_id,
                        None,
                        int(message.get("message_id", 0)) or None,
                        media_kind,
                        media_prompt,
                    )
                    return

            author_name = str(user.get("first_name") or user.get("username") or "гой").strip()
            self._handle_dm_dialogue(chat_id, user_id, text, author_name)
            return

        chats = self.store.list_chats()
        lines = [
            "🛡 <b>TopicGuard — Модератор веток с ключами и фри-триалами</b>\n",
            "Бот следит за тематическими ветками (ключи, раздачи, фри триалы, схемы) "
            "и автоматически перекидывает весь оффтоп, вопросы («а как?», «не ворк?») и «спс/забрал» "
            "в <b>General чат</b> с кликабельной ссылкой на ветку и пост, а оригинал удаляет из ветки.\n",
            "<b>Как подключить в приватку:</b>",
            "1️⃣ Добавь бота в свою супергруппу с ветками (Topics).",
            "2️⃣ Выдай боту права <b>Администратора</b> (обязательно: <i>Удаление сообщений / Delete messages</i> и <i>Управление темами / Manage topics</i>).",
            "3️⃣ Отключи Privacy Mode в @BotFather (<code>/setprivacy</code> → <b>Disable</b>) или просто сделай бота админом — в супергруппах админ-боты видят все сообщения.\n",
            "<b>Команды прямо в ветках группы:</b>",
            "• <code>/panel</code> — интерактивная панель с кнопками прямо в ветке",
            "• <code>/guard smart</code> — оставлять ключи, коды, ссылки и инфу по фри-триалам (по умолчанию)",
            "• <code>/guard keys</code> — оставлять только ключи/промокоды/аккаунты",
            "• <code>/guard trials</code> — оставлять гайды, ссылки, БИНы и триал-инфу",
            "• <code>/guard admin</code> — ветка только для постов админов",
            "• <code>/guard off</code> — выключить фильтр в этой ветке",
            "• <code>/setgeneral</code> — назначить текущую ветку как General (куда кидать оффтоп)",
            "• <code>/wl</code> (ответом на сообщение) — добавить/убрать человека в белый список",
            "• <code>/status</code> — статистика и список всех веток\n",
            "💡 <i>Лайфхак: если нужно оставить любое сообщение в ветке, добавь в него <code>#инфо</code>, <code>#ключ</code>, <code>#триал</code> или <code>!keep</code>.</i>",
        ]

        if chats:
            lines.append("\n📊 <b>Подключенные группы:</b>")
            for c in chats[:10]:
                lines.append(
                    f"• <b>{html.escape(c['title'] or str(c['chat_id']))}</b> "
                    f"(перенесено в General: <code>{c['moved_count']}</code>)"
                )

        if user_id in self.config.admin_tg_ids:
            lines.append(
                "\n🎙 <b>Голос бота:</b> <code>/voices</code> — выбрать голос кнопкой, "
                "<code>/voicemode on</code> / <code>off</code> — отвечать голосовыми."
            )

        self.api.send_message(chat_id=chat_id, text="\n".join(lines), parse_mode="HTML")

    # --- Голос ElevenLabs: переключатель в ЛС -------------------------------
    def _handle_dm_dialogue(self, chat_id: int, user_id: int, text: str, author_name: str = "гой") -> None:
        """Обычное сообщение в ЛС: отвечаем как Кирюха, а не рекламной справкой."""
        if not (self.config.ai_enabled and self.config.ai_api_key):
            self.api.send_message(chat_id=chat_id, text="😿 мозги сейчас отключены, зайди позже")
            return
        if not self._chat_limiter.allow(user_id):
            self.api.send_message(chat_id=chat_id, text="🐾 не так быстро, пару секунд")
            return

        # Входящее пишем сразу — даже если модель промолчит, контекст сохранится
        self.store.add_memory(chat_id, None, "user", author_name, text)

        def worker() -> None:
            try:
                hist = self.store.get_memory(chat_id, None, limit=20)
                reply = generate_reply(
                    text=text,
                    author_name=author_name,
                    api_url=self.config.ai_api_url,
                    api_key=self.config.ai_api_key,
                    model=self.config.ai_model,
                    history=hist,
                    exa_api_key=self.config.exa_api_key,
                    is_private_chat=True,
                    chat_key=f"dm:{user_id}",
                    fallback=(
                        (self.config.fallback_api_url, self.config.fallback_api_key, self.config.fallback_model)
                        if self.config.fallback_api_url and self.config.fallback_api_key and self.config.fallback_model
                        else None
                    ),
                    memory_block=self._memory_block(chat_id, None, author_name),
                    owner_name=self.config.owner_name,
                    owner_username=self.config.owner_username,
                )
                if not reply:
                    reply = random.choice(SILENT_FALLBACKS)
                # В ЛС тоже отвечаем голосом, если включён голосовой режим
                audio = None
                target = int(self.config.private_chat_id or 0)
                if target and self._voice_mode_on(target) and len(reply) <= 600:
                    audio = self._speak(reply)
                if audio:
                    try:
                        self.api.send_voice(chat_id=chat_id, voice=audio)
                    except Exception:
                        audio = None
                if not audio:
                    self.api.send_message(
                        chat_id=chat_id,
                        text=html.escape(reply),
                        parse_mode="HTML",
                    )
                self.store.add_memory(chat_id, None, "assistant", "Kiryxa", reply)
            except Exception as e:
                logger.warning("DM dialogue failed: %s", e)

        self._submit(worker)

    def _active_voice_id(self) -> str:
        return self.store.get_setting(
            "tts.voice_id",
            self.config.elevenlabs_voice_id or DEFAULT_VOICE_ID,
        )

    def _voice_speed(self) -> float:
        try:
            return min(1.2, max(0.7, float(self.store.get_setting("tts.speed", "1.0"))))
        except Exception:
            return 1.0

    def _voice_mode_on(self, chat_id: Optional[int] = None) -> bool:
        """Голосовые ответы — глобальный переключатель для всех чатов.

        Раньше ключ был per-chat, и включение в приватке не действовало в
        других группах, где бот тоже общается.
        """
        return self.store.get_setting("tts.voice_replies", "0") == "1"

    def _speak(self, text: str, voice_id: Optional[str] = None) -> Optional[bytes]:
        if not self.muse or not text:
            return None
        try:
            return self.muse.generate_voice(
                text,
                voice=voice_id or self._active_voice_id(),
                speed=self._voice_speed(),
            )
        except Exception as e:
            logger.warning("TTS failed: %s", e)
            return None

    def _voice_keyboard(self) -> Dict[str, Any]:
        rows: List[List[Dict[str, Any]]] = []
        row: List[Dict[str, Any]] = []
        for label, vid, _desc in VOICE_CATALOG:
            row.append(inline_btn(label, f"tts:set:{vid}"))
            if len(row) == 3:
                rows.append(row)
                row = []
        if row:
            rows.append(row)
        speed = self._voice_speed()
        rows.append([
            inline_btn("🐢 −0.05", "tts:spd:-0.05"),
            inline_btn(f"×{speed:.2f}", "tts:spd:info"),
            inline_btn("🐇 +0.05", "tts:spd:+0.05"),
        ])
        target = int(self.config.private_chat_id or 0)
        mode = "вкл" if target and self._voice_mode_on(target) else "выкл"
        rows.append([inline_btn(f"🔊 Отвечать голосом: {mode}", "tts:mode")])
        return inline_keyboard(rows)

    def _voice_switcher_text(self) -> str:
        cur = VOICE_BY_ID.get(self._active_voice_id(), self._active_voice_id())
        target = int(self.config.private_chat_id or 0)
        mode = "вкл" if target and self._voice_mode_on(target) else "выкл"
        return (
            "🎙 <b>Голос Кирюхи</b>\n"
            f"Текущий: <b>{html.escape(cur)}</b> · скорость <b>×{self._voice_speed():.2f}</b> · "
            f"голосовые ответы: <b>{mode}</b>\n\n"
            "Жми кнопку — сразу пришлю пример в ЛС. Скорость можно подкрутить 🐢/🐇."
        )

    def _send_voice_sample(self, chat_id: int, voice_id: str, caption: str = "") -> None:
        sample = (
            "Привет! Это мой новый голос, ня. Ну как, звучит пушисто? "
            "Могу шёпотом, могу с задором — скажи, какой больше нравится."
        )
        audio = self._speak(sample, voice_id)
        if audio:
            try:
                self.api.send_voice(
                    chat_id=chat_id,
                    voice=audio,
                    caption=caption or None,
                    parse_mode="HTML",
                )
                return
            except Exception as e:
                logger.warning("Failed to send voice sample: %s", e)
        self.api.send_message(
            chat_id=chat_id,
            text="😿 Не вышло озвучить — все ключи ElevenLabs в кулдауне, попробуй позже.",
        )

    def _handle_voice_switcher(self, chat_id: int, user_id: int, text: str) -> None:
        if user_id not in self.config.admin_tg_ids:
            self.api.send_message(
                chat_id=chat_id,
                text="⛔ Переключатель голоса доступен только админам.",
                parse_mode="HTML",
            )
            return

        parts = text.split(maxsplit=1)
        cmd = parts[0].lower().split("@")[0]
        arg = parts[1].strip() if len(parts) > 1 else ""
        target = int(self.config.private_chat_id or 0)

        if cmd == "/voicemode":
            if not arg:
                state = "вкл" if target and self._voice_mode_on(target) else "выкл"
                self.api.send_message(
                    chat_id=chat_id,
                    text=f"🔊 Голосовые ответы сейчас: <b>{state}</b>\n<code>/voicemode on</code> или <code>/voicemode off</code>",
                    parse_mode="HTML",
                )
                return
            on = arg.lower() in ("on", "вкл", "1", "да", "yes", "true")
            self.store.set_setting("tts.voice_replies", "1" if on else "0")
            self.api.send_message(
                chat_id=chat_id,
                text=(
                    "🔊 Теперь отвечаю голосом в приватке."
                    if on
                    else "💬 Вернулся к текстовым ответам."
                ),
            )
            return

        if cmd == "/voicespeed":
            try:
                new_speed = min(1.2, max(0.7, float(arg.replace(",", "."))))
            except ValueError:
                self.api.send_message(
                    chat_id=chat_id,
                    text="Скорость — число от 0.7 до 1.2, например <code>/voicespeed 1.15</code>",
                    parse_mode="HTML",
                )
                return
            self.store.set_setting("tts.speed", f"{new_speed:.2f}")
            self._submit(self._send_voice_sample, chat_id, self._active_voice_id(),
                         f"🎙 Скорость ×{new_speed:.2f}")
            return

        # /voices или /voice <имя|номер>
        if arg:
            vid = VOICE_BY_NAME.get(arg.lower())
            if not vid and arg in VOICE_BY_ID:
                vid = arg
            if not vid and arg.isdigit():
                idx = int(arg) - 1
                if 0 <= idx < len(VOICE_CATALOG):
                    vid = VOICE_CATALOG[idx][1]
            if not vid:
                self.api.send_message(
                    chat_id=chat_id,
                    text="Не знаю такой голос. Открой /voices и выбери кнопкой.",
                )
                return
            self.store.set_setting("tts.voice_id", vid)
            self._submit(self._send_voice_sample, chat_id, vid, f"🎙 Выбран голос: {html.escape(VOICE_BY_ID.get(vid, vid))}")
            return

        self.api.send_message(
            chat_id=chat_id,
            text=self._voice_switcher_text(),
            parse_mode="HTML",
            reply_markup=self._voice_keyboard(),
        )

    def _handle_tts_callback(self, cq_id: str, data: str, user_id: int, chat_id: int, msg_id: int) -> None:
        if user_id not in self.config.admin_tg_ids:
            self.api.answer_callback_query(cq_id, text="⛔ Только админ", show_alert=True)
            return

        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""
        target = int(self.config.private_chat_id or 0)

        if action == "mode":
            if not target:
                self.api.answer_callback_query(cq_id, text="Приватка не настроена")
                return
            new_val = "0" if self._voice_mode_on(target) else "1"
            self.store.set_setting("tts.voice_replies", new_val)
            self.api.answer_callback_query(
                cq_id,
                text="Голосовые ответы включены" if new_val == "1" else "Голосовые ответы выключены",
            )
            try:
                self.api.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=self._voice_switcher_text(),
                    parse_mode="HTML",
                    reply_markup=self._voice_keyboard(),
                )
            except Exception:
                pass
            return

        if action == "spd":
            arg = parts[2] if len(parts) > 2 else ""
            if arg == "info":
                self.api.answer_callback_query(cq_id, text=f"Скорость ×{self._voice_speed():.2f}")
                return
            delta = 0.05 if arg.startswith("+") else -0.05
            new_speed = min(1.2, max(0.7, round(self._voice_speed() + delta, 2)))
            self.store.set_setting("tts.speed", f"{new_speed:.2f}")
            self.api.answer_callback_query(cq_id, text=f"Скорость ×{new_speed:.2f}")
            try:
                self.api.edit_message_text(
                    chat_id=chat_id,
                    message_id=msg_id,
                    text=self._voice_switcher_text(),
                    parse_mode="HTML",
                    reply_markup=self._voice_keyboard(),
                )
            except Exception:
                pass
            self._submit(self._send_voice_sample, chat_id, self._active_voice_id(),
                         f"🎙 Скорость ×{new_speed:.2f}")
            return

        if action == "set" and len(parts) >= 3:
            vid = parts[2]
            if vid not in VOICE_BY_ID:
                self.api.answer_callback_query(cq_id, text="Неизвестный голос")
                return
            self.store.set_setting("tts.voice_id", vid)
            self.api.answer_callback_query(cq_id, text=f"Голос: {VOICE_BY_ID[vid]}")
            self._submit(self._send_voice_sample, chat_id, vid, f"🎙 {html.escape(VOICE_BY_ID[vid])}")
            return

        self.api.answer_callback_query(cq_id, text="Не понял кнопку")

    def _handle_group_command(
        self,
        message: Dict[str, Any],
        chat_cfg: Dict[str, Any],
        tid: Optional[int],
    ) -> bool:
        text = str(message.get("text") or "").strip()
        parts = text.split(maxsplit=1)
        cmd_raw = parts[0].lower()
        arg = parts[1].strip() if len(parts) > 1 else ""

        if "@" in cmd_raw:
            cmd_name, target_bot = cmd_raw.split("@", 1)
            if self.bot_username and target_bot.lower() != self.bot_username.lower():
                return False
        else:
            cmd_name = cmd_raw

        known_cmds = {
            "/start",
            "/help",
            "/panel",
            "/guard",
            "/setgeneral",
            "/status",
            "/wl",
            "/whitelist",
            "/autoguard",
            "/adminbypass",
            "/notify",
            "/photo",
            "/video",
            "/voice",
        }
        if cmd_name not in known_cmds:
            return False

        chat_id = int(chat_cfg["chat_id"])
        msg_id = int(message["message_id"])
        user = message.get("from") or {}
        user_id = int(user.get("id", 0))
        is_admin = bool(message.get("sender_chat")) or self.is_user_admin(chat_id, user_id)

        if cmd_name in ("/photo", "/video", "/voice"):
            kind_map = {"/photo": "photo", "/video": "video", "/voice": "voice"}
            kind = kind_map[cmd_name]

            def _media_err(text: str) -> None:
                self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=tid,
                    text=text,
                    parse_mode="HTML",
                    reply_to_message_id=msg_id,
                )

            # Генерация открыта всем, но с фильтрами и лимитером (см. ниже)
            if not self.muse:
                _media_err("😿 Генерация выключена на этом сервере.")
                return True
            if not arg:
                _media_err(
                    f"🐾 <i>Напиши описание после команды, например: "
                    f"<code>{cmd_name} пушистая лисичка в наушниках</code> ня :3</i>"
                )
                return True
            if len(arg) > 400:
                _media_err("😾 Слишком длинный промпт — уложись в 400 символов.")
                return True

            is_drake = self._is_owner(user_id, str(user.get("username") or ""))
            refusal = screen_request(arg, is_drake=is_drake)
            if refusal:
                _media_err(html.escape(refusal))
                return True
            if not self._chat_limiter.allow(user_id):
                _media_err("🐾 Не так быстро, дай выдохнуть пару секунд.")
                return True

            threading.Thread(
                target=self._handle_muse_media_generation,
                args=(chat_id, tid, msg_id, kind, arg),
                daemon=True,
            ).start()
            author_first = str(user.get("first_name") or user.get("username") or "гой").strip()
            self.store.add_memory(chat_id, tid, "user", author_first, f"[запросил {kind}: {arg}]")
            return True

        if not is_admin:
            # Чистим команды только в своей приватке. Во внешних чатах бот
            # ничего не модерирует — удалять чужие сообщения там нельзя.
            if chat_id == int(self.config.private_chat_id or 0):
                try:
                    self.api.delete_message(chat_id, msg_id)
                except Exception:
                    pass
            return True

        # Delete the admin command message itself to keep the topic clean
        # (только в своей приватке, во внешних чатах ничего не удаляем)
        if chat_id == int(self.config.private_chat_id or 0):
            try:
                self.api.delete_message(chat_id, msg_id)
            except Exception:
                pass

        if cmd_name in ("/start", "/help"):
            self._send_help_in_group(chat_id, tid)
            return True

        if cmd_name == "/setgeneral":
            self.store.set_general_thread(chat_id, tid)
            if tid is not None:
                self.store.upsert_topic(chat_id, tid, mode="general", is_manual=True)
            loc_str = f"ветка #{tid}" if tid is not None else "основной General чат"
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=tid,
                text=f"✅ Назначен чат для переноса лишней инфы (General): <b>{loc_str}</b>.",
                parse_mode="HTML",
            )
            return True

        if cmd_name == "/guard":
            if tid is None:
                self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=None,
                    text="ℹ️ Вы находитесь в основном <b>General</b> чате. Вызовите <code>/guard</code> или <code>/panel</code> внутри нужной ветки.",
                    parse_mode="HTML",
                )
                return True

            sub = arg.lower().strip()
            mode_map = {
                "smart": "smart",
                "умный": "smart",
                "on": "smart",
                "вкл": "smart",
                "keys": "keys",
                "ключи": "keys",
                "key": "keys",
                "trials": "trials",
                "триалы": "trials",
                "триал": "trials",
                "info": "trials",
                "инфа": "trials",
                "admin": "admin",
                "админ": "admin",
                "off": "off",
                "выкл": "off",
                "chat": "off",
                "general": "general",
            }
            if not sub:
                self._send_topic_panel(chat_id, tid)
                return True

            chosen = mode_map.get(sub)
            if not chosen:
                self.api.send_message(
                    chat_id=chat_id,
                    message_thread_id=tid,
                    text=(
                        "⚠️ Неизвестный режим. Доступные варианты:\n"
                        "• <code>/guard smart</code> — Ключи + Фри триалы (по умолчанию)\n"
                        "• <code>/guard keys</code> — Только ключи и коды\n"
                        "• <code>/guard trials</code> — Только триалы, гайды и ссылки\n"
                        "• <code>/guard admin</code> — Только посты админов\n"
                        "• <code>/guard off</code> — Без фильтрации"
                    ),
                    parse_mode="HTML",
                )
                return True

            if chosen == "general":
                self.store.set_general_thread(chat_id, tid)
            t_info = self.store.upsert_topic(chat_id, tid, mode=chosen, is_manual=True)
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=tid,
                text=(
                    f"✅ Режим для <b>{html.escape(t_info['topic_name'])}</b> обновлён:\n"
                    f"👉 <b>{MODE_LABELS.get(chosen, chosen)}</b>"
                ),
                parse_mode="HTML",
            )
            return True

        if cmd_name == "/panel":
            self._send_topic_panel(chat_id, tid)
            return True

        if cmd_name == "/status":
            self._send_status(chat_id, tid)
            return True

        if cmd_name in ("/wl", "/whitelist"):
            self._handle_whitelist_cmd(message, chat_id, tid, arg)
            return True

        if cmd_name == "/autoguard":
            cur = bool(chat_cfg.get("auto_guard", 1))
            new_val = not cur if not arg else (arg.lower() in ("on", "1", "true", "вкл", "да"))
            self.store.set_auto_guard(chat_id, new_val)
            state_txt = "ВКЛЮЧЕНА ✅" if new_val else "ВЫКЛЮЧЕНА ❌"
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=tid,
                text=f"⚙️ Авто-защита новых веток: <b>{state_txt}</b>",
                parse_mode="HTML",
            )
            return True

        if cmd_name == "/adminbypass":
            cur = bool(chat_cfg.get("admin_bypass", 0))
            new_val = not cur if not arg else (arg.lower() in ("on", "1", "true", "вкл", "да"))
            self.store.set_admin_bypass(chat_id, new_val)
            state_txt = (
                "ВКЛЮЧЕН ✅ (сообщения админов не переносятся)"
                if new_val
                else "ВЫКЛЮЧЕН 🔥 (оффтоп от всех переносится в General)"
            )
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=tid,
                text=f"👑 Иммунитет админов на флуд в ветках: <b>{state_txt}</b>",
                parse_mode="HTML",
            )
            return True

        if cmd_name == "/notify":
            sec = int(arg) if arg.isdigit() else (0 if chat_cfg.get("notify_sec", 0) else 5)
            self.store.set_notify_sec(chat_id, sec)
            txt = (
                f"🔔 Временное уведомление в ветке при переносе: <b>{sec} сек.</b>"
                if sec > 0
                else "🔕 Временное уведомление в ветке отключено (тихий перенос в General)."
            )
            self.api.send_message(chat_id=chat_id, message_thread_id=tid, text=txt, parse_mode="HTML")
            return True

        return True

    def _handle_whitelist_cmd(
        self,
        message: Dict[str, Any],
        chat_id: int,
        tid: Optional[int],
        arg: str,
    ) -> None:
        reply = message.get("reply_to_message")
        target_uid: Optional[int] = None
        target_uname: str = ""

        if isinstance(reply, dict) and reply.get("from") and not reply.get("forum_topic_created"):
            r_user = reply["from"]
            target_uid = int(r_user.get("id", 0))
            target_uname = str(r_user.get("username") or r_user.get("first_name") or "")
        elif arg.lstrip("-").isdigit():
            target_uid = int(arg)

        if not target_uid:
            wl = self.store.list_whitelist(chat_id)
            if not wl:
                txt = (
                    "📋 <b>Белый список пуст.</b>\n"
                    "Чтобы добавить/удалить пользователя, ответьте на его сообщение командой <code>/wl</code> "
                    "или введите <code>/wl USER_ID</code>."
                )
            else:
                items = [
                    f"• <code>{w['user_id']}</code> ({html.escape(w['username'] or 'без ника')})"
                    for w in wl
                ]
                txt = "📋 <b>Белый список (без фильтрации в ветках):</b>\n" + "\n".join(items)
                if len(txt) > 3900:
                    txt = txt[:3900].rsplit("\n", 1)[0] + "\n…<i>список обрезан</i>"
            self.api.send_message(chat_id=chat_id, message_thread_id=tid, text=txt, parse_mode="HTML")
            return

        if self.store.is_whitelisted(chat_id, target_uid):
            self.store.remove_whitelist(chat_id, target_uid)
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=tid,
                text=f"➖ Пользователь <code>{target_uid}</code> убран из белого списка.",
                parse_mode="HTML",
            )
        else:
            self.store.add_whitelist(chat_id, target_uid, target_uname)
            self.api.send_message(
                chat_id=chat_id,
                message_thread_id=tid,
                text=f"✅ Пользователь <code>{target_uid}</code> ({html.escape(target_uname)}) добавлен в белый список.",
                parse_mode="HTML",
            )

    def _build_panel_content(
        self, chat_id: int, tid: Optional[int]
    ) -> Tuple[str, Dict[str, Any]]:
        chat_cfg = self.store.get_chat(chat_id)
        gen_tid = chat_cfg.get("general_thread_id")
        gen_label = f"Ветка #{gen_tid}" if gen_tid is not None else "Основной General"
        auto_guard = bool(chat_cfg.get("auto_guard", 1))
        admin_bypass = bool(chat_cfg.get("admin_bypass", 0))

        if tid is not None:
            t_info = self.store.get_topic(chat_id, tid) or self.store.upsert_topic(chat_id, tid)
            cur_mode = t_info.get("mode", "smart")
            t_name = html.escape(t_info.get("topic_name") or f"Ветка #{tid}")
            text = (
                f"🛡 <b>Настройка ветки: {t_name}</b> (ID: <code>{tid}</code>)\n\n"
                f"• Текущий режим: <b>{MODE_LABELS.get(cur_mode, cur_mode)}</b>\n"
                f"• Куда летит оффтоп: <b>{gen_label}</b>\n"
                f"• Авто-защита новых веток: <b>{'ВКЛ ✅' if auto_guard else 'ВЫКЛ ❌'}</b>\n"
                f"• Оффтоп админов: <b>{'Игнорировать 👑' if admin_bypass else 'Переносить в General 🔥'}</b>\n"
                f"• Статистика ветки: оставлено <code>{t_info.get('kept_count', 0)}</code> | "
                f"перенесено <code>{t_info.get('moved_count', 0)}</code>\n\n"
                f"Выберите режим работы для этой ветки:"
            )
            tid_val = tid
        else:
            cur_mode = "general"
            text = (
                f"🛡 <b>Панель управления TopicGuard (Основной General)</b>\n\n"
                f"• Куда летит оффтоп: <b>{gen_label}</b>\n"
                f"• Авто-защита новых веток: <b>{'ВКЛ ✅' if auto_guard else 'ВЫКЛ ❌'}</b>\n"
                f"• Оффтоп админов: <b>{'Игнорировать 👑' if admin_bypass else 'Переносить в General 🔥'}</b>\n"
                f"• Всего перенесено в группе: <code>{chat_cfg.get('moved_count', 0)}</code>\n\n"
                f"Чтобы настроить конкретную ветку с ключами или триалами, вызовите <code>/panel</code> внутри неё."
            )
            tid_val = 0

        def _mark(m: str, label: str) -> str:
            return f"✅ {label}" if cur_mode == m else label

        rows: List[List[Dict[str, Any]]] = []
        if tid_val > 0:
            rows.extend(
                [
                    [
                        inline_btn(
                            _mark("smart", "🧠 Умный (Ключи + Триалы)"),
                            callback_data=f"tg:mode:{tid_val}:smart",
                        )
                    ],
                    [
                        inline_btn(
                            _mark("keys", "🔑 Только ключи"),
                            callback_data=f"tg:mode:{tid_val}:keys",
                        ),
                        inline_btn(
                            _mark("trials", "🎁 Триалы / Инфа"),
                            callback_data=f"tg:mode:{tid_val}:trials",
                        ),
                    ],
                    [
                        inline_btn(
                            _mark("admin", "👑 Только админы"),
                            callback_data=f"tg:mode:{tid_val}:admin",
                        ),
                        inline_btn(
                            _mark("off", "💬 Выкл фильтр"),
                            callback_data=f"tg:mode:{tid_val}:off",
                        ),
                    ],
                    [
                        inline_btn(
                            "🏠 Сделать эту ветку General чатом",
                            callback_data=f"tg:setgen:{tid_val}",
                        )
                    ],
                ]
            )
        else:
            rows.append(
                [
                    inline_btn(
                        "🏠 Сделать основной чат General",
                        callback_data="tg:setgen:0",
                    )
                ]
            )

        rows.append(
            [
                inline_btn(
                    f"Авто-защита: {'ВКЛ ✅' if auto_guard else 'ВЫКЛ ❌'}",
                    callback_data=f"tg:toggle_auto:{tid_val}",
                ),
                inline_btn(
                    f"Админ-иммунитет: {'ВКЛ 👑' if admin_bypass else 'ВЫКЛ 🔥'}",
                    callback_data=f"tg:toggle_adm:{tid_val}",
                ),
            ]
        )
        rows.append(
            [
                inline_btn("🗑 Закрыть панель", callback_data=f"tg:close:{tid_val}"),
            ]
        )

        return text, inline_keyboard(rows)

    def _send_topic_panel(self, chat_id: int, tid: Optional[int]) -> None:
        text, kb = self._build_panel_content(chat_id, tid)
        self.api.send_message(
            chat_id=chat_id,
            message_thread_id=tid,
            text=text,
            parse_mode="HTML",
            reply_markup=kb,
        )

    def _send_status(self, chat_id: int, tid: Optional[int]) -> None:
        chat_cfg = self.store.get_chat(chat_id)
        topics = self.store.list_topics(chat_id)
        gen_tid = chat_cfg.get("general_thread_id")
        gen_str = f"Ветка #{gen_tid}" if gen_tid is not None else "Основной General чат"

        lines = [
            f"📊 <b>Статус TopicGuard — {html.escape(chat_cfg.get('title') or str(chat_id))}</b>\n",
            f"• 🏠 Чат для оффтопа (General): <b>{gen_str}</b>",
            f"• 🛡 Авто-защита веток: <b>{'ВКЛ ✅' if chat_cfg.get('auto_guard', 1) else 'ВЫКЛ ❌'}</b>",
            f"• 👑 Иммунитет админов: <b>{'ВКЛ' if chat_cfg.get('admin_bypass', 0) else 'ВЫКЛ (переносит у всех)'}</b>",
            f"• 🧹 Всего перенесено сообщений: <code>{chat_cfg.get('moved_count', 0)}</code>\n",
        ]

        if topics:
            lines.append("📂 <b>Известные ветки:</b>")
            for t in topics:
                m_label = MODE_LABELS.get(t["mode"], t["mode"])
                lines.append(
                    f"• <b>{html.escape(t['topic_name'])}</b> (<code>#{t['thread_id']}</code>) — {m_label} "
                    f"[✅ {t['kept_count']} | ↪️ {t['moved_count']}]"
                )
        else:
            lines.append("<i>Ветки появятся в списке автоматически при первых сообщениях в них.</i>")

        text = "\n".join(lines)
        if len(text) > 3900:
            # Telegram режет сообщения >4096 символов, и команда молча падала
            text = text[:3900].rsplit("\n", 1)[0] + "\n…<i>список обрезан</i>"
        self.api.send_message(
            chat_id=chat_id,
            message_thread_id=tid,
            text=text,
            parse_mode="HTML",
        )

    def _send_help_in_group(self, chat_id: int, tid: Optional[int]) -> None:
        text = (
            "🛡 <b>TopicGuard — Управление ветками</b>\n\n"
            "Бот автоматически оставляет в тематических ветках <b>ключи, промокоды, аккаунты, "
            "ссылки, БИНы и гайды по фри-триалам</b>, а любые обсуждения, вопросы и «спс/забрал» "
            "переносит в <b>General чат</b> с цитатой и ссылкой на ветку.\n\n"
            "<b>Команды (для админов):</b>\n"
            "• <code>/panel</code> — открыть меню настройки текущей ветки\n"
            "• <code>/guard smart</code> — умный режим (ключи + инфа по фри триалам)\n"
            "• <code>/guard keys</code> — только ключи, промокоды и аккаунты\n"
            "• <code>/guard trials</code> — гайды, ссылки и инфа по триалам\n"
            "• <code>/guard admin</code> — только сообщения админов\n"
            "• <code>/guard off</code> — отключить перенос в этой ветке\n"
            "• <code>/setgeneral</code> — сделать текущую ветку General-чатом для оффтопа\n"
            "• <code>/adminbypass</code> — переключить перенос оффтопа от админов\n"
            "• <code>/wl</code> (реплаем) — белый список\n"
            "• <code>/status</code> — сводка по всем веткам\n\n"
            "💡 <i>Чтобы принудительно оставить любое сообщение в ветке, добавьте тег "
            "<code>#инфо</code>, <code>#ключ</code>, <code>#триал</code> или <code>!keep</code>.</i>"
        )
        self.api.send_message(
            chat_id=chat_id,
            message_thread_id=tid,
            text=text,
            parse_mode="HTML",
        )

    def handle_callback_query(self, cq: Dict[str, Any]) -> None:
        cq_id = str(cq.get("id") or "")
        data = str(cq.get("data") or "")
        user = cq.get("from") or {}
        user_id = int(user.get("id", 0))
        message = cq.get("message") or {}
        chat = message.get("chat") or {}
        chat_id = int(chat.get("id", 0))
        msg_id = int(message.get("message_id", 0))

        if data.startswith("tts:"):
            self._handle_tts_callback(cq_id, data, user_id, chat_id, msg_id)
            return

        if not data.startswith("tg:"):
            return

        if not self.is_user_admin(chat_id, user_id):
            self.api.answer_callback_query(
                cq_id, text="⛔ Только администраторы могут менять настройки веток.", show_alert=True
            )
            return

        parts = data.split(":")
        action = parts[1] if len(parts) > 1 else ""

        if action == "close":
            try:
                self.api.delete_message(chat_id, msg_id)
            except Exception:
                pass
            self.api.answer_callback_query(cq_id, text="Панель закрыта")
            return

        if action == "mode" and len(parts) >= 4:
            tid_val = int(parts[2])
            new_mode = parts[3]
            self.store.upsert_topic(chat_id, tid_val, mode=new_mode, is_manual=True)
            text, kb = self._build_panel_content(chat_id, tid_val if tid_val > 0 else None)
            try:
                self.api.edit_message_text(chat_id, msg_id, text=text, reply_markup=kb)
            except Exception:
                pass
            self.api.answer_callback_query(
                cq_id, text=f"Режим: {MODE_LABELS.get(new_mode, new_mode)}"
            )
            return

        if action == "setgen" and len(parts) >= 3:
            tid_val = int(parts[2])
            new_gen = tid_val if tid_val > 0 else None
            self.store.set_general_thread(chat_id, new_gen)
            if new_gen is not None:
                self.store.upsert_topic(chat_id, new_gen, mode="general", is_manual=True)
            text, kb = self._build_panel_content(chat_id, new_gen)
            try:
                self.api.edit_message_text(chat_id, msg_id, text=text, reply_markup=kb)
            except Exception:
                pass
            self.api.answer_callback_query(cq_id, text="Назначено как General чат!")
            return

        if action == "toggle_auto" and len(parts) >= 3:
            tid_val = int(parts[2])
            chat_cfg = self.store.get_chat(chat_id)
            new_val = not bool(chat_cfg.get("auto_guard", 1))
            self.store.set_auto_guard(chat_id, new_val)
            text, kb = self._build_panel_content(chat_id, tid_val if tid_val > 0 else None)
            try:
                self.api.edit_message_text(chat_id, msg_id, text=text, reply_markup=kb)
            except Exception:
                pass
            self.api.answer_callback_query(
                cq_id, text=f"Авто-защита: {'ВКЛ' if new_val else 'ВЫКЛ'}"
            )
            return

        if action == "toggle_adm" and len(parts) >= 3:
            tid_val = int(parts[2])
            chat_cfg = self.store.get_chat(chat_id)
            new_val = not bool(chat_cfg.get("admin_bypass", 0))
            self.store.set_admin_bypass(chat_id, new_val)
            text, kb = self._build_panel_content(chat_id, tid_val if tid_val > 0 else None)
            try:
                self.api.edit_message_text(chat_id, msg_id, text=text, reply_markup=kb)
            except Exception:
                pass
            self.api.answer_callback_query(
                cq_id,
                text=f"Иммунитет админов: {'ВКЛ' if new_val else 'ВЫКЛ'}",
            )
            return
