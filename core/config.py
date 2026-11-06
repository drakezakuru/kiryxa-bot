"""Configuration loader for TopicGuard Bot."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT_DIR / "config.json"
KEY_RE = re.compile(r"sk_[A-Za-z0-9_-]{20,}")

logger = logging.getLogger("topic_guard.config")

_FALSE_BOOL_STRINGS = {"false", "0", "no", "off"}
_TRUE_BOOL_STRINGS = {"true", "1", "yes", "on"}


def _to_int(value: object, default: int) -> int:
    """Мягкий разбор int: кривые значения не должны ронять старт."""
    if value is None:
        return default
    try:
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return default
        return int(value)
    except (ValueError, TypeError):
        return default


def _to_bool(value: object, default: bool = False) -> bool:
    """Строгий разбор bool: строки "false"/"0"/"no"/"off" не должны давать True."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if not text:
        return default
    if text in _FALSE_BOOL_STRINGS:
        return False
    if text in _TRUE_BOOL_STRINGS:
        return True
    return default


@dataclass
class Config:
    bot_token: str = ""
    admin_tg_ids: List[int] = field(default_factory=list)
    # Владелец бота: используется для «свой/создатель» в промптах и ответах
    owner_name: str = "Owner"
    owner_username: str = ""
    owner_tg_ids: List[int] = field(default_factory=list)
    default_auto_guard: bool = True
    default_admin_bypass: bool = True
    default_notify_sec: int = 0
    poll_timeout: int = 25
    data_dir: str = str(ROOT_DIR / "data")
    ai_enabled: bool = True
    ai_api_url: str = "https://api.inceptionlabs.ai/v1/chat/completions"
    ai_api_key: str = ""
    ai_model: str = "mercury-3"
    # Резервная модель (например, uncensored на Instabox): используется только если основная недоступна
    fallback_api_url: str = ""
    fallback_api_key: str = ""
    fallback_model: str = ""
    exa_api_key: str = ""
    private_chat_id: int = 0
    muse_enabled: bool = True
    # Видео Muse зависает и блокирует весь аккаунт (включая фото)
    muse_video_enabled: bool = True
    muse_api_url: str = "http://127.0.0.1:12001"
    muse_voice: str = "aria"
    vision_enabled: bool = True
    vision_api_url: str = "http://127.0.0.1:12006"
    vision_api_key: str = ""
    vision_model: str = "gemini-3.8-flash-high"
    # Паки стикеров, индексируемых по эмодзи (публичные имена наборов в Telegram)
    sticker_sets: List[str] = field(default_factory=list)
    # ElevenLabs: пул ключей + голос
    elevenlabs_enabled: bool = True
    elevenlabs_keys: List[str] = field(default_factory=list)
    elevenlabs_keys_file: str = ""
    elevenlabs_voice_id: str = ""
    elevenlabs_model: str = ""

    @property
    def db_path(self) -> Path:
        return Path(self.data_dir) / "topic_guard.db"


def load_config(path: Optional[str | Path] = None) -> Config:
    cfg_path = Path(path) if path else DEFAULT_CONFIG_PATH
    try:
        cfg_dir = cfg_path.resolve().parent
    except (OSError, RuntimeError):
        cfg_dir = cfg_path.parent
    raw: dict = {}
    if cfg_path.is_file():
        try:
            raw = json.loads(cfg_path.read_text(encoding="utf-8"))
        except Exception:
            raw = {}

    bot_token = (
        os.environ.get("TELEGRAM_BOT_TOKEN")
        or os.environ.get("BOT_TOKEN")
        or str(raw.get("bot_token") or "").strip()
    )

    admin_ids_raw = raw.get("admin_tg_ids") or []
    env_admins = os.environ.get("ADMIN_TG_IDS", "").strip()
    if env_admins:
        for part in env_admins.split(","):
            part = part.strip()
            if part.lstrip("-").isdigit():
                admin_ids_raw.append(int(part))

    admin_tg_ids: List[int] = []
    for x in admin_ids_raw:
        try:
            val = int(x)
            if val not in admin_tg_ids:
                admin_tg_ids.append(val)
        except (ValueError, TypeError):
            pass

    # Владелец: имя/username для промптов и id для проверки «свой».
    # Никаких личных данных в коде — всё приходит из config.json или env.
    owner_name = (
        str(os.environ.get("OWNER_NAME") or "").strip()
        or str(raw.get("owner_name") or "").strip()
        or "Owner"
    )
    owner_username = (
        os.environ.get("OWNER_USERNAME")
        or str(raw.get("owner_username") or "").strip()
    ).strip().lstrip("@")

    owner_ids_raw = raw.get("owner_tg_ids") or []
    if isinstance(owner_ids_raw, str):
        owner_ids_raw = [p.strip() for p in owner_ids_raw.split(",") if p.strip()]
    elif isinstance(owner_ids_raw, (list, tuple)):
        owner_ids_raw = list(owner_ids_raw)
    else:
        owner_ids_raw = [owner_ids_raw]
    env_owners = os.environ.get("OWNER_TG_IDS", "").strip()
    if env_owners:
        for part in env_owners.split(","):
            part = part.strip()
            if part.lstrip("-").isdigit():
                owner_ids_raw.append(int(part))

    owner_tg_ids: List[int] = []
    for x in owner_ids_raw:
        try:
            val = int(x)
            if val not in owner_tg_ids:
                owner_tg_ids.append(val)
        except (ValueError, TypeError):
            pass

    default_auto_guard = _to_bool(raw.get("default_auto_guard"), True)
    default_admin_bypass = _to_bool(raw.get("default_admin_bypass"), True)
    default_notify_sec = _to_int(raw.get("default_notify_sec"), 0)
    poll_timeout = _to_int(raw.get("poll_timeout"), 25)

    raw_data_dir = str(raw.get("data_dir") or "").strip()
    data_dir_path = Path(raw_data_dir).expanduser() if raw_data_dir else ROOT_DIR / "data"
    if not data_dir_path.is_absolute():
        # Относительный путь считаем от каталога конфига (обычно корень проекта).
        data_dir_path = cfg_dir / data_dir_path
    try:
        data_dir_path = data_dir_path.resolve()
    except (OSError, RuntimeError) as e:
        # Битый/зацикленный symlink не должен ронять старт.
        logger.warning("Cannot resolve data_dir %s: %s", data_dir_path, e)
        data_dir_path = Path(os.path.abspath(data_dir_path))
    try:
        data_dir_path.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        logger.warning(
            "Cannot create data_dir %s (%s); falling back to %s",
            data_dir_path,
            e,
            ROOT_DIR / "data",
        )
        data_dir_path = ROOT_DIR / "data"
        try:
            data_dir_path.mkdir(parents=True, exist_ok=True)
        except OSError as fallback_err:
            logger.warning("Cannot create default data_dir %s: %s", data_dir_path, fallback_err)
    data_dir = str(data_dir_path)

    ai_enabled = _to_bool(raw.get("ai_enabled"), True)
    ai_api_url = (
        os.environ.get("AI_API_URL")
        or str(raw.get("ai_api_url") or "https://api.inceptionlabs.ai/v1/chat/completions").strip()
    )
    ai_api_key = (
        os.environ.get("AI_API_KEY")
        or os.environ.get("INCEPTION_API_KEY")
        or str(raw.get("ai_api_key") or "").strip()
    )
    ai_model = (
        os.environ.get("AI_MODEL")
        or str(raw.get("ai_model") or "mercury-3").strip()
    )
    fallback_api_url = (
        os.environ.get("FALLBACK_API_URL")
        or str(raw.get("fallback_api_url") or "").strip()
    )
    fallback_api_key = (
        os.environ.get("FALLBACK_API_KEY")
        or str(raw.get("fallback_api_key") or "").strip()
    )
    fallback_model = (
        os.environ.get("FALLBACK_MODEL")
        or str(raw.get("fallback_model") or "").strip()
    )
    exa_api_key = (
        os.environ.get("EXA_API_KEY")
        or str(raw.get("exa_api_key") or "").strip()
    )
    private_chat_id_raw = raw.get("private_chat_id") or os.environ.get("PRIVATE_CHAT_ID")
    private_chat_id = _to_int(private_chat_id_raw, 0)

    muse_enabled = _to_bool(raw.get("muse_enabled"), True)
    muse_video_enabled = _to_bool(raw.get("muse_video_enabled"), True)
    muse_api_url = (
        os.environ.get("MUSE_API_URL")
        or str(raw.get("muse_api_url") or "http://127.0.0.1:12001").strip()
    )
    muse_voice = (
        os.environ.get("MUSE_VOICE")
        or str(raw.get("muse_voice") or "aria").strip()
    )

    vision_enabled = _to_bool(raw.get("vision_enabled"), True)
    vision_api_url = (
        os.environ.get("VISION_API_URL")
        or str(raw.get("vision_api_url") or "http://127.0.0.1:12006").strip()
    )
    vision_api_key = (
        os.environ.get("VISION_API_KEY")
        or str(raw.get("vision_api_key") or "").strip()
    )
    vision_model = (
        os.environ.get("VISION_MODEL")
        or str(raw.get("vision_model") or "gemini-3.8-flash-high").strip()
    )
    sticker_sets: List[str] = []
    raw_sets = os.environ.get("STICKER_SETS") or raw.get("sticker_sets") or []
    if isinstance(raw_sets, str):
        raw_sets = [p.strip() for p in raw_sets.split(",") if p.strip()]
    for name in raw_sets or []:
        name = str(name).strip()
        if name and name not in sticker_sets:
            sticker_sets.append(name)

    # ElevenLabs: ключей много, они мрут — держим файл со списком, а не один ключ
    elevenlabs_enabled = _to_bool(raw.get("elevenlabs_enabled"), True)
    elevenlabs_voice_id = (
        os.environ.get("ELEVENLABS_VOICE_ID")
        or str(raw.get("elevenlabs_voice_id") or "").strip()
    )
    elevenlabs_model = (
        os.environ.get("ELEVENLABS_MODEL")
        or str(raw.get("elevenlabs_model") or "").strip()
    )
    elevenlabs_keys_file = (
        os.environ.get("ELEVENLABS_KEYS_FILE")
        or str(raw.get("elevenlabs_keys_file") or "").strip()
        or str(Path(data_dir) / "elevenlabs_keys.txt")
    )
    elevenlabs_keys: List[str] = []
    env_keys = os.environ.get("ELEVENLABS_KEYS") or os.environ.get("ELEVENLABS_API_KEY") or ""
    single = str(raw.get("elevenlabs_api_key") or "").strip()
    if single:
        env_keys = ",".join(p for p in (env_keys, single) if p)
    for part in re.split(r"[,;\s]+", env_keys):
        part = part.strip()
        if part and part not in elevenlabs_keys:
            elevenlabs_keys.append(part)
    try:
        keys_path = Path(elevenlabs_keys_file)
        if keys_path.is_file():
            for k in KEY_RE.findall(keys_path.read_text(encoding="utf-8", errors="ignore")):
                if k not in elevenlabs_keys:
                    elevenlabs_keys.append(k)
    except Exception:
        pass

    return Config(
        bot_token=bot_token,
        admin_tg_ids=admin_tg_ids,
        owner_name=owner_name,
        owner_username=owner_username,
        owner_tg_ids=owner_tg_ids,
        default_auto_guard=default_auto_guard,
        default_admin_bypass=default_admin_bypass,
        default_notify_sec=default_notify_sec,
        poll_timeout=poll_timeout,
        data_dir=data_dir,
        ai_enabled=ai_enabled,
        ai_api_url=ai_api_url,
        ai_api_key=ai_api_key,
        ai_model=ai_model,
        fallback_api_url=fallback_api_url,
        fallback_api_key=fallback_api_key,
        fallback_model=fallback_model,
        exa_api_key=exa_api_key,
        private_chat_id=private_chat_id,
        muse_enabled=muse_enabled,
        muse_video_enabled=muse_video_enabled,
        muse_api_url=muse_api_url,
        muse_voice=muse_voice,
        vision_enabled=vision_enabled,
        vision_api_url=vision_api_url,
        vision_api_key=vision_api_key,
        vision_model=vision_model,
        sticker_sets=sticker_sets,
        elevenlabs_enabled=elevenlabs_enabled,
        elevenlabs_voice_id=elevenlabs_voice_id,
        elevenlabs_model=elevenlabs_model,
        elevenlabs_keys_file=elevenlabs_keys_file,
        elevenlabs_keys=elevenlabs_keys,
    )
