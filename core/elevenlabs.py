"""ElevenLabs TTS client with key rotation.

Ключи из пачки часто умирают или упираются в лимит, поэтому клиент держит
список и переключается на следующий при 401/402/429. Один и тот же мёртвый
ключ повторно не дёргаем: у него есть cooldown.
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("topic_guard.elevenlabs")

API_BASE = "https://api.elevenlabs.io/v1"
KEY_RE = re.compile(r"sk_[A-Za-z0-9_-]{20,}")

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_COOLDOWN_PATH = ROOT_DIR / "data" / "elevenlabs_cooldown.json"

# Минимальный размер валидного mp3: короткие реплики («да») бывают < 1000 байт.
MIN_AUDIO_BYTES = 200
# Не чаще одного раза в 2 секунды пишем кулдауны на диск.
COOLDOWN_SAVE_INTERVAL = 2.0

# Кулдауны: сколько секунд не трогать ключ после ошибки
COOLDOWN_AUTH = 1800.0   # 401/403 — ключ мёртв или без прав
COOLDOWN_QUOTA = 300.0   # 429 — лимит запросов
COOLDOWN_PAYMENT = 3600.0  # 402 — кончились кредиты

DEFAULT_VOICE_ID = "cgSgspJ2msm6clMCkdW9"   # Jessica — Playful, Bright, Warm
DEFAULT_MODEL = "eleven_multilingual_v2"

# Голоса для переключателя в ЛС: (подпись, voice_id, описание)
VOICE_CATALOG = [
    ("Jessica", "cgSgspJ2msm6clMCkdW9", "игривый, яркий — няшный дефолт"),
    ("AImee", "MBGgaJPW6debJHBqQsSJ", "мягкий полушёпот, ASMR"),
    ("Laura", "FGY2WhTYpPnrIDTdsKH5", "солнечный, с задором"),
    ("Will", "bIHbv24MWmeRgasZH58o", "расслабленный молодой парень"),
    ("Liam", "TX3LPaxmHKxFdv7VOQHJ", "энергичный, как для рилсов"),
    ("Charlie", "IKne3meq5aSn9XLyUdCD", "молодой австралийский, уверенный"),
    ("Eric", "cjVigY5qzO86Huf0OWal", "гладкий, обволакивающий"),
    ("Chris", "iP95p4xoKVk53GoZ742B", "обаятельный, простой"),
    ("Sarah", "EXAVITQu4vr4xnSDxMaL", "тёплый, спокойный"),
    ("Lily", "pFZP5JQG7iQjIQuC4Bku", "бархатистый, актёрский"),
    ("Bella", "hpp4J3VqNfWAUOO0d1Us", "светлый, приветливый"),
    ("Alice", "Xb7hH8MSUJpSbSDYk0k2", "чёткий, дружелюбный"),
    ("Matilda", "XrExE9yKIg1WjnnlVkGX", "низкий женский альт"),
]
VOICE_BY_ID = {vid: label for label, vid, _ in VOICE_CATALOG}
VOICE_BY_NAME = {label.lower(): vid for label, vid, _ in VOICE_CATALOG}


def load_keys_from_file(path: str | Path) -> List[str]:
    """Достать ключи из txt/md: подходит и чистый список, и таблица из отчёта."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="ignore")
    except Exception as e:
        logger.debug("Cannot read ElevenLabs keys file %s: %s", path, e)
        return []
    seen = set()
    out: List[str] = []
    for k in KEY_RE.findall(text):
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


class ElevenLabsClient:
    """TTS через ElevenLabs с ротацией ключей."""

    def __init__(
        self,
        api_keys: List[str],
        voice_id: str = DEFAULT_VOICE_ID,
        model: str = DEFAULT_MODEL,
        timeout: float = 30.0,
        cooldown_path: Optional[str | Path] = None,
        speed: float = 1.0,
    ) -> None:
        self.api_keys = [k for k in api_keys if k]
        self.voice_id = voice_id or DEFAULT_VOICE_ID
        self.model = model or DEFAULT_MODEL
        self.timeout = timeout
        # ElevenLabs принимает speed 0.7–1.2; всё остальное обрезаем
        self.speed = min(1.2, max(0.7, float(speed or 1.0)))
        self._lock = threading.Lock()
        self._save_lock = threading.Lock()
        self._index = 0
        self._cooldown_until: dict[str, float] = {}
        self.cooldown_path = Path(cooldown_path) if cooldown_path else DEFAULT_COOLDOWN_PATH
        self._last_save = 0.0
        self._load_cooldowns()

    # --- служебное -------------------------------------------------------
    @property
    def enabled(self) -> bool:
        return bool(self.api_keys)

    def _load_cooldowns(self) -> None:
        """Поднять кулдауны с диска, отбросив уже истёкшие."""
        try:
            raw = json.loads(self.cooldown_path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return
        except (OSError, ValueError, TypeError) as e:
            logger.debug("Cannot load ElevenLabs cooldowns from %s: %s", self.cooldown_path, e)
            return
        if not isinstance(raw, dict):
            return
        now = time.time()
        loaded: dict[str, float] = {}
        for key, until in raw.items():
            if not isinstance(key, str):
                continue
            try:
                ts = float(until)
            except (TypeError, ValueError):
                continue
            if ts > now:
                loaded[key] = ts
        self._cooldown_until = loaded

    def _save_cooldowns(self) -> None:
        """Атомарно записать активные кулдауны (tmp + os.replace).

        Пишем не чаще раза в COOLDOWN_SAVE_INTERVAL секунд; проверка внутри
        лока, чтобы параллельные TTS-вызовы не устроили гонку записи.
        """
        with self._save_lock:
            now = time.time()
            if now - self._last_save < COOLDOWN_SAVE_INTERVAL:
                return
            self._last_save = now
            with self._lock:
                data = {
                    key: until
                    for key, until in self._cooldown_until.items()
                    if until > now
                }
            tmp = self.cooldown_path.with_name(
                f".{self.cooldown_path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
            )
            try:
                self.cooldown_path.parent.mkdir(parents=True, exist_ok=True)
                tmp.write_text(
                    json.dumps(data, ensure_ascii=False, sort_keys=True),
                    encoding="utf-8",
                )
                os.replace(tmp, self.cooldown_path)
            except OSError as e:
                logger.warning(
                    "Cannot save ElevenLabs cooldowns to %s: %s", self.cooldown_path, e
                )
                try:
                    tmp.unlink(missing_ok=True)
                except OSError:
                    pass

    def available_keys(self) -> int:
        now = time.time()
        with self._lock:
            return sum(1 for k in self.api_keys if self._cooldown_until.get(k, 0.0) <= now)

    def _next_key(self) -> Optional[str]:
        now = time.time()
        with self._lock:
            n = len(self.api_keys)
            for _ in range(n):
                key = self.api_keys[self._index % n]
                self._index = (self._index + 1) % n
                if self._cooldown_until.get(key, 0.0) <= now:
                    return key
        return None

    def _cool(self, key: str, seconds: float) -> None:
        with self._lock:
            self._cooldown_until[key] = time.time() + seconds
        self._save_cooldowns()

    # --- API -------------------------------------------------------------
    def subscription(self, key: str) -> Optional[dict]:
        req = urllib.request.Request(
            f"{API_BASE}/user/subscription",
            headers={"xi-api-key": key, "User-Agent": "TopicGuardBot/1.0"},
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception:
            return None

    def synthesize(self, text: str, voice_id: Optional[str] = None, speed: Optional[float] = None) -> Optional[bytes]:
        """Озвучить текст. Пробует ключи по кругу, возвращает mp3 или None."""
        clean = re.sub(r"[*_`#~]", "", text or "").strip()
        if not clean:
            return None
        # ElevenLabs режет длинные запросы; наши реплики короткие, но подстрахуемся
        clean = clean[:900]

        payload = {
            "text": clean,
            "model_id": self.model,
            "voice_settings": {
                "stability": 0.45,
                "similarity_boost": 0.75,
                "style": 0.35,
                "use_speaker_boost": True,
                "speed": min(1.2, max(0.7, float(speed if speed is not None else self.speed))),
            },
        }
        data = json.dumps(payload).encode("utf-8")
        url = f"{API_BASE}/text-to-speech/{voice_id or self.voice_id}?output_format=mp3_44100_128"

        tried = 0
        while tried < len(self.api_keys):
            key = self._next_key()
            if not key:
                logger.warning("ElevenLabs: все ключи в кулдауне")
                return None
            tried += 1
            req = urllib.request.Request(
                url,
                data=data,
                headers={
                    "xi-api-key": key,
                    "Content-Type": "application/json",
                    "Accept": "audio/mpeg",
                    "User-Agent": "TopicGuardBot/1.0",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    audio = resp.read()
                if audio and len(audio) >= MIN_AUDIO_BYTES:
                    logger.info("ElevenLabs TTS ok (%s, key %s…)", self.model, key[:8])
                    return audio
                logger.warning(
                    "ElevenLabs вернул слишком короткое аудио (%s байт, key %s…)",
                    len(audio or b""),
                    key[:8],
                )
            except urllib.error.HTTPError as e:
                if e.code == 429:
                    self._cool(key, COOLDOWN_QUOTA)
                elif e.code in (401, 403):
                    self._cool(key, COOLDOWN_AUTH)
                elif e.code == 402:
                    self._cool(key, COOLDOWN_PAYMENT)
                else:
                    self._cool(key, 60.0)
                logger.debug("ElevenLabs HTTP %s on key %s…", e.code, key[:8])
            except Exception as e:
                self._cool(key, 60.0)
                logger.debug("ElevenLabs error on key %s…: %s", key[:8], e)
        return None