"""Client for muse2api bridge and neural Voice/TTS: generates images, videos, and voice messages (ГС)."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("topic_guard.muse")

CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")

PHOTO_INTENT_RE = re.compile(
    r"(?i)(?:^/photo(?:@\w+)?\s+(.+)|(?:кирюх\w*|киря|кир|@\w+)\s*[,:]?\s*(?:нарисуй\w*|сген\w*|изобрази|сделай (?:мне\s+)?(?:фото|картинк\w*|арт\w*|пикч\w*)|скинь (?:фото|картинку|арт))\s+(.+)|(?:нарисуй|нарисуешь)\s+(?:мне\s+)?(.+)|(?:сген\w*|создай|сделай)\s+(?:мне\s+)?(?:фото|картинк\w*|арт\w*|изображени\w*)\s+(.+))",
)
VIDEO_INTENT_RE = re.compile(
    r"(?i)(?:^/video(?:@\w+)?\s+(.+)|(?:кирюх\w*|киря|кир|@\w+)\s*[,:]?\s*(?:сген\w*|сделай|создай|запил\w*|замут\w*|забаха\w*|сними|хочу)\s+(?:мне\s+)?(?:видео|видос\w*)\s*(.*)|(?:кирюх\w*|киря|кир|@\w+)\s*[,:]?\s*(?:видео|видос\w*)\s+(?:мне\s+)?(?:сген\w*|сделай|создай|запил\w*)\s*(.*)|(?:сген\w*|сделай|создай)\s+(?:мне\s+)?(?:видео|видос\w*)\s+(.+))",
)
VOICE_INTENT_RE = re.compile(
    r"(?i)(?:^/voice(?:@\w+)?\s+(.+)|(?:кирюх\w*|киря|кир|@\w+)\s*[,:]?\s*(?:скажи голосовым|отправь гс|скинь гс|запиши гс|озвучь|скажи голосом|наговори|наболтай)\s*[:,-]?\s*(.+)|(?:кирюх\w*|киря|кир|@\w+)\s*[,:]?\s*(?:голосовым|гс)\s*[:,-]?\s*(?:скажи|наговори|озвучь)\s*(.+))",
)

NON_IMAGE_WORDS = re.compile(
    r"(?i)\b(?:пароль|текст|код|скрипт|промпт|ключ|токен|число|символ)\b",
)

DICTIONARY = {
    "фури": "furry",
    "фурри": "furry",
    "фембойчика": "femboy character",
    "фембойчик": "femboy character",
    "фембоя": "femboy",
    "фембой": "femboy",
    "автора": "author boy",
    "автор": "author boy",
    "андрю": "cute anime boy Andrew with cat ears",
    "андрюху": "cute anime boy Andrew with cat ears",
    "лисичку": "fluffy fox",
    "лисичка": "fluffy fox",
    "лиса": "fox",
    "лисы": "foxes",
    "лису": "fox",
    "лисой": "fox",
    "волка": "wolf",
    "волк": "wolf",
    "котика": "cute cat",
    "котик": "cute cat",
    "кошка": "cat",
    "кошку": "cat",
    "девушку": "anime girl",
    "девушка": "anime girl",
    "парня": "anime boy",
    "парень": "anime boy",
    "аниме": "anime style",
    "бежит": "running",
    "танцует": "dancing",
    "в лесу": "in a forest",
    "в комнате": "in a room",
    "гоя": "a guy",
    "гое": "a guy",
    "гой": "a guy",
    "гою": "a guy",
    "гоем": "a guy",
    "человек": "man",
    "человека": "man",
    "парня": "a guy",
    "парень": "a guy",
    "подписку": "subscription",
    "подписка": "subscription",
    "подписки": "subscription",
    "купил": "bought",
    "купила": "bought",
    "год": "a year",
    "года": "a year",
    "годовой": "annual",
    "кот": "cat",
    "кота": "cat",
    "котом": "cat",
    "коту": "cat",
    "играющим": "playing",
    "играет": "playing",
    "играющего": "playing",
    "утка": "duck",
    "утку": "duck",
    "уткой": "duck",
    "свитч": "Nintendo Switch",
    "свитче": "Nintendo Switch",
    "ниндтендо": "Nintendo",
    "фембоем": "femboy",
    "фембоем~": "femboy",
    "фембою": "femboy",
    "котиком": "a kitten",
    "котику": "a kitten",
    "котике": "a kitten",
    "котики": "kittens",
    "котёнком": "a kitten",
    "котенком": "a kitten",
    "котёнок": "a kitten",
    "котенок": "a kitten",
    "котёнку": "a kitten",
    "котенку": "a kitten",
    "кошкой": "a cat",
    "котом": "a cat",
    "коту": "a cat",
    "парнем": "a guy",
    "парню": "a guy",
    "девушкой": "a girl",
    "девочку": "a girl",
    "играющим": "playing",
    "играющей": "playing",
    "играющего": "playing",
    "играет": "playing",
    "играть": "playing",
    "сидит": "sitting",
    "лежит": "lying",
    "спит": "sleeping",
    "едет": "driving",
    "летит": "flying",
    "танцуют": "dancing",
    "обнимает": "hugging",
    "улыбается": "smiling",
    "смеётся": "laughing",
    "смеется": "laughing",
    "в": "in",
    "на": "on",
    "с": "with",
    "о": "about",
    "и": "and",
}

# Русские окончания, которые мешают найти слово в словаре: «котиком» -> «котик».
# Срезаем самое длинное совпадение, но не короче 3 символов основы.
_RU_ENDINGS = (
    "ами", "ями", "ого", "его", "ому", "ему", "ыми", "ими", "ая", "яя", "ое", "ее",
    "ые", "ие", "ой", "ей", "ам", "ям", "ах", "ях", "ом", "ем", "ой", "ей", "ую", "юю",
    "ов", "ев", "ий", "ый", "а", "я", "о", "е", "у", "ю", "ы", "и", "ь",
)
_STOPWORD_KEYS = {"в", "на", "с", "о", "и", "in", "on", "with", "about", "and"}


def _stem_lookup(word: str) -> Optional[str]:
    """Словарная форма с учётом падежей: «фембоем» -> «фембой» -> femboy."""
    w = word.strip().lower()
    if not w:
        return None
    direct = DICTIONARY.get(w)
    if direct:
        return direct
    for ending in _RU_ENDINGS:
        if w.endswith(ending) and len(w) - len(ending) >= 3:
            stem = w[: -len(ending)]
            for suffix in ("", "а", "я", "о", "е", "и", "ы", "к", "ч", "ц"):
                hit = DICTIONARY.get(stem + suffix)
                if hit:
                    return hit
    return None


# Слова-маркеры типа медиа, которые перебивают любые опечатки в глаголе
VIDEO_WORD_RE = re.compile(r"(?i)\b(?:видео|видос\w*|видик|клип|анимаци\w*)\b")
VOICE_WORD_RE = re.compile(r"(?i)\b(?:голосов\w*|гс|озвуч\w*|наговори\w*)\b")
PHOTO_WORD_RE = re.compile(r"(?i)\b(?:фото|картинк\w*|арт\w*|пикч\w*|изображени\w*|нарису\w*)\b")
# «Сгенерируй/сделай/замути» в любой форме, включая опечатки вроде «сгенирр руй»
MAKE_VERB_RE = re.compile(
    r"(?i)\b(?:сген\w*|созда\w*|сдела\w*|сделай|замут\w*|запил\w*|нарису\w*|сними|изобраз\w*)\b"
)


REFINE_RE = re.compile(
    r"(?i)\b(?:а не|вместо|не так|нормальн\w*|заново|по-?другому|"
    r"сделай\s+(?:ещ[её]|друг\w+))\b"
)
# Глаголы правки: сами по себе означают «перегенерируй прошлое»
REFINE_VERB_RE = re.compile(r"(?i)\b(?:передел\w*|перерису\w*|исправ\w*)\b")
# Короткие команды-правки: «Добавь пчел», «Убери утку»
REFINE_EDIT_RE = re.compile(r"(?i)\b(?:добав\w*|убери|убрать|замен\w*|поменя\w*)\b")


def detect_media_intent(
    text: str, last_kind: Optional[str] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Detect if user asks for photo, video, or voice message.

    last_kind — тип последней генерации в этом чате. Если человек не назвал
    медиа явно, но правит прошлый результат («сделай нормального гоя а не
    утку»), считаем это запросом того же типа.

    Returns: ("photo"|"video"|"voice"|None, prompt_text)
    """
    if not text:
        return None, None
    stripped = text.strip()

    mv = VIDEO_INTENT_RE.search(stripped)
    if mv:
        p = next((g for g in mv.groups() if g), "")
        return "video", p.strip()

    ms = VOICE_INTENT_RE.search(stripped)
    if ms:
        p = next((g for g in ms.groups() if g), "")
        return "voice", p.strip()

    # «видео» в тексте перебивает фото-паттерн: «сген-руй видео» ≠ картинка
    if video_hint := VIDEO_WORD_RE.search(stripped):
        prompt = _strip_media_words(stripped)
        if video_hint and MAKE_VERB_RE.search(stripped) and prompt:
            return "video", prompt

    mp = PHOTO_INTENT_RE.search(stripped)
    if mp:
        p = next((g for g in mp.groups() if g), "")
        if p and not NON_IMAGE_WORDS.search(p):
            return "photo", p.strip()

    # Опечатки ломают регулярки — смотрим на слова напрямую.
    # Порядок важен: видео проверяем раньше фото, иначе «видео» уедет в картинку.
    if MAKE_VERB_RE.search(stripped):
        prompt = _strip_media_words(stripped)
        if VIDEO_WORD_RE.search(stripped):
            return "video", prompt
        if VOICE_WORD_RE.search(stripped):
            return "voice", prompt
        if PHOTO_WORD_RE.search(stripped):
            return "photo", prompt

    # Правка прошлой генерации без слова «фото/видео»:
    # «Кирюха, сделай нормального гоя а не утку», «Добавь пчел».
    if last_kind in ("photo", "video"):
        words = stripped.split()
        short_command = len(words) <= 4 and "?" not in stripped and REFINE_EDIT_RE.search(stripped)
        corrects = (
            (MAKE_VERB_RE.search(stripped) and REFINE_RE.search(stripped))
            or REFINE_VERB_RE.search(stripped)
            or short_command
        )
        if corrects:
            prompt = _strip_media_words(stripped)
            if prompt:
                return last_kind, prompt

    return None, None


_MEDIA_NOISE_RE = re.compile(
    r"(?i)\b(?:кирюх\w*|кирюш\w*|киря|кир|@\w+|пожалуйста|пж|мне|"
    r"сген\w*|созда\w*|сдела\w*|сделай|замут\w*|запил\w*|нарису\w*|изобраз\w*|сними|"
    r"видео|видос\w*|видик|клип|анимаци\w*|фото|картинк\w*|арт\w*|пикч\w*|изображени\w*|"
    r"голосов\w*|гс|озвуч\w*|наговори\w*)\b",
)


def _strip_media_words(text: str) -> str:
    """Убрать служебные слова, оставив только описание сцены."""
    cleaned = _MEDIA_NOISE_RE.sub(" ", text or "")
    cleaned = re.sub(r"[?!.,;:\"«»]+", " ", cleaned)
    return " ".join(cleaned.split()).strip()


VIDEO_SOUND_RE = re.compile(r"(?im)^\s*(?:звук|sound)\s*[:—-]\s*(.+)$")
VIDEO_SECONDS_RE = re.compile(r"(?i)(\d{1,2})\s*(?:с|сек|секунд|seconds?)\b")
VIDEO_NO_COMMENT_RE = re.compile(r"(?im)^\s*без\s+коммент\w*.*$")
VIDEO_CLIP_LINE_RE = re.compile(r"(?im)^\s*(?:ровно\s+)?(?:один|1|\d+)\s+(?:коротк\w*\s+)?клип\w*.*$")
VIDEO_LEN_LINE_RE = re.compile(r"(?im)длину\s+не\s+меняй[^\n]*")


def extract_video_directives(text: str) -> Tuple[str, Optional[str], Optional[int], int]:
    """Вытащить из текста параметры Muse: звук, длину, число клипов.

    «Сгенерируй видео: Anime girl...\nЗвук: тихий фон без слов.\nРовно один
    короткий клип (~10 с)» — инструкции не должны попадать в визуальный промпт,
    иначе переводчик превращает их в описание сцены.
    """
    raw = text or ""
    sound: Optional[str] = None
    m = VIDEO_SOUND_RE.search(raw)
    if m:
        sound = m.group(1).strip().rstrip(".")
        raw = VIDEO_SOUND_RE.sub("", raw)

    seconds: Optional[int] = None
    m2 = VIDEO_SECONDS_RE.search(raw)
    if m2:
        try:
            seconds = max(3, min(30, int(m2.group(1))))
        except (TypeError, ValueError):
            seconds = None

    clips = 1
    m3 = re.search(r"(?im)^\s*(?:ровно\s+)?(один|1|\d+)\s+(?:коротк\w*\s+)?клип\w*", raw)
    if m3:
        val = m3.group(1).lower()
        try:
            clips = 1 if val == "один" else int(val)
        except ValueError:
            clips = 1
    clips = max(1, min(6, clips))

    raw = VIDEO_NO_COMMENT_RE.sub("", raw)
    raw = VIDEO_CLIP_LINE_RE.sub("", raw)
    raw = VIDEO_LEN_LINE_RE.sub("", raw)
    cleaned = " ".join(raw.split()).strip(" ,;.-—")
    return cleaned, sound, seconds, clips


def translate_prompt_llm(
    prompt: str,
    kind: str = "image",
    api_url: str = "",
    api_key: str = "",
    model: str = "mercury-3",
    timeout: float = 6.0,
) -> Optional[str]:
    """Translate and enrich user visual prompt into detailed English Stable Diffusion / Flux prompt via LLM."""
    if not prompt or not api_url or not api_key:
        return None
    system = (
        "You are an expert prompt engineer for AI visual diffusion models (Flux / Midjourney / Stable Diffusion). "
        "Convert the user visual prompt into a vivid, descriptive English prompt. "
        "Preserve character names, ethnicity, age, gender, body type, specific actions, clothing, settings, and lighting. "
        "Do NOT add anime/manga/cartoon styling unless the user explicitly asked for it. "
        "Do NOT censor, soften, moralize, or omit details the user wrote. "
        "Output ONLY the English prompt text, no quotes, no markdown, no conversational filler."
    )
    user_msg = f"Generate {kind} prompt for: {prompt}"
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        "max_tokens": 140,
        "temperature": 0.7,
    }
    if "mercury" in model.lower() or "inceptionlabs" in api_url.lower():
        payload["reasoning_effort"] = "instant"

    req = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "User-Agent": "TopicGuardBot/1.0",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        choices = body.get("choices") or []
        if choices:
            raw = str((choices[0].get("message") or {}).get("content") or "").strip()
            raw = re.sub(r"^```(?:json|markdown)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip().strip('"«»').strip()
            if raw and len(raw) >= 10 and not CYRILLIC_RE.search(raw[:60]):
                return raw
    except Exception as e:
        logger.debug("LLM prompt translation failed: %s", e)
    return None


def optimize_visual_prompt(
    prompt: str,
    kind: str = "image",
    api_url: str = "",
    api_key: str = "",
    model: str = "mercury-3",
) -> str:
    """Optimize prompt for Muse diffusion model by expanding Russian terms into English."""
    if not prompt:
        return "cute fluffy furry femboy character" if kind == "video" else "cute fluffy anime character"
    if not CYRILLIC_RE.search(prompt):
        return prompt

    # 1. Try fast LLM translation first (takes ~0.4s and understands all Russian context and names)
    if api_url and api_key:
        llm_opt = translate_prompt_llm(prompt, kind=kind, api_url=api_url, api_key=api_key, model=model)
        if llm_opt:
            logger.info("LLM optimized %s prompt %r -> %r", kind, prompt, llm_opt)
            return llm_opt

    low = prompt.lower()
    words = low.split()
    translated_words: List[str] = []
    for w in words:
        cleaned = re.sub(r"[^\w-]", "", w)
        if not cleaned:
            continue
        trans = _stem_lookup(cleaned)
        if trans:
            translated_words.append(trans)
        elif not CYRILLIC_RE.search(cleaned):
            translated_words.append(cleaned)

    # Ничего не перевели — отдаём как есть, Muse справится лучше с оригиналом,
    # чем с обрубком из предлогов.
    known = [w for w in translated_words if w.lower() not in _STOPWORD_KEYS]
    if not known:
        return f"{prompt}, smooth animation, 4k" if kind == "video" else prompt

    # Предлог в начале («with femboy...») сбивает модель: она цепляется за «with»
    # и рисует пустоту. Начинаем с подлежащего.
    while translated_words and translated_words[0].lower() in _STOPWORD_KEYS:
        translated_words.pop(0)
    # Хвостовой предлог — та же беда с другой стороны: «Cat in» вместо сцены.
    while translated_words and translated_words[-1].lower() in _STOPWORD_KEYS:
        translated_words.pop()
    base = " ".join(translated_words)
    if base:
        base = base[:1].upper() + base[1:]

    if kind == "video":
        # Модель склонна терять персонажа и рисовать только объект действия,
        # поэтому подлежащее дублируем отдельной фразой-якорем.
        anchor = _subject_anchor(base)
        tail = f"{anchor}, " if anchor else ""
        return f"{base}, {tail}smooth animation, cinematic lighting, cute aesthetic, 4k"
    return f"{base}, high quality, detailed, anime aesthetic"


def _subject_anchor(base: str) -> str:
    """Короткая фраза, заставляющая модель держать персонажа в кадре."""
    low = base.lower()
    if any(word in low for word in ("femboy", "anime boy", "anime girl", "a guy", "author boy")):
        return "the character clearly visible in frame"
    if "kitten" in low or "cat" in low:
        return "the cat clearly visible in frame"
    if "fox" in low:
        return "the fox clearly visible in frame"
    return ""


async def _synthesize_edge_tts(text: str, voice: str = "ru-RU-DmitryNeural") -> bytes:
    import edge_tts
    clean = re.sub(r"[*_`#~]", "", text).strip()
    tts = edge_tts.Communicate(clean, voice)
    chunks: List[bytes] = []
    async for chunk in tts.stream():
        if chunk["type"] == "audio":
            chunks.append(chunk["data"])
    return b"".join(chunks)


class MuseClient:
    """Client for local muse2api server with Edge-TTS neural voice synthesis."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:12001",
        default_voice: str = "lumi",
        llm_api_url: str = "",
        llm_api_key: str = "",
        llm_model: str = "mercury-3",
        elevenlabs: Optional[object] = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.default_voice = default_voice
        self.llm_api_url = llm_api_url
        self.llm_api_key = llm_api_key
        self.llm_model = llm_model
        # ElevenLabs-клиент (core.elevenlabs.ElevenLabsClient) или None.
        # Голос Кирюхи живёт там: Edge-TTS звучит как диктор, а не фурри.
        self.elevenlabs = elevenlabs

    def generate_image(self, prompt: str, timeout: float = 300.0) -> Optional[bytes]:
        url = f"{self.base_url}/v1/images/generations"
        opt_prompt = optimize_visual_prompt(
            prompt,
            kind="image",
            api_url=self.llm_api_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
        )
        payload = {"prompt": opt_prompt, "n": 1}
        # Muse может рендерить минуты; второй такой же заход только удваивал ожидание
        for attempt in range(1):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data") or []
                if not items:
                    return None
                b64 = items[0].get("b64_json")
                if b64:
                    return base64.b64decode(b64)
            except Exception as e:
                if attempt == 0:
                    import time
                    time.sleep(2)
                    continue
                logger.warning("Muse image generation failed for %r: %s", prompt, e)
        return None

    def generate_video(
        self,
        prompt: str,
        timeout: float = 330.0,
        sound: Optional[str] = None,
        seconds: Optional[int] = None,
        clips: Optional[int] = None,
    ) -> Optional[bytes]:
        url = f"{self.base_url}/v1/videos/generations"
        clean_prompt, ex_sound, ex_seconds, ex_clips = extract_video_directives(prompt)
        sound = sound or ex_sound
        seconds = seconds or ex_seconds
        clips = max(1, min(6, clips or ex_clips or 1))
        opt_prompt = optimize_visual_prompt(
            clean_prompt or prompt,
            kind="video",
            api_url=self.llm_api_url,
            api_key=self.llm_api_key,
            model=self.llm_model,
        )
        payload: Dict[str, Any] = {"prompt": opt_prompt, "clips": clips}
        if sound:
            payload["sound"] = sound
        if seconds:
            payload["seconds"] = seconds
        for attempt in range(1):
            req = urllib.request.Request(
                url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                items = data.get("data") or []
                if not items:
                    return None
                b64 = items[0].get("b64_json")
                if b64:
                    return base64.b64decode(b64)
            except Exception as e:
                if attempt == 0:
                    import time
                    time.sleep(2)
                    continue
                logger.warning("Muse video generation failed for %r: %s", prompt, e)
        return None

    def generate_voice(
        self,
        text: str,
        voice: Optional[str] = None,
        timeout: float = 60.0,
        speed: Optional[float] = None,
    ) -> Optional[bytes]:
        clean_text = re.sub(r"[*_`#~]", "", text).strip()
        if not clean_text:
            return None

        # 1. ElevenLabs — живой няшный голос, если есть рабочий ключ
        if self.elevenlabs is not None:
            try:
                audio = self.elevenlabs.synthesize(clean_text, voice_id=voice, speed=speed)
                if audio:
                    return audio
            except Exception as e:
                logger.warning("ElevenLabs error: %s, falling back", e)

        # 2. Русский текст — Edge-TTS (0.3 с, 100% надёжность)
        if CYRILLIC_RE.search(clean_text):
            try:
                return asyncio.run(_synthesize_edge_tts(clean_text))
            except Exception as e:
                logger.warning("Edge-TTS error: %s, falling back to Muse", e)

        # English or fallback: use Muse
        url = f"{self.base_url}/v1/audio/speech"
        v = voice or self.default_voice
        payload = {"input": clean_text, "voice": v}
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception as e:
            logger.warning("Muse voice synthesis failed for %r: %s", clean_text, e)
        return None
