"""Smart message & topic classifier for TopicGuard Bot (Regex + Mercury-3 AI Judge).

Distinguishes useful drops (API keys, promo codes, account combos, free-trial guides,
referral/trial links, BINs, jailbreak prompts, skills, config files) from off-topic chatter,
questions, thanks, status comments, and flood media that belong in the General chat.
"""

from __future__ import annotations

import json
import logging
import re
import urllib.request
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger("topic_guard.classifier")


@dataclass
class ClassificationResult:
    keep: bool
    category: str
    reason: str


def utf16_index_to_py(text: str, idx: int) -> int:
    """Convert a UTF-16 code-unit offset (Telegram entities) to a Python index.

    Telegram entity ``offset``/``length`` values count UTF-16 code units, while
    Python slices ``str`` by code points.  Characters outside the BMP (emoji,
    etc.) occupy two UTF-16 units but only one Python index, so raw slicing
    shifts the snippet after any such character.
    """
    if idx <= 0:
        return 0
    units = 0
    for py_idx, ch in enumerate(text):
        units += len(ch.encode("utf-16-le")) // 2
        if units >= idx:
            return py_idx + 1
    return len(text)


def _parse_ai_verdict(answer: str) -> bool:
    """Map a model answer to keep (True) or move (False).

    The first token wins when it starts with MOVE/KEEP.  Otherwise the earliest
    standalone MOVE/KEEP occurrence wins.  Ties and unrecognised answers
    default to KEEP (keeping a message is the safer mistake).
    """
    normalized = (answer or "").strip().upper()

    first_match = re.match(r"\W*(\w+)", normalized)
    if first_match:
        first = first_match.group(1)
        if first.startswith("MOVE"):
            return False
        if first.startswith("KEEP"):
            return True

    move_match = re.search(r"\bMOVE\b", normalized)
    keep_match = re.search(r"\bKEEP\b", normalized)
    if move_match and keep_match:
        return keep_match.start() <= move_match.start()
    if move_match:
        return False
    if keep_match:
        return True

    logger.debug("AI returned unrecognised verdict %r; defaulting to KEEP", answer)
    return True


# Topic name patterns for auto-detecting topic purpose
TOPIC_CHAT_KEYWORDS = (
    "general",
    "общий",
    "чат",
    "chat",
    "флуд",
    "flood",
    "курилка",
    "общение",
    "оффтоп",
    "offtop",
    "беседка",
    "разговор",
    "болталк",
    "спам",
)

TOPIC_KEYS_KEYWORDS = (
    "ключ",
    "key",
    "keys",
    "раздач",
    "акк",
    "acc",
    "промо",
    "promo",
    "код",
    "code",
    "drop",
    "дроп",
    "cookie",
    "куки",
    "токен",
    "token",
)

TOPIC_TRIALS_KEYWORDS = (
    "триал",
    "trial",
    "free",
    "фри",
    "халяв",
    "абуз",
    "abuse",
    "схем",
    "инф",
    "info",
    "способ",
    "гайд",
    "guide",
    "метод",
    "method",
    "бин",
    "bin",
    "реф",
    "ref",
)

# Explicit keep prefixes/hashtags at the start or anywhere in the message
EXPLICIT_KEEP_RE = re.compile(
    r"(?:^|\s)(?:!keep|!оставить|#keep|#важно|#инфо|#ключ|#ключи|#триал|#гайд|#раздача|#drop|#info)\b",
    re.IGNORECASE,
)

# Known API key / token patterns
API_KEY_PATTERNS = [
    # OpenAI / Anthropic / DeepSeek / Cursor / OpenRouter / Inception / Generic sk- or sk_
    re.compile(r"\bsk[-_][a-zA-Z0-9_-]{16,}\b"),
    # Google Gemini / Firebase AIza
    re.compile(r"\bAIza[0-9A-Za-z_-]{25,}\b"),
    # Groq
    re.compile(r"\bgsk_[a-zA-Z0-9]{20,}\b"),
    # xAI Grok
    re.compile(r"\bxai-[a-zA-Z0-9]{20,}\b"),
    # HuggingFace
    re.compile(r"\bhf_[a-zA-Z0-9]{20,}\b"),
    # GitHub PATs
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[a-zA-Z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[a-zA-Z0-9_]{20,}\b"),
    # Perplexity
    re.compile(r"\bpplx-[a-zA-Z0-9]{20,}\b"),
    # Replicate
    re.compile(r"\br8_[a-zA-Z0-9]{20,}\b"),
    # Cerebras / Fireworks / NVIDIA NIM / Tavily / Brave / Together / Mistral
    re.compile(r"\b(?:csk|fw|nvapi|tvly)-[a-zA-Z0-9_-]{16,}\b"),
    # JWT tokens
    re.compile(r"\beyJ[a-zA-Z0-9_-]{15,}\.[a-zA-Z0-9_-]{10,}\.[a-zA-Z0-9_-]{10,}\b"),
    # UUID keys
    re.compile(
        r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
    ),
    # 32-64 char hex keys/tokens
    re.compile(r"\b[0-9a-fA-F]{32,64}\b"),
    # Multi-group license / promo / gift keys: XXXX-XXXX-XXXX or XXXXX-XXXXX-XXXXX
    re.compile(r"\b[A-Z0-9]{4,6}(?:-[A-Z0-9]{4,6}){2,5}\b"),
    # Scanner / endpoint reports (Base URL:, Endpoint:, ✅ Working:)
    re.compile(r"(?:Base\s*URL|Endpoint|✅\s*Working|⚡\s*\S+-сканер)\s*:", re.IGNORECASE),
]

# Email:password or login:password combo lines
EMAIL_PASS_RE = re.compile(
    r"\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+:[^\s:]{4,}\b"
)
LOGIN_PASS_RE = re.compile(
    r"(?:^|\n)\s*[a-zA-Z0-9_.-]{4,32}:[a-zA-Z0-9!@#$%^&*()_+=-]{5,64}\s*(?:$|\n)"
)

# Explicit key/promo/code label followed by value: "Промокод: FREE2026", "Key: abc..."
LABELED_CODE_RE = re.compile(
    r"(?:ключ(?:и)?|промо(?:код)?|код|key|api[ _-]?key|token|токен|promo(?:[ _-]?code)?|code|пароль|pass(?:word)?|invite|инвайт)\s*[:=-]\s*[`'\"]?[A-Za-z0-9_!@#$%^&*+-]{4,}[`'\"]?",
    re.IGNORECASE,
)

# URLs in plain text
URL_RE = re.compile(
    r"(?:https?://[^\s<>()\"']+|(?:t\.me|claude\.ai|cursor\.com|chat\.openai\.com|chatgpt\.com|perplexity\.ai|gemini\.google\.com)/[A-Za-z0-9_./?=&%-]+)",
    re.IGNORECASE,
)

# Card BIN pattern (6-8 digits or 6 digits + x/* mask) when accompanied by bin/card context
BIN_NUMBER_RE = re.compile(r"\b[3456]\d{5}(?:[\dxX*]{2,10})?\b")
BIN_CONTEXT_RE = re.compile(
    r"\b(?:бин|бины|bin|bins|vcc|карт[аыу]|card|zip|индекс|биллинг|billing)\b",
    re.IGNORECASE,
)

# Step-by-step guide structure (e.g. "1. ... 2. ..." or "1) ... 2) ..." or bullet list >= 3 items)
STEP_LIST_RE = re.compile(
    r"(?:(?:^|\n)\s*1[.)]\s+.+(?:\n\s*2[.)]\s+.+))|(?:(?:^|\n)\s*[•\-—]\s+.+(?:\n\s*[•\-—]\s+.+){2,})",
    re.MULTILINE,
)

# Free trial / method / giveaway informative keywords
TRIAL_INFO_KEYWORDS_RE = re.compile(
    r"\b(?:триал\w*|trial\w*|промокод\w*|promo\w*|абуз\w*|способ\w*|гайд\w*|метод\w*|инструкц\w*|обход\w*|рефк\w*|referral\w*|инвайт\w*|invite\w*|гостев\w+|guest\s*pass|бин\w*|bin\w*|раздают|раздач\w*|успейте\s+забрать|ворк\s+если\s+надо)\b",
    re.IGNORECASE,
)

# Question / help request / begging patterns
QUESTION_OR_BEGGING_START_RE = re.compile(
    r"^\s*(?:а\s+как|как\s+(?:сделать|получить|активировать|обойти|забрать|юзать|использовать|войти|регнуть)|где\s+(?:взять|найти|купить|достать)|у\s+кого\s+(?:есть|остался|работает)|кто\s+(?:знает|может|пробовал|даст|скинет)|почему\s+|зачем\s+|когда\s+(?:будет|некст|раздача)|подскажите|помогите|объясните|скиньте|дайте|дай\s+(?:ключ|личный|под\s+хайд|промо|бин|инфу)|можно\s+(?:ключ|личный|мне)|есть\s+у\s+кого|что\s+делать|какой\s+(?:бин|впн|прокси|антик)|работает\s+еще|еще\s+ворк|ворк\s*\?|актуально\s*\?)",
    re.IGNORECASE,
)

# Short reaction / claim / status comments ("спс", "забрал", "не ворк", "+rep", etc.)
SHORT_CHATTER_RE = re.compile(
    r"^\s*(?:спс\w*|спасиб\w*|пасиб\w*|сяб|благодар\w*|от\s+души|забрал\w*|взял\w*|успел\w*|не\s+успел\w*|все\s+забрали|уже\s+забрали|разобрали|убили|мертв\w*|сдох\w*|не\s+ворк\w*|ворк\w*|работает|не\s+работает|актуально|не\s+актуально|валид|невалид|лимит\w*|ошибк\w*|фродит|бан\w*|\+rep|\++|-+|имба|топ\w*|кайф\w*|легенд\w*|гой|красав\w*|лучш\w*|ждем|ждём|го\s+еще|еще|ещё|ага|пон|понял|ок|окей|да|нет|хз|лол|кек|ахах\w*|хаха\w*|[👍👎🔥❤️🙏👏😂😭💀🗿🎉💯✨🤝🫡]+)\s*[.!?,)*0-9]*\s*$",
    re.IGNORECASE,
)

AI_SYSTEM_PROMPT = (
    "Ты строгий ИИ-модератор тематических веток приватного Telegram-форума.\n"
    "В тематических ветках (например: 'Abuse and Keys', 'Инфа', 'Unreal Skills', 'Jailbreaks') "
    "разрешены ТОЛЬКО полезные публикации (дропы/контент по теме):\n"
    "- API-ключи, токены, промокоды, инвайты, аккаунты (login:pass), результаты сканеров эндпоинтов (Base URL, Key, Working models)\n"
    "- Рабочие способы, гайды, схемы абуза, инструкции по получению фри-триалов, рабочие БИНы карт\n"
    "- Полезные ссылки на сервисы/рефки/инструменты/скрипты/джейлбрейк-промпты/скиллы\n"
    "- Конкретные анонсы раздач (например: 'раздают 300M токенов в ZCode, успейте забрать', 'модель X ворк на этом эндпоинте')\n\n"
    "ВСЁ ОСТАЛЬНОЕ нужно переносить в General чат (ответить MOVE):\n"
    "- Любые вопросы ('а как сделать?', 'где взять?', 'почему ошибка?', 'у кого есть ключ?', 'можешь чекнуть карту?')\n"
    "- Просьбы и попрошайничество ('дайте ключ', 'скиньте в лс', 'наколдуй ключ')\n"
    "- Комментарии и реакции ('спс', 'забрал', 'не ворк', 'убили', 'у меня 4к мертвых', 'быстро сдох', 'хз')\n"
    "- Обычное общение, споры, шутки, флуд, обсуждение чужих постов без новой полезной инфы/ключей.\n\n"
    "Ответь СТРОГО одним словом: KEEP (если это полезный дроп/ключ/гайд/инфа) или MOVE (если это обсуждение/вопрос/флуд/оффтоп)."
)


def classify_topic_name(topic_name: str) -> str:
    """Return suggested mode ('smart', 'keys', 'trials', 'off') based on topic name."""
    if not topic_name:
        return "smart"
    low = topic_name.lower().strip()
    for kw in TOPIC_CHAT_KEYWORDS:
        if kw in low:
            return "off"
    has_keys = any(kw in low for kw in TOPIC_KEYS_KEYWORDS)
    has_trials = any(kw in low for kw in TOPIC_TRIALS_KEYWORDS)
    if has_keys and not has_trials:
        return "smart"
    if has_trials:
        return "smart"
    return "smart"


def _extract_entities_info(message: Dict[str, Any]) -> Dict[str, Any]:
    text = str(message.get("text") or message.get("caption") or "")
    entities = list(message.get("entities") or []) + list(
        message.get("caption_entities") or []
    )
    has_url_entity = False
    code_snippets: List[str] = []

    for ent in entities:
        etype = ent.get("type")
        if etype in ("url", "text_link"):
            has_url_entity = True
        elif etype in ("code", "pre"):
            offset = int(ent.get("offset", 0))
            length = int(ent.get("length", 0))
            start = utf16_index_to_py(text, offset)
            end = utf16_index_to_py(text, offset + length)
            if length > 0 and 0 <= start < len(text):
                code_snippets.append(text[start:end].strip())

    return {
        "has_url_entity": has_url_entity,
        "code_snippets": code_snippets,
    }


def has_key_or_credentials(text: str, code_snippets: Optional[List[str]] = None) -> Optional[str]:
    """Check if text or code blocks contain an API key, promo code, token, or login:pass."""
    if not text and not code_snippets:
        return None

    for pat in API_KEY_PATTERNS:
        if pat.search(text):
            return "api_key_or_token"

    if EMAIL_PASS_RE.search(text) or LOGIN_PASS_RE.search(text):
        return "credentials_combo"

    if LABELED_CODE_RE.search(text):
        return "labeled_code_or_key"

    if code_snippets:
        for snip in code_snippets:
            cleaned = snip.strip()
            if len(cleaned) >= 5 and re.match(r"^[A-Za-z0-9_.:/@!#$%^&*+-]{5,200}$", cleaned):
                return "code_block_token"
            if len(cleaned) >= 12 and ("\n" in cleaned or ":" in cleaned or "-" in cleaned):
                return "code_block_content"

    backtick_matches = re.findall(r"`([^`]{5,200})`", text)
    for bm in backtick_matches:
        cleaned = bm.strip()
        if re.match(r"^[A-Za-z0-9_.:/@!#$%^&*+-]{5,200}$", cleaned):
            return "inline_code_token"

    return None


def _is_question_or_chatter(text: str) -> bool:
    """Return True if the text is a short question, help request, thanks, or claim comment."""
    stripped = text.strip()
    if not stripped:
        return True

    if SHORT_CHATTER_RE.match(stripped):
        return True

    low = stripped.lower()
    if len(stripped) < 90:
        chatter_phrases = (
            "забрал",
            "спс",
            "спасибо",
            "пасиб",
            "не ворк",
            "уже не ворк",
            "все забрали",
            "убили",
            "не работает",
            "не подходит",
            "лимит",
            "ошибка",
            "дай личный",
            "можно ключ",
            "скинь в лс",
            "когда еще",
            "когда некст",
            "еще будет",
            "успел",
            "не успел",
        )
        if any(p in low for p in chatter_phrases) and not STEP_LIST_RE.search(stripped):
            return True

    if QUESTION_OR_BEGGING_START_RE.search(stripped) and not STEP_LIST_RE.search(stripped):
        return True

    if stripped.endswith("?") and "\n" not in stripped and len(stripped) < 140:
        return True

    return False


def has_trial_or_method_info(text: str, has_url_entity: bool = False) -> Optional[str]:
    """Check if text contains a free-trial link, referral, BIN info, or method guide."""
    if not text:
        return None

    stripped = text.strip()
    has_url = has_url_entity or bool(URL_RE.search(stripped))

    # 1. BIN + card context
    if BIN_NUMBER_RE.search(stripped) and BIN_CONTEXT_RE.search(stripped):
        if not QUESTION_OR_BEGGING_START_RE.search(stripped):
            return "bin_info"

    # 2. Step-by-step guide
    if STEP_LIST_RE.search(stripped) and len(stripped) >= 35:
        return "step_by_step_guide"

    # 3. URL present (referral link, trial page, tool link, promo link)
    if has_url:
        # Ignore internal chat links like https://t.me/c/<chat_id>/...
        if re.fullmatch(r"https?://t\.me/c/\d+/\d+(?:/\d+)?", stripped):
            return None
        without_urls = URL_RE.sub("", stripped).strip()
        if not without_urls:
            return "trial_or_resource_link"
        if not (
            without_urls.endswith("?")
            and len(without_urls) < 80
            and QUESTION_OR_BEGGING_START_RE.search(without_urls)
        ):
            return "trial_or_resource_link"

    # 4. Informative paragraph about free trials / promos / methods
    if TRIAL_INFO_KEYWORDS_RE.search(stripped):
        if not _is_question_or_chatter(stripped) and len(stripped) >= 45:
            return "trial_info_post"

    return None


def ai_classify_message(
    text: str,
    topic_name: str,
    mode: str,
    api_url: str,
    api_key: str,
    model: str = "mercury-3",
    reply_snippet: str = "",
    timeout: float = 6.0,
) -> Optional[bool]:
    """Query Mercury-3 (or compatible OpenAI endpoint) to decide KEEP (True) or MOVE (False).

    Returns None if the API call fails so caller can fall back to deterministic rules.
    """
    if not api_key or not api_url or not text.strip():
        return None

    user_content = f"Ветка: {topic_name or 'Ключи и Триалы'} (режим: {mode})\n"
    if reply_snippet:
        user_content += f"Ответ на сообщение: {reply_snippet[:200]}\n"
    user_content += f"Сообщение пользователя:\n{text[:1500]}"

    payload: Dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": AI_SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
        "max_tokens": 32,
        "temperature": 0.5,
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
        if not choices:
            return None
        ans = str((choices[0].get("message") or {}).get("content") or "").strip()
        return _parse_ai_verdict(ans)
    except Exception as e:
        logger.warning("AI classification fallback (error: %s)", e)
        return None


def classify_message(
    message: Dict[str, Any],
    mode: str = "smart",
    is_admin: bool = False,
    is_whitelisted: bool = False,
    admin_bypass: bool = False,
    topic_name: str = "",
    ai_enabled: bool = False,
    ai_api_url: str = "",
    ai_api_key: str = "",
    ai_model: str = "mercury-3",
) -> ClassificationResult:
    """Classify a forum topic message and decide whether to keep it or move it to General."""
    mode = (mode or "smart").lower().strip()

    if mode in ("off", "general"):
        return ClassificationResult(True, "topic_unwatched", "Ветка без фильтрации")

    if is_whitelisted:
        return ClassificationResult(True, "whitelisted", "Пользователь в белом списке")

    text = str(message.get("text") or message.get("caption") or "").strip()

    # Explicit keep tag (#важно, #инфо, !keep, etc.)
    if text and EXPLICIT_KEEP_RE.search(text):
        return ClassificationResult(True, "explicit_keep_tag", "Явная метка сохранения")

    if mode == "admin":
        if is_admin:
            return ClassificationResult(True, "admin_post", "Пост от администратора")
        return ClassificationResult(
            False,
            "non_admin_in_admin_topic",
            "Ветка только для постов администрации — обсуждение в General",
        )

    if is_admin and admin_bypass:
        return ClassificationResult(True, "admin_bypass", "Иммунитет администратора")

    # Pure flood media (stickers, voice, video_note, dice, animations without keys/guides)
    if (
        message.get("sticker")
        or message.get("voice")
        or message.get("video_note")
        or message.get("dice")
    ):
        return ClassificationResult(
            False,
            "flood_media",
            "Стикеры / голосовые / кружочки перенесены в General",
        )

    # Non-GIF document attachment (e.g. .txt with keys, cookies.json, configs, archives)
    if message.get("document") and not message.get("animation"):
        return ClassificationResult(True, "document_drop", "Файл / документ с данными")

    # Fast drop for empty media (e.g. photo/video/gif without any text caption)
    if not text:
        return ClassificationResult(
            False,
            "uncaptioned_media",
            "Медиа без описания / ключей перенесено в General",
        )

    # Fast drop for obvious ultra-short reactions ("спс", "забрал", "не ворк", "+rep", emojis)
    if SHORT_CHATTER_RE.match(text):
        return ClassificationResult(
            False,
            "short_chatter",
            "Короткий комментарий / реакция перенесена в General",
        )

    ent_info = _extract_entities_info(message)
    code_snippets = ent_info["code_snippets"]
    has_url_entity = ent_info["has_url_entity"]

    # Deterministic check for clear API keys, promo codes, tokens, credentials
    key_match = has_key_or_credentials(text, code_snippets=code_snippets)
    if key_match:
        if not (
            text.endswith("?")
            and len(text) < 120
            and QUESTION_OR_BEGGING_START_RE.search(text)
        ):
            return ClassificationResult(True, key_match, "Содержит ключ / код / эндпоинт")

    # Deterministic check for numbered/bulleted step-by-step guides (1) ... 2) ...)
    if STEP_LIST_RE.search(text) and len(text) >= 35 and not QUESTION_OR_BEGGING_START_RE.search(text):
        return ClassificationResult(True, "step_by_step_guide", "Пошаговый гайд / инструкция")

    # If AI classifier (Mercury-3) is configured and enabled, ask AI for context-aware verdict
    if ai_enabled and ai_api_key and mode != "keys":
        reply_msg = message.get("reply_to_message") or {}
        reply_snip = ""
        if isinstance(reply_msg, dict) and not reply_msg.get("forum_topic_created"):
            reply_snip = str(reply_msg.get("text") or reply_msg.get("caption") or "").strip()

        ai_decision = ai_classify_message(
            text=text,
            topic_name=topic_name,
            mode=mode,
            api_url=ai_api_url,
            api_key=ai_api_key,
            model=ai_model,
            reply_snippet=reply_snip,
        )
        if ai_decision is True:
            return ClassificationResult(
                True, "ai_keep", "ИИ (Mercury-3): полезная инфа / раздача по теме ветки"
            )
        if ai_decision is False:
            return ClassificationResult(
                False,
                "ai_move_general",
                "ИИ (Mercury-3): обсуждение / вопрос / оффтоп перенесён в General",
            )

    if mode == "keys":
        if has_url_entity or URL_RE.search(text):
            if not _is_question_or_chatter(text):
                return ClassificationResult(True, "key_link", "Содержит ссылку / раздачу")
        return ClassificationResult(
            False,
            "offtopic_in_keys",
            "Лишнее сообщение из ветки с ключами перенесено в General",
        )

    # Fallback heuristic for 'smart' and 'trials' modes when AI is disabled or unreachable
    trial_match = has_trial_or_method_info(text, has_url_entity=has_url_entity)
    if trial_match:
        return ClassificationResult(
            True, trial_match, "Содержит инфу по фри триалам / ссылку / гайд"
        )

    if (
        message.get("forward_from_chat") or message.get("forward_origin")
    ) and len(text) >= 60:
        return ClassificationResult(
            True, "forwarded_info", "Пересланный информационный пост"
        )

    return ClassificationResult(
        False,
        "chatter_or_offtopic",
        "Обсуждение / лишняя инфа перенесена в General чат",
    )
