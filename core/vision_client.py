"""Vision client for analyzing photos and GIFs and picking reactions/comments."""

from __future__ import annotations

import base64
import json
import logging
import re
import urllib.request
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("topic_guard.vision")

ALLOWED_REACTIONS = {
    "🔥", "👍", "👎", "❤️", "😂", "🗿", "😱", "💩", "🤡", "🤯",
    "🤨", "😴", "😭", "🍌", "🌭", "👀", "⚡", "🎉", "💅", "👾",
    "😍", "🥰", "👏", "🕊", "💔", "💯", "🤣", "🫡", "🤝", "🐳",
}

VISION_SYSTEM_PROMPT = """Ты — Кирюха, пушистый няшка-фембойчик, геймер и смотрящий чата. Живой участник, не ассистент.

Посмотри на изображение (фото или гифка) и сделай ДВЕ вещи.

1. Реакция: РОВНО 1 эмодзи под то, что реально в кадре, из списка:
[🔥, 👍, 👎, ❤️, 😂, 🗿, 😱, 💩, 🤡, 🤯, 🤨, 😴, 😭, 🍌, 🌭, 👀, ⚡, 🎉, 💅, 👾, 😍, 🥰, 👏, 💯, 🤣, 🫡]
Выбирай по смыслу кадра: смешное — 😂/🤣, милое — 😍/🥰, кринж — 🤡, внезапное — 🤯, тупняк — 🗿.

2. Комментарий: 1 короткая фраза, живая речь, как в чате.
   - ОБЯЗАТЕЛЬНО зацепись за КОНКРЕТНУЮ деталь ЭТОГО кадра: кто в нём, что делает, что написано, что на фоне, какое выражение лица/морды. Без детали комментарий не годится.
   - НЕ начинай с шаблонов: «фурр-фурр», «мурк», «ня», «мяу», «няшка», «лапки», «хвостик», «пушистик», «босс». Этих слов вообще может не быть — смысл важнее.
   - Не пиши приветствия, «как дела», «я вижу на картинке» и описания в стиле отчёта.
   - Не спрашивай, что нужно сделать, и не предлагай помощь.

Если ниже даны твои прошлые комментарии — новый должен быть заметно ДРУГИМ: другая мысль, другие слова, другая реакция.

ОТВЕТЬ СТРОГО В ФОРМАТЕ JSON:
{"reaction": "🔥", "comment": "..."}"""


class VisionClient:
    """Multimodal Vision client (Gemini 3.8 Flash on local cliproxyapi bridge)."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:12006",
        api_key: str = "",
        model: str = "gemini-3.8-flash-high",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model

    def analyze_image(
        self,
        image_bytes: bytes,
        caption: str = "",
        mime: str = "image/jpeg",
        timeout: float = 18.0,
        recent: Optional[List[str]] = None,
    ) -> Optional[Tuple[str, str]]:
        """Analyze image bytes with Vision and return (emoji_reaction, comment).

        recent — прошлые комментарии бота в этом чате: показываем их модели,
        чтобы она не выдавала одну и ту же реакцию на одинаковые гифки.
        """
        if not image_bytes or not self.api_key:
            return None

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"

        user_text = "Что на картинке? Выбери эмодзи и дай свой комментарий."
        if caption:
            user_text += f"\nПодпись к картинке от автора: {caption}"
        if recent:
            past = "\n".join(f"- {r}" for r in list(recent)[-5:])
            user_text += (
                "\n\nТВОИ ПРОШЛЫЕ КОММЕНТАРИИ (ни в коем случае не повторяй их "
                f"формулировки и не ставь ту же реакцию):\n{past}"
            )

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": VISION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {"type": "image_url", "image_url": {"url": data_uri}},
                    ],
                },
            ],
            "max_tokens": 140,
            "temperature": 0.95,
            "top_p": 0.95,
        }

        url = f"{self.base_url}/v1/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "TopicGuardBot/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw_reply = str(
                (data.get("choices") or [{}])[0].get("message", {}).get("content") or ""
            ).strip()
            if not raw_reply:
                return None

            # Clean markdown code blocks if any
            clean = raw_reply
            if clean.startswith("```"):
                clean = re.sub(r"^```(?:json)?\s*", "", clean)
                clean = re.sub(r"\s*```$", "", clean)
            clean = clean.strip()

            parsed = json.loads(clean)
            reaction = str(parsed.get("reaction") or "").strip()
            comment = str(parsed.get("comment") or "").strip()

            # Ensure reaction is a single valid emoji
            if reaction not in ALLOWED_REACTIONS:
                for em in ALLOWED_REACTIONS:
                    if em in reaction:
                        reaction = em
                        break
                else:
                    reaction = "👀"

            return reaction, comment
        except Exception as e:
            logger.debug("Vision analysis error: %s", e)
            return None
