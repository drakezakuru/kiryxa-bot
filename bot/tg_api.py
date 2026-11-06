"""Telegram Bot API client using Python stdlib (urllib.request) + MockTelegramAPI for testing."""

from __future__ import annotations

import json
import logging
import random
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("topic_guard.tg_api")


class TelegramError(Exception):
    def __init__(self, method: str, description: str, error_code: int = 0) -> None:
        super().__init__(f"Telegram API error in {method} ({error_code}): {description}")
        self.method = method
        self.description = description
        self.error_code = error_code


def sniff_image_type(data: bytes) -> "tuple[str, str]":
    """Return ``(mime, extension)`` for image bytes based on well-known magic numbers."""
    if data.startswith(b"\x89PNG"):
        return "image/png", "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", "jpg"
    if data.startswith(b"GIF8"):
        return "image/gif", "gif"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp", "webp"
    # Telegram accepts unknown image payloads as JPEG; keep the historical fallback.
    return "image/jpeg", "jpg"


def sniff_video_type(data: bytes) -> "tuple[str, str]":
    """Return ``(mime, extension)`` for video bytes based on well-known magic numbers."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        if data[8:12] == b"qt  ":
            return "video/quicktime", "mov"
        return "video/mp4", "mp4"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBM":
        return "video/webm", "webm"
    return "video/mp4", "mp4"


def inline_btn(
    text: str,
    callback_data: Optional[str] = None,
    url: Optional[str] = None,
) -> Dict[str, Any]:
    btn: Dict[str, Any] = {"text": text}
    if callback_data is not None:
        btn["callback_data"] = callback_data
    if url is not None:
        btn["url"] = url
    return btn


def inline_keyboard(rows: List[List[Dict[str, Any]]]) -> Dict[str, Any]:
    return {"inline_keyboard": rows}


class TelegramAPI:
    """Synchronous Telegram Bot API client over HTTPS."""

    # Retry policy for transient Telegram API failures.
    MAX_ATTEMPTS = 3  # total attempts for HTTP 429 (initial request + 2 retries)
    MAX_SERVER_RETRIES = 1  # one retry for HTTP 5xx
    MAX_TOTAL_WAIT = 90.0  # total backoff budget per call, seconds
    MAX_SINGLE_WAIT = 60.0  # never sleep longer than this at once
    DEFAULT_RETRY_AFTER = 3.0  # fallback when parameters.retry_after is absent
    RATE_LIMIT_JITTER = 0.5  # random extra delay added to retry_after

    def __init__(self, token: str, base_url: str = "https://api.telegram.org") -> None:
        self.token = token.strip()
        self.base_url = f"{base_url.rstrip('/')}/bot{self.token}"

    def call(self, method: str, payload: Optional[Dict[str, Any]] = None, timeout: int = 35) -> Any:
        url = f"{self.base_url}/{method}"
        clean_payload = {k: v for k, v in (payload or {}).items() if v is not None}
        data = json.dumps(clean_payload).encode("utf-8")

        def make_request() -> urllib.request.Request:
            # ``data`` is immutable bytes, so the same body can safely be resent
            # on every retry attempt.
            return urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

        return self._execute_with_retry(method, make_request, timeout)

    @staticmethod
    def _http_error_details(
        exc: urllib.error.HTTPError,
    ) -> "tuple[str, int, Dict[str, Any]]":
        """Consume an HTTPError and return (description, error_code, parsed body)."""
        raw = ""
        try:
            raw = exc.read().decode("utf-8", errors="replace")
        except Exception:
            pass
        finally:
            try:
                exc.close()
            except Exception:
                pass

        try:
            err_json = json.loads(raw)
            if not isinstance(err_json, dict):
                raise ValueError("error body is not an object")
            desc = str(err_json.get("description", raw or str(exc)))
            code = int(err_json.get("error_code", exc.code))
        except Exception:
            err_json = {}
            desc = raw or str(exc)
            code = int(exc.code)
        return desc, code, err_json

    @classmethod
    def _retry_after(cls, err_json: Dict[str, Any]) -> float:
        params = err_json.get("parameters")
        raw = params.get("retry_after") if isinstance(params, dict) else None
        if raw is None:
            return cls.DEFAULT_RETRY_AFTER
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return cls.DEFAULT_RETRY_AFTER
        return max(0.0, value)

    def _execute_with_retry(
        self,
        method: str,
        request_factory: Callable[[], urllib.request.Request],
        timeout: int,
    ) -> Any:
        """Perform a request, retrying HTTP 429 and 5xx responses with bounded backoff."""
        total_wait = 0.0
        server_retries = 0
        attempt = 0

        while True:
            attempt += 1
            req = request_factory()
            try:
                with urllib.request.urlopen(req, timeout=timeout) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                break
            except urllib.error.HTTPError as exc:
                status = int(exc.code)
                desc, code, err_json = self._http_error_details(exc)

                if status == 429:
                    retry_after = self._retry_after(err_json)
                    if attempt >= self.MAX_ATTEMPTS:
                        raise TelegramError(
                            method,
                            "rate limited after "
                            f"{attempt} attempts (retry_after={retry_after:g}s): {desc}",
                            code,
                        ) from exc

                    remaining = self.MAX_TOTAL_WAIT - total_wait
                    if remaining <= 0:
                        raise TelegramError(
                            method,
                            "rate-limit wait budget "
                            f"({self.MAX_TOTAL_WAIT:g}s) exhausted after {attempt} "
                            f"attempts (retry_after={retry_after:g}s): {desc}",
                            code,
                        ) from exc

                    delay = min(
                        retry_after + random.uniform(0.0, self.RATE_LIMIT_JITTER),
                        self.MAX_SINGLE_WAIT,
                        remaining,
                    )
                    logger.warning(
                        "Telegram API %s rate limited (HTTP 429, retry_after=%gs); "
                        "retrying in %.2fs (attempt %d/%d)",
                        method,
                        retry_after,
                        delay,
                        attempt,
                        self.MAX_ATTEMPTS,
                    )
                    time.sleep(delay)
                    total_wait += delay
                    continue

                if 500 <= status < 600:
                    if server_retries >= self.MAX_SERVER_RETRIES:
                        raise TelegramError(method, desc, code) from exc

                    remaining = self.MAX_TOTAL_WAIT - total_wait
                    if remaining <= 0:
                        raise TelegramError(
                            method,
                            "wait budget "
                            f"({self.MAX_TOTAL_WAIT:g}s) exhausted after {attempt} "
                            f"attempts: {desc}",
                            code,
                        ) from exc

                    delay = min(random.uniform(1.0, 2.0), remaining)
                    server_retries += 1
                    logger.warning(
                        "Telegram API %s server error (HTTP %d); retrying in %.2fs "
                        "(attempt %d/%d)",
                        method,
                        status,
                        delay,
                        attempt,
                        self.MAX_ATTEMPTS,
                    )
                    time.sleep(delay)
                    total_wait += delay
                    continue

                # Other 4xx (and anything else) is not retryable.
                raise TelegramError(method, desc, code) from exc
            except Exception as exc:
                raise TelegramError(method, str(exc), 0) from exc

        if not body.get("ok"):
            raise TelegramError(
                method,
                str(body.get("description", "Unknown error")),
                int(body.get("error_code", 0)),
            )
        return body.get("result")

    def get_me(self) -> Dict[str, Any]:
        return self.call("getMe")

    def get_updates(
        self,
        offset: Optional[int] = None,
        timeout: int = 25,
        allowed_updates: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        return self.call(
            "getUpdates",
            {
                "offset": offset,
                "timeout": timeout,
                "allowed_updates": allowed_updates
                or ["message", "edited_message", "callback_query"],
            },
            timeout=timeout + 10,
        )

    def send_message(
        self,
        chat_id: int | str,
        text: str,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None,
        disable_web_page_preview: bool = True,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "message_thread_id": message_thread_id,
            "parse_mode": parse_mode,
            "reply_markup": reply_markup,
            "link_preview_options": {"is_disabled": disable_web_page_preview},
        }
        if reply_to_message_id is not None:
            payload["reply_parameters"] = {
                "message_id": reply_to_message_id,
                "allow_sending_without_reply": True,
            }

        try:
            return self.call("sendMessage", payload)
        except TelegramError as e:
            # Fallback 1: if HTML parse error, retry as plain text without HTML
            if parse_mode and ("can't parse entities" in e.description.lower() or "parse" in e.description.lower()):
                payload["parse_mode"] = None
                return self.call("sendMessage", payload)
            # Fallback 2: if reply error, retry without reply parameters
            if reply_to_message_id and "replied" in e.description.lower():
                payload.pop("reply_parameters", None)
                return self.call("sendMessage", payload)
            raise e

    def send_sticker(
        self,
        chat_id: int | str,
        sticker: str,
        message_thread_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.call(
            "sendSticker",
            {
                "chat_id": chat_id,
                "sticker": sticker,
                "message_thread_id": message_thread_id,
                "reply_to_message_id": reply_to_message_id,
            },
        )

    def send_animation(
        self,
        chat_id: int | str,
        animation: str,
        message_thread_id: Optional[int] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        return self.call(
            "sendAnimation",
            {
                "chat_id": chat_id,
                "animation": animation,
                "message_thread_id": message_thread_id,
                "caption": caption,
                "parse_mode": parse_mode,
                "reply_to_message_id": reply_to_message_id,
            },
        )

    def _post_multipart(
        self,
        method: str,
        fields: Dict[str, Any],
        file_field: str,
        filename: str,
        data_bytes: bytes,
        mime: str,
        timeout: int = 90,
    ) -> Any:
        url = f"{self.base_url}/{method}"
        file_bytes = self._as_bytes(data_bytes)

        def make_request() -> urllib.request.Request:
            # Rebuild the multipart body (and boundary) for every attempt. The
            # file itself was normalized to immutable bytes above, so retries
            # are safe even if the caller originally passed a file-like object.
            boundary = uuid.uuid4().hex
            body = bytearray()
            for k, v in fields.items():
                if v is not None:
                    body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode("utf-8"))
            body.extend(f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\nContent-Type: {mime}\r\n\r\n'.encode("utf-8"))
            body.extend(file_bytes)
            body.extend(b"\r\n")
            body.extend(f"--{boundary}--\r\n".encode("utf-8"))
            return urllib.request.Request(
                url,
                data=bytes(body),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
                method="POST",
            )

        return self._execute_with_retry(method, make_request, timeout)

    @staticmethod
    def _as_bytes(data: Any) -> bytes:
        """Normalize uploaded data to bytes so it can be reused across retries."""
        if isinstance(data, bytes):
            return data
        if isinstance(data, (bytearray, memoryview)):
            return bytes(data)
        if hasattr(data, "read"):
            if hasattr(data, "seek"):
                try:
                    data.seek(0)
                except (OSError, ValueError):
                    pass
            chunk = data.read()
            return chunk.encode("utf-8") if isinstance(chunk, str) else bytes(chunk)
        return bytes(data)

    def send_photo(
        self,
        chat_id: int | str,
        photo: bytes,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        photo_bytes = self._as_bytes(photo)
        mime, ext = sniff_image_type(photo_bytes)
        if filename is None:
            filename = f"photo.{ext}"
        fields = {
            "chat_id": chat_id,
            "caption": caption,
            "message_thread_id": message_thread_id,
            "parse_mode": parse_mode,
            "reply_to_message_id": reply_to_message_id,
        }
        return self._post_multipart("sendPhoto", fields, "photo", filename, photo_bytes, mime)

    def send_video(
        self,
        chat_id: int | str,
        video: bytes,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        video_bytes = self._as_bytes(video)
        mime, ext = sniff_video_type(video_bytes)
        if filename is None:
            filename = f"video.{ext}"
        fields = {
            "chat_id": chat_id,
            "caption": caption,
            "message_thread_id": message_thread_id,
            "parse_mode": parse_mode,
            "reply_to_message_id": reply_to_message_id,
        }
        return self._post_multipart("sendVideo", fields, "video", filename, video_bytes, mime, timeout=120)

    def send_voice(
        self,
        chat_id: int | str,
        voice: bytes,
        filename: str = "voice.mp3",
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        fields = {
            "chat_id": chat_id,
            "caption": caption,
            "message_thread_id": message_thread_id,
            "parse_mode": parse_mode,
            "reply_to_message_id": reply_to_message_id,
        }
        return self._post_multipart("sendVoice", fields, "voice", filename, voice, "audio/mpeg")

    def copy_message(
        self,
        chat_id: int | str,
        from_chat_id: int | str,
        message_id: int,
        message_thread_id: Optional[int] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
    ) -> Dict[str, Any]:
        return self.call(
            "copyMessage",
            {
                "chat_id": chat_id,
                "from_chat_id": from_chat_id,
                "message_id": message_id,
                "message_thread_id": message_thread_id,
                "caption": caption,
                "parse_mode": parse_mode,
            },
        )

    def forward_message(
        self,
        chat_id: int | str,
        from_chat_id: int | str,
        message_id: int,
        message_thread_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "from_chat_id": from_chat_id,
            "message_id": message_id,
        }
        if message_thread_id is not None:
            payload["message_thread_id"] = message_thread_id
        return self.call("forwardMessage", payload)

    def delete_message(self, chat_id: int | str, message_id: int) -> bool:
        return bool(
            self.call("deleteMessage", {"chat_id": chat_id, "message_id": message_id})
        )

    def edit_message_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None,
        disable_web_page_preview: bool = True,
    ) -> Any:
        return self.call(
            "editMessageText",
            {
                "chat_id": chat_id,
                "message_id": message_id,
                "text": text,
                "parse_mode": parse_mode,
                "reply_markup": reply_markup,
                "disable_web_page_preview": disable_web_page_preview,
            },
        )

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> bool:
        return bool(
            self.call(
                "answerCallbackQuery",
                {
                    "callback_query_id": callback_query_id,
                    "text": text,
                    "show_alert": show_alert,
                },
            )
        )

    def set_message_reaction(
        self,
        chat_id: int | str,
        message_id: int,
        reaction: Optional[str] = None,
        is_big: bool = False,
    ) -> bool:
        payload: Dict[str, Any] = {
            "chat_id": chat_id,
            "message_id": message_id,
        }
        if reaction:
            payload["reaction"] = [{"type": "emoji", "emoji": reaction}]
            if is_big:
                payload["is_big"] = True
        else:
            payload["reaction"] = []
        try:
            return bool(self.call("setMessageReaction", payload))
        except Exception:
            return False

    def get_file_bytes(self, file_id: str, timeout: int = 25) -> Optional[bytes]:
        try:
            file_info = self.call("getFile", {"file_id": file_id})
            file_path = file_info.get("file_path")
            if not file_path:
                return None
            download_url = f"https://api.telegram.org/file/bot{self.token}/{file_path}"
            req = urllib.request.Request(download_url, headers={"User-Agent": "TopicGuardBot/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read()
        except Exception:
            return None

    def get_chat_administrators(self, chat_id: int | str) -> List[Dict[str, Any]]:
        return self.call("getChatAdministrators", {"chat_id": chat_id})

    def get_chat_member(self, chat_id: int | str, user_id: int) -> Dict[str, Any]:
        return self.call("getChatMember", {"chat_id": chat_id, "user_id": user_id})

    def restrict_chat_member(
        self,
        chat_id: int | str,
        user_id: int,
        permissions: Optional[Dict[str, bool]] = None,
        until_date: Optional[int] = None,
    ) -> bool:
        if permissions is None:
            permissions = {
                "can_send_messages": False,
                "can_send_audios": False,
                "can_send_documents": False,
                "can_send_photos": False,
                "can_send_videos": False,
                "can_send_video_notes": False,
                "can_send_voice_notes": False,
                "can_send_polls": False,
                "can_send_other_messages": False,
                "can_add_web_page_previews": False,
                "can_change_info": False,
                "can_invite_users": False,
                "can_pin_messages": False,
                "can_manage_topics": False,
            }
        return bool(
            self.call(
                "restrictChatMember",
                {
                    "chat_id": chat_id,
                    "user_id": user_id,
                    "permissions": permissions,
                    "until_date": until_date,
                },
            )
        )

    def delete_messages(self, chat_id: int | str, message_ids: List[int]) -> bool:
        if not message_ids:
            return True
        return bool(
            self.call(
                "deleteMessages",
                {"chat_id": chat_id, "message_ids": message_ids[:100]},
            )
        )

    def set_my_commands(
        self,
        commands: List[Dict[str, str]],
        scope: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return bool(
            self.call("setMyCommands", {"commands": commands, "scope": scope})
        )


class MockTelegramAPI:
    """In-memory mock of TelegramAPI for unit testing."""

    def __init__(self, admin_user_ids: Optional[List[int]] = None) -> None:
        self.sent_messages: List[Dict[str, Any]] = []
        self.copied_messages: List[Dict[str, Any]] = []
        self.deleted_messages: List[Dict[str, Any]] = []
        self.edited_messages: List[Dict[str, Any]] = []
        self.answered_callbacks: List[Dict[str, Any]] = []
        self.restricted_members: List[Dict[str, Any]] = []
        self.reactions: List[Dict[str, Any]] = []
        self.admin_user_ids = set(admin_user_ids or [1001])
        self._next_msg_id = 5000

    def get_me(self) -> Dict[str, Any]:
        return {"id": 8982824390, "is_bot": True, "username": "DemoBot", "first_name": "Demo"}

    def send_message(
        self,
        chat_id: int | str,
        text: str,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None,
        disable_web_page_preview: bool = True,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "text": text,
            "message_thread_id": message_thread_id,
            "parse_mode": parse_mode,
            "reply_markup": reply_markup,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(rec)
        return rec

    def send_sticker(
        self,
        chat_id: int | str,
        sticker: str,
        message_thread_id: Optional[int] = None,
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "sticker": sticker,
            "message_thread_id": message_thread_id,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(rec)
        return rec

    def send_animation(
        self,
        chat_id: int | str,
        animation: str,
        message_thread_id: Optional[int] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "animation": animation,
            "message_thread_id": message_thread_id,
            "caption": caption,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(rec)
        return rec

    def send_photo(
        self,
        chat_id: int | str,
        photo: bytes,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "photo_len": len(photo),
            "caption": caption,
            "message_thread_id": message_thread_id,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(rec)
        return rec

    def send_video(
        self,
        chat_id: int | str,
        video: bytes,
        filename: Optional[str] = None,
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "video_len": len(video),
            "caption": caption,
            "message_thread_id": message_thread_id,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(rec)
        return rec

    def send_voice(
        self,
        chat_id: int | str,
        voice: bytes,
        filename: str = "voice.mp3",
        caption: Optional[str] = None,
        message_thread_id: Optional[int] = None,
        parse_mode: Optional[str] = "HTML",
        reply_to_message_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "voice_len": len(voice),
            "caption": caption,
            "message_thread_id": message_thread_id,
            "reply_to_message_id": reply_to_message_id,
        }
        self.sent_messages.append(rec)
        return rec

    def copy_message(
        self,
        chat_id: int | str,
        from_chat_id: int | str,
        message_id: int,
        message_thread_id: Optional[int] = None,
        caption: Optional[str] = None,
        parse_mode: Optional[str] = "HTML",
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "from_chat_id": int(from_chat_id),
            "source_message_id": message_id,
            "message_thread_id": message_thread_id,
            "caption": caption,
        }
        self.copied_messages.append(rec)
        return rec

    def forward_message(
        self,
        chat_id: int | str,
        from_chat_id: int | str,
        message_id: int,
        message_thread_id: Optional[int] = None,
    ) -> Dict[str, Any]:
        self._next_msg_id += 1
        rec = {
            "message_id": self._next_msg_id,
            "chat_id": int(chat_id),
            "from_chat_id": int(from_chat_id),
            "source_message_id": message_id,
            "message_thread_id": message_thread_id,
            "forwarded": True,
        }
        self.copied_messages.append(rec)
        return rec

    def delete_message(self, chat_id: int | str, message_id: int) -> bool:
        self.deleted_messages.append({"chat_id": int(chat_id), "message_id": int(message_id)})
        return True

    def delete_messages(self, chat_id: int | str, message_ids: List[int]) -> bool:
        for mid in message_ids:
            self.deleted_messages.append({"chat_id": int(chat_id), "message_id": int(mid)})
        return True

    def set_message_reaction(
        self,
        chat_id: int | str,
        message_id: int,
        reaction: Optional[str] = None,
        is_big: bool = False,
    ) -> bool:
        self.reactions.append({"chat_id": int(chat_id), "message_id": int(message_id), "reaction": reaction})
        return True

    def get_file_bytes(self, file_id: str, timeout: int = 25) -> Optional[bytes]:
        return b"\x89PNG\r\n\x1a\nfakeimagebytes"

    def restrict_chat_member(
        self,
        chat_id: int | str,
        user_id: int,
        permissions: Optional[Dict[str, bool]] = None,
        until_date: Optional[int] = None,
    ) -> bool:
        self.restricted_members.append(
            {"chat_id": int(chat_id), "user_id": int(user_id), "until_date": until_date}
        )
        return True

    def edit_message_text(
        self,
        chat_id: int | str,
        message_id: int,
        text: str,
        parse_mode: Optional[str] = "HTML",
        reply_markup: Optional[Dict[str, Any]] = None,
        disable_web_page_preview: bool = True,
    ) -> Dict[str, Any]:
        rec = {
            "chat_id": int(chat_id),
            "message_id": int(message_id),
            "text": text,
            "reply_markup": reply_markup,
        }
        self.edited_messages.append(rec)
        return rec

    def answer_callback_query(
        self,
        callback_query_id: str,
        text: Optional[str] = None,
        show_alert: bool = False,
    ) -> bool:
        self.answered_callbacks.append(
            {"callback_query_id": callback_query_id, "text": text, "show_alert": show_alert}
        )
        return True

    def get_chat_administrators(self, chat_id: int | str) -> List[Dict[str, Any]]:
        return [
            {"user": {"id": uid, "is_bot": False, "first_name": f"Admin{uid}"}, "status": "administrator"}
            for uid in self.admin_user_ids
        ]

    def get_chat_member(self, chat_id: int | str, user_id: int) -> Dict[str, Any]:
        status = "administrator" if user_id in self.admin_user_ids else "member"
        return {"user": {"id": user_id, "is_bot": False}, "status": status}

    def set_my_commands(
        self,
        commands: List[Dict[str, str]],
        scope: Optional[Dict[str, Any]] = None,
    ) -> bool:
        return True
