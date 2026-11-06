"""Unit tests for Telegram API retry handling (HTTP 429 / 5xx), no network."""

from __future__ import annotations

import io
import json
import sys
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bot.tg_api import TelegramAPI, TelegramError


class _FakeResponse:
    """Minimal context-manager response supporting .read()."""

    def __init__(self, payload: dict) -> None:
        self._body = json.dumps(payload).encode("utf-8")

    def read(self) -> bytes:
        return self._body

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False


def _http_error(status: int, payload: dict) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        url="https://api.telegram.org/bottest/sendMessage",
        code=status,
        msg="error",
        hdrs={},
        fp=io.BytesIO(json.dumps(payload).encode("utf-8")),
    )


def _rate_limit_error(retry_after: float = 1) -> urllib.error.HTTPError:
    return _http_error(
        429,
        {
            "ok": False,
            "error_code": 429,
            "description": "Too Many Requests: retry later",
            "parameters": {"retry_after": retry_after},
        },
    )


class TestTelegramAPIRetries(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TelegramAPI("123:TEST")
        self.payload = {"chat_id": 1, "text": "hi"}

    def test_429_retries_then_returns_result(self) -> None:
        ok = _FakeResponse({"ok": True, "result": {"message_id": 42}})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=[_rate_limit_error(), ok]
        ) as urlopen, mock.patch("bot.tg_api.time.sleep") as sleep:
            result = self.api.call("sendMessage", self.payload)

        self.assertEqual(result, {"message_id": 42})
        self.assertEqual(urlopen.call_count, 2)

        # The same JSON body must be resent on the retry.
        first_req = urlopen.call_args_list[0].args[0]
        second_req = urlopen.call_args_list[1].args[0]
        self.assertEqual(first_req.data, second_req.data)
        self.assertEqual(json.loads(second_req.data), self.payload)

        sleep.assert_called_once()
        delay = sleep.call_args.args[0]
        self.assertGreaterEqual(delay, 1.0)
        self.assertLessEqual(delay, 1.5)  # retry_after + jitter(0..0.5)

    def test_three_consecutive_429_raises_telegram_error(self) -> None:
        errors = [_rate_limit_error() for _ in range(3)]
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=errors
        ) as urlopen, mock.patch("bot.tg_api.time.sleep") as sleep:
            with self.assertRaises(TelegramError) as ctx:
                self.api.call("sendMessage", self.payload)

        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertIn("retry_after=1", str(ctx.exception))
        self.assertEqual(ctx.exception.error_code, 429)

    def test_400_is_not_retried(self) -> None:
        error = _http_error(
            400,
            {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: chat not found",
            },
        )
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=[error]
        ) as urlopen, mock.patch("bot.tg_api.time.sleep") as sleep:
            with self.assertRaises(TelegramError) as ctx:
                self.api.call("sendMessage", self.payload)

        self.assertEqual(urlopen.call_count, 1)
        sleep.assert_not_called()
        self.assertEqual(ctx.exception.error_code, 400)
        self.assertIn("chat not found", ctx.exception.description)

    def test_500_is_retried_once(self) -> None:
        error = _http_error(
            500,
            {
                "ok": False,
                "error_code": 500,
                "description": "Internal Server Error",
            },
        )
        ok = _FakeResponse({"ok": True, "result": True})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=[error, ok]
        ) as urlopen, mock.patch("bot.tg_api.time.sleep") as sleep:
            result = self.api.call("sendMessage", self.payload)

        self.assertTrue(result)
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()
        delay = sleep.call_args.args[0]
        self.assertGreaterEqual(delay, 1.0)
        self.assertLessEqual(delay, 2.0)

    def test_retry_after_uses_default_and_is_capped(self) -> None:
        missing_parameters = _http_error(
            429,
            {"ok": False, "error_code": 429, "description": "Too Many Requests"},
        )
        ok = _FakeResponse({"ok": True, "result": True})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen",
            side_effect=[missing_parameters, ok],
        ), mock.patch("bot.tg_api.time.sleep") as sleep:
            self.api.call("sendMessage", self.payload)

        delay = sleep.call_args.args[0]
        self.assertGreaterEqual(delay, 3.0)  # default retry_after
        self.assertLessEqual(delay, 3.5)

        # Huge retry_after values are capped by MAX_SINGLE_WAIT and the total
        # wait budget, and still end with TelegramError after 3 attempts.
        errors = [_rate_limit_error(120) for _ in range(3)]
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=errors
        ) as urlopen, mock.patch("bot.tg_api.time.sleep") as sleep:
            with self.assertRaises(TelegramError):
                self.api.call("sendMessage", self.payload)

        self.assertEqual(urlopen.call_count, 3)
        delays = [call.args[0] for call in sleep.call_args_list]
        self.assertLessEqual(max(delays), 60.0)
        self.assertLessEqual(sum(delays), 90.0)

    def test_multipart_retry_resends_file_bytes(self) -> None:
        # A file-like upload is consumed once and normalized to bytes, so the
        # rebuilt multipart body on retry still contains the file.
        ok = _FakeResponse({"ok": True, "result": {"message_id": 7}})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=[_rate_limit_error(), ok]
        ) as urlopen, mock.patch("bot.tg_api.time.sleep") as sleep:
            result = self.api.send_voice(chat_id=1, voice=io.BytesIO(b"voice-bytes"))

        self.assertEqual(result, {"message_id": 7})
        self.assertEqual(urlopen.call_count, 2)
        sleep.assert_called_once()
        for call in urlopen.call_args_list:
            req = call.args[0]
            self.assertTrue(req.data.endswith(b"\r\n"))
            self.assertIn(b"voice-bytes", req.data)
            self.assertIn(b"multipart/form-data", req.headers["Content-type"].encode())


class TestSendMessageReplyPayload(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TelegramAPI("123:TEST")

    def test_payload_uses_reply_parameters_only(self) -> None:
        ok = _FakeResponse({"ok": True, "result": {"message_id": 10}})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", return_value=ok
        ) as urlopen:
            self.api.send_message(chat_id=1, text="hi", reply_to_message_id=77)

        request = urlopen.call_args.args[0]
        payload = json.loads(request.data)
        self.assertEqual(
            payload["reply_parameters"],
            {"message_id": 77, "allow_sending_without_reply": True},
        )
        self.assertNotIn("reply_to_message_id", payload)
        self.assertNotIn("allow_sending_without_reply", payload)

    def test_reply_fallback_drops_reply_parameters_only(self) -> None:
        reply_error = _http_error(
            400,
            {
                "ok": False,
                "error_code": 400,
                "description": "Bad Request: message to be replied not found",
            },
        )
        ok = _FakeResponse({"ok": True, "result": {"message_id": 11}})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", side_effect=[reply_error, ok]
        ) as urlopen:
            result = self.api.send_message(
                chat_id=1, text="hi", reply_to_message_id=77
            )

        self.assertEqual(result, {"message_id": 11})
        self.assertEqual(urlopen.call_count, 2)
        first = json.loads(urlopen.call_args_list[0].args[0].data)
        second = json.loads(urlopen.call_args_list[1].args[0].data)
        self.assertIn("reply_parameters", first)
        self.assertNotIn("reply_parameters", second)
        self.assertNotIn("reply_to_message_id", second)


class TestMediaMimeSniffing(unittest.TestCase):
    def setUp(self) -> None:
        self.api = TelegramAPI("123:TEST")

    def _upload_body(self, data: bytes, kind: str = "photo") -> bytes:
        ok = _FakeResponse({"ok": True, "result": {"message_id": 1}})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", return_value=ok
        ) as urlopen:
            if kind == "photo":
                self.api.send_photo(chat_id=1, photo=data)
            else:
                self.api.send_video(chat_id=1, video=data)
        return urlopen.call_args.args[0].data

    def _assert_upload(self, data: bytes, mime: str, filename: str) -> None:
        body = self._upload_body(data)
        self.assertIn(f'filename="{filename}"'.encode("utf-8"), body)
        self.assertIn(f"Content-Type: {mime}".encode("utf-8"), body)

    def test_png_bytes_are_sent_as_png(self) -> None:
        self._assert_upload(b"\x89PNG\r\n\x1a\nfake-png", "image/png", "photo.png")

    def test_jpeg_bytes_are_sent_as_jpeg(self) -> None:
        self._assert_upload(
            b"\xff\xd8\xff\xe0\x00\x10JFIFfake-jpeg", "image/jpeg", "photo.jpg"
        )

    def test_gif_bytes_are_sent_as_gif(self) -> None:
        self._assert_upload(b"GIF89a-fake-gif", "image/gif", "photo.gif")

    def test_webp_bytes_are_sent_as_webp(self) -> None:
        self._assert_upload(
            b"RIFF\x24\x00\x00\x00WEBPVP8 fake-webp", "image/webp", "photo.webp"
        )

    def test_unknown_image_bytes_fall_back_to_jpeg(self) -> None:
        self._assert_upload(b"\x00\x01\x02fake", "image/jpeg", "photo.jpg")

    def test_mp4_bytes_are_sent_as_mp4(self) -> None:
        body = self._upload_body(
            b"\x00\x00\x00\x18ftypmp42fake-mp4", kind="video"
        )
        self.assertIn(b'filename="video.mp4"', body)
        self.assertIn(b"Content-Type: video/mp4", body)

    def test_webm_bytes_are_sent_as_webm(self) -> None:
        body = self._upload_body(
            b"RIFF\x00\x00\x00\x00WEBMfake-webm", kind="video"
        )
        self.assertIn(b'filename="video.webm"', body)
        self.assertIn(b"Content-Type: video/webm", body)


class TestRestrictPermissions(unittest.TestCase):
    def test_default_permissions_use_granular_fields(self) -> None:
        api = TelegramAPI("123:TEST")
        ok = _FakeResponse({"ok": True, "result": True})
        with mock.patch(
            "bot.tg_api.urllib.request.urlopen", return_value=ok
        ) as urlopen:
            result = api.restrict_chat_member(chat_id=-100, user_id=42)

        self.assertTrue(result)
        request = urlopen.call_args.args[0]
        self.assertTrue(request.full_url.endswith("/restrictChatMember"))
        payload = json.loads(request.data)
        permissions = payload["permissions"]

        for field in (
            "can_send_audios",
            "can_send_documents",
            "can_send_photos",
            "can_send_videos",
            "can_send_video_notes",
            "can_send_voice_notes",
            "can_send_polls",
            "can_send_other_messages",
        ):
            self.assertIn(field, permissions)
            self.assertFalse(permissions[field])

        self.assertNotIn("can_send_media_messages", permissions)


if __name__ == "__main__":
    unittest.main()