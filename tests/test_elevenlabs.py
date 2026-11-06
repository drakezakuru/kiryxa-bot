"""Tests for ElevenLabs TTS client: key file parsing and rotation."""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.elevenlabs import ElevenLabsClient, load_keys_from_file

K1 = "sk_" + "a" * 48
K2 = "sk_" + "b" * 48
K3 = "sk_" + "c" * 48
# Ключи ElevenLabs бывают не только hex: в суффиксе встречаются "_" и "-".
NON_HEX_KEY = "sk_" + "A1b2_c3-D4e5_f6-G7h8"
assert len(NON_HEX_KEY) == 23


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *args: object) -> bool:
        return False

    def read(self) -> bytes:
        return self.payload


class TestElevenLabsKeys(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.cooldown_path = Path(self._tmp.name) / "elevenlabs_cooldown.json"

    def _client(self, keys: list[str]) -> ElevenLabsClient:
        return ElevenLabsClient(keys, voice_id="v", cooldown_path=self.cooldown_path)

    def test_load_keys_from_markdown_table(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keys.md"
            path.write_text(
                "# elevenlabs\n\n| Ключ | Статус |\n|---|---|\n"
                f"| `{K1}` | ok |\n"
                f"| `{K2}` | ok |\n"
                f"| `{K1}` | дубль |\n"
                f"| {K3} | без бэктиков |\n",
                encoding="utf-8",
            )
            keys = load_keys_from_file(path)
        self.assertEqual(keys, [K1, K2, K3])

    def test_load_non_hex_key_from_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "keys.txt"
            path.write_text(f"# отчёт\n{NON_HEX_KEY}\n", encoding="utf-8")
            keys = load_keys_from_file(path)
        self.assertEqual(keys, [NON_HEX_KEY])

    def test_missing_file_is_empty(self) -> None:
        self.assertEqual(load_keys_from_file("/nonexistent/keys.md"), [])

    def test_rotation_skips_cooling_keys(self) -> None:
        client = self._client([K1, K2])
        self.assertTrue(client.enabled)
        self.assertEqual(client.available_keys(), 2)

        seen = {client._next_key() for _ in range(4)}
        self.assertEqual(seen, {K1, K2})

        client._cool(K1, 300)
        self.assertEqual(client._next_key(), K2)
        self.assertEqual(client._next_key(), K2)
        self.assertEqual(client.available_keys(), 1)

    def test_all_keys_cooling_returns_none(self) -> None:
        client = self._client([K1])
        client._cool(K1, 300)
        self.assertIsNone(client._next_key())
        self.assertIsNone(client.synthesize("привет"))

    def test_no_keys_disabled(self) -> None:
        client = self._client([])
        self.assertFalse(client.enabled)
        self.assertIsNone(client.synthesize("привет"))

    def test_short_audio_is_accepted(self) -> None:
        client = self._client([K1])
        audio = b"x" * 300
        with mock.patch(
            "core.elevenlabs.urllib.request.urlopen",
            return_value=_FakeResponse(audio),
        ):
            result = client.synthesize("да")
        self.assertEqual(result, audio)

    def test_tiny_audio_is_rejected(self) -> None:
        client = self._client([K1])
        with mock.patch(
            "core.elevenlabs.urllib.request.urlopen",
            return_value=_FakeResponse(b"x" * 199),
        ):
            result = client.synthesize("да")
        self.assertIsNone(result)

    def test_cooldown_persists_and_loads(self) -> None:
        first = self._client([K1])
        first._cool(K1, 300)
        self.assertTrue(self.cooldown_path.is_file())

        stored = json.loads(self.cooldown_path.read_text(encoding="utf-8"))
        self.assertIn(K1, stored)
        self.assertGreater(stored[K1], time.time())

        second = self._client([K1])
        self.assertEqual(second.available_keys(), 0)
        self.assertIsNone(second._next_key())

    def test_expired_cooldown_is_dropped_on_load(self) -> None:
        self.cooldown_path.write_text(
            json.dumps({K1: time.time() - 60}),
            encoding="utf-8",
        )
        client = self._client([K1])
        self.assertEqual(client.available_keys(), 1)
        self.assertEqual(client._next_key(), K1)

    def test_corrupted_cooldown_file_is_ignored(self) -> None:
        self.cooldown_path.write_text("{not json", encoding="utf-8")
        client = self._client([K1])
        self.assertEqual(client.available_keys(), 1)


if __name__ == "__main__":
    unittest.main()