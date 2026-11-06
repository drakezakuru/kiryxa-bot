"""Дополнительные тесты конфига и персистентного offset Telegram."""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import _to_int, load_config
from main import is_stale, load_offset, save_offset

NON_HEX_KEY = "sk_" + "A1b2_c3-D4e5_f6-G7h8"


def _write_config(directory: Path, payload: dict) -> Path:
    path = directory / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


class TestConfigParsing(unittest.TestCase):
    def _load(self, directory: Path, payload: dict):
        cfg_path = _write_config(directory, payload)
        with mock.patch.dict(os.environ, {}, clear=True):
            return load_config(cfg_path)

    def test_to_int_handles_bad_values_and_bools(self) -> None:
        self.assertEqual(_to_int("abc", 25), 25)
        self.assertEqual(_to_int(None, 25), 25)
        self.assertEqual(_to_int("", 25), 25)
        self.assertEqual(_to_int(True, 25), 1)
        self.assertEqual(_to_int(False, 25), 0)
        self.assertEqual(_to_int("7", 25), 7)

    def test_invalid_poll_timeout_falls_back_to_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cfg = self._load(Path(tmp), {"poll_timeout": "abc", "data_dir": str(data_dir)})
        self.assertEqual(cfg.poll_timeout, 25)

    def test_valid_poll_timeout_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cfg = self._load(Path(tmp), {"poll_timeout": "40", "data_dir": str(data_dir)})
        self.assertEqual(cfg.poll_timeout, 40)

    def test_default_notify_sec_string_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cfg = self._load(Path(tmp), {"default_notify_sec": "7", "data_dir": str(data_dir)})
        self.assertEqual(cfg.default_notify_sec, 7)

    def test_invalid_default_notify_sec_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cfg = self._load(
                Path(tmp), {"default_notify_sec": "abc", "data_dir": str(data_dir)}
            )
        self.assertEqual(cfg.default_notify_sec, 0)

    def test_elevenlabs_enabled_false_string(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "data"
            cfg = self._load(
                Path(tmp), {"elevenlabs_enabled": "false", "data_dir": str(data_dir)}
            )
        self.assertFalse(cfg.elevenlabs_enabled)

    def test_relative_data_dir_becomes_absolute_and_exists(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = _write_config(Path(tmp), {"data_dir": "nested/data"})
            with mock.patch.dict(os.environ, {}, clear=True):
                cfg = load_config(cfg_path)
            resolved = Path(cfg.data_dir)
            expected = (Path(tmp) / "nested" / "data").resolve()
            self.assertTrue(resolved.is_absolute())
            self.assertEqual(resolved, expected)
            self.assertTrue(resolved.is_dir())

    def test_non_hex_elevenlabs_key_from_file_is_parsed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            keys_file = Path(tmp) / "keys.txt"
            keys_file.write_text(f"{NON_HEX_KEY}\n", encoding="utf-8")
            data_dir = Path(tmp) / "data"
            cfg = self._load(
                Path(tmp),
                {
                    "data_dir": str(data_dir),
                    "elevenlabs_keys_file": str(keys_file),
                },
            )
        self.assertIn(NON_HEX_KEY, cfg.elevenlabs_keys)


class TestTelegramOffset(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = Path(self._tmp.name) / "telegram_offset.json"

    def test_load_missing_file_returns_zero(self) -> None:
        self.assertEqual(load_offset(self.path), 0)

    def test_load_corrupted_file_returns_zero(self) -> None:
        self.path.write_text("{broken", encoding="utf-8")
        self.assertEqual(load_offset(self.path), 0)

    def test_save_then_load(self) -> None:
        save_offset(self.path, 12345)
        self.assertEqual(load_offset(self.path), 12345)

    def test_save_overwrites_previous_value(self) -> None:
        save_offset(self.path, 10)
        save_offset(self.path, 11)
        self.assertEqual(load_offset(self.path), 11)

    def test_save_is_atomic_and_leaves_no_tmp_file(self) -> None:
        save_offset(self.path, 5)
        leftovers = list(self.path.parent.glob(f".{self.path.name}.*.tmp"))
        self.assertEqual(leftovers, [])
        self.assertTrue(self.path.is_file())

    def test_is_stale_for_processed_and_new_updates(self) -> None:
        self.assertTrue(is_stale(100, 100))
        self.assertTrue(is_stale(99, 100))
        self.assertFalse(is_stale(101, 100))
        self.assertFalse(is_stale(1, 0))


if __name__ == "__main__":
    unittest.main()