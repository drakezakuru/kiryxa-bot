"""Защита от возврата захардкоженных секретов в исходники."""

from __future__ import annotations

import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCAN_PATHS = (ROOT / "core", ROOT / "bot", ROOT / "main.py")

SECRET_MARKERS = ("sk_live_", "sk-proj-", "AIzaSy")
BOT_TOKEN_RE = re.compile(r"\b\d{6,}:[A-Za-z0-9_-]{30,}\b")


def _iter_source_files():
    """Все текстовые файлы core/, bot/ и main.py, кроме кэша байткода."""
    for path in SCAN_PATHS:
        if path.is_file():
            yield path
        elif path.is_dir():
            for child in sorted(path.rglob("*")):
                if child.is_file() and "__pycache__" not in child.parts:
                    yield child


class TestNoHardcodedSecrets(unittest.TestCase):
    def test_no_secret_markers_in_source(self) -> None:
        for path in _iter_source_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = path.relative_to(ROOT)
            for marker in SECRET_MARKERS:
                with self.subTest(file=str(rel), marker=marker):
                    self.assertNotIn(
                        marker,
                        text,
                        f"Захардкоженный секрет ({marker!r}) найден в {rel}",
                    )

    def test_no_telegram_bot_token_literal(self) -> None:
        for path in _iter_source_files():
            text = path.read_text(encoding="utf-8", errors="ignore")
            rel = path.relative_to(ROOT)
            self.assertIsNone(
                BOT_TOKEN_RE.search(text),
                f"Похоже на захардкоженный Telegram bot_token в {rel}",
            )

    def test_gitignore_covers_secrets(self) -> None:
        gitignore = ROOT / ".gitignore"
        self.assertTrue(gitignore.is_file(), "В корне проекта нет .gitignore")
        entries = {
            line.strip()
            for line in gitignore.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        self.assertIn("config.json", entries, ".gitignore должен игнорировать config.json")
        self.assertIn("data/", entries, ".gitignore должен игнорировать data/")


if __name__ == "__main__":
    unittest.main()