"""Tests for web_search module."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.web_search import (
    extract_search_query,
    get_web_facts,
    needs_web_search,
)


class TestWebSearch(unittest.TestCase):
    def test_needs_web_search(self) -> None:
        self.assertTrue(needs_web_search("В инете чекни дату Сталинградской войны"))
        self.assertTrue(needs_web_search("кирюха погугли когда выйдет gta 6"))
        self.assertTrue(needs_web_search("загугли курс биткоина"))
        self.assertTrue(needs_web_search("найди в интернете инфу про claude 4.5"))
        self.assertTrue(needs_web_search("в каком году началась война?"))
        self.assertFalse(needs_web_search("киря как дела"))
        self.assertFalse(needs_web_search("спс забрал ключ"))

    def test_extract_search_query(self) -> None:
        q = extract_search_query("В инете чекни дату Сталинградской войны")
        self.assertIn("Сталинградской", q)
        self.assertNotIn("инете", q.lower())
        self.assertNotIn("чекни", q.lower())

    def test_release_date_queries(self) -> None:
        self.assertTrue(needs_web_search("Поищи когда выйдет гта 6"))
        self.assertTrue(needs_web_search("когда выйдет гта 6"))
        self.assertTrue(needs_web_search("Найди точную дату"))

        # Короткий вопрос про дату без темы — тема берётся из цитаты
        text = (
            '[В ответ на Kiryxa: "гта 6 выходит осенью 2026 — точная дата ещё не названа"]\n'
            "Найди точную дату"
        )
        self.assertTrue(needs_web_search(text))
        q = extract_search_query(text)
        self.assertIn("гта 6", q.lower())
        self.assertIn("дата выхода", q.lower())

    def test_freebies_queries(self) -> None:
        self.assertTrue(needs_web_search("Найди халяву мб фри трил мб гдето хоть фри бакс дают"))
        self.assertTrue(needs_web_search("где дают бесплатные кредиты на нейронки"))
        q = extract_search_query("Найди халяву мб фри трил мб гдето хоть фри бакс дают")
        self.assertNotIn("мб", q.lower())

    def test_get_web_facts_with_mock_exa(self) -> None:
        import core.web_search
        orig_search_exa = core.web_search.search_exa
        try:
            core.web_search.search_exa = lambda q, api_key, **kwargs: [f"Результат по {q}"]
            facts = get_web_facts("В инете чекни дату Сталинградской войны", exa_api_key="fake_key")
            self.assertIn("РЕЗУЛЬТАТЫ ПОИСКА В ИНТЕРНЕТЕ", facts)
            self.assertIn("Результат по", facts)
        finally:
            core.web_search.search_exa = orig_search_exa

    def test_get_web_facts_marks_results_untrusted(self) -> None:
        with mock.patch(
            "core.web_search.search_exa",
            return_value=["GTA 6 выйдет 19 ноября 2026"],
        ):
            facts = get_web_facts("когда выйдет gta 6", exa_api_key="fake_key")

        self.assertIn("[НЕДОВЕРЕННЫЕ РЕЗУЛЬТАТЫ ПОИСКА", facts)
        self.assertIn("[/НЕДОВЕРЕННЫЕ РЕЗУЛЬТАТЫ ПОИСКА]", facts)
        self.assertIn("GTA 6 выйдет 19 ноября 2026", facts)
        self.assertLess(
            facts.index("[НЕДОВЕРЕННЫЕ РЕЗУЛЬТАТЫ ПОИСКА"),
            facts.index("GTA 6 выйдет"),
        )
        self.assertLess(
            facts.index("GTA 6 выйдет"),
            facts.index("[/НЕДОВЕРЕННЫЕ РЕЗУЛЬТАТЫ ПОИСКА]"),
        )

    def test_get_web_facts_empty_without_search_intent(self) -> None:
        self.assertEqual(get_web_facts("киря как дела"), "")


if __name__ == "__main__":
    unittest.main()
