"""Name normalisation, name location, fuzzy matching and chunking — pure logic, no models."""

import pytest

from app.services.bbox_service import ConsecutiveWordNameLocator
from app.services.fuzzy_service import SIMILARITY_THRESHOLD, TokenSortNameMatcher
from app.services.ner_service import PersonNameNormalizer
from app.services.rag_service import TextChunker
from tests.fakes import word


class TestPersonNameNormalizer:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Dr. Aisha Patel", "Aisha Patel"),
            ("Prof. Michael O'Sullivan", "Michael O'Sullivan"),
            ("Kevin O'Brien,", "Kevin O'Brien"),
            ("Fatima Al-Rashidi", "Fatima Al-Rashidi"),
            ("Robert Chen's", "Robert Chen"),
            ("Aisha\nPatel", "Aisha Patel"),
            ("  Jennifer   Liu  ", "Jennifer Liu"),
            ("Dr.", ""),
        ],
    )
    def test_normalize(self, raw, expected):
        assert PersonNameNormalizer().normalize(raw) == expected

    def test_normalize_all_dedupes_and_keeps_order(self):
        names = ["Dr. Aisha Patel", "Robert Chen", "Aisha Patel", "Dr.", "Robert Chen"]
        assert PersonNameNormalizer().normalize_all(names) == ["Aisha Patel", "Robert Chen"]


class TestConsecutiveWordNameLocator:
    locator = ConsecutiveWordNameLocator()

    def test_ignores_case_and_edge_punctuation(self):
        boxes = [word("RICHARD", 0, 10, 20), word("Hernandez,", 0, 45, 20, w=60)]
        [box] = self.locator.locate(["Richard Hernandez"], boxes)
        assert (box["x"], box["y"], box["width"], box["page"]) == (10, 20, 95, 0)

    def test_requires_consecutive_words_on_one_page(self):
        boxes = [
            word("John", 0, 10, 20),
            word("met", 0, 45, 20),
            word("Smith", 0, 80, 20),
            word("John", 0, 10, 700),
            word("Smith", 1, 10, 20),
        ]
        assert self.locator.locate(["John Smith"], boxes) == []

    def test_returns_every_occurrence_in_document_order(self):
        boxes = [
            word("Jennifer", 1, 10, 50),
            word("Liu", 1, 45, 50),
            word("Jennifer", 0, 10, 300),
            word("Liu", 0, 45, 300),
        ]
        found = self.locator.locate(["Jennifer Liu", "Jennifer Liu"], boxes)
        assert [(b["page"], b["y"]) for b in found] == [(0, 300), (1, 50)]

    def test_merged_box_spans_all_words(self):
        boxes = [word("Mary", 0, 10, 20, h=10), word("Ann", 0, 50, 18, h=14), word("Lee", 0, 90, 20, w=20, h=10)]
        [box] = self.locator.locate(["Mary Ann Lee"], boxes)
        assert (box["x"], box["y"], box["width"], box["height"]) == (10, 18, 100, 14)


class TestTokenSortNameMatcher:
    matcher = TokenSortNameMatcher(SIMILARITY_THRESHOLD)

    @pytest.mark.parametrize(
        "extracted, first, last, should_match",
        [
            ("Robert Chen", "Robert", "Chen", True),  # exact
            ("Chen Robert", "Robert", "Chen", True),  # reversed order
            ("ROBERT CHEN", "robert", "chen", True),  # case
            ("Margret Thopmson", "Margaret", "Thompson", True),  # OCR-style typos (score 91)
            ("John Smith", "Jo", "Sm", False),  # partial names
            ("Robert Chen", "James", "Chen", False),  # same surname, different person
        ],
    )
    def test_threshold_decisions(self, extracted, first, last, should_match):
        matches = self.matcher.match([extracted], [{"first_name": first, "last_name": last}])
        assert bool(matches) is should_match
        if matches:
            assert matches[0]["score"] >= 0.9

    def test_picks_the_best_candidate_per_query(self):
        matches = self.matcher.match(["Robert Chen", "Robet Chen"], [{"first_name": "Robert", "last_name": "Chen"}])
        assert matches == [{"extracted_name": "Robert Chen", "matched_name": "Robert Chen", "score": 1.0}]

    def test_threshold_is_configurable(self):
        lenient = TokenSortNameMatcher(threshold=60)
        assert lenient.match(["John Smith"], [{"first_name": "Jo", "last_name": "Sm"}])


class TestTextChunker:
    chunker = TextChunker()

    @pytest.mark.parametrize("size", [10, 25, 80, 500])
    def test_preserves_text_and_never_splits_words(self, size):
        text = "Board minutes:\nRichard Hernandez called the meeting to order at 10:05 AM. " * 5
        chunks = self.chunker.chunk(text, size)
        assert "".join(chunks) == text
        words = set(text.split())
        assert all(w in words for chunk in chunks for w in chunk.split())

    def test_respects_size_except_for_single_long_words(self):
        text = "short words " * 20 + "x" * 60 + " tail"
        chunks = self.chunker.chunk(text, 30)
        assert all(len(c) <= 30 or len(c.split()) == 1 for c in chunks)
        assert "x" * 60 + " " in chunks

    def test_empty_text(self):
        assert self.chunker.chunk("", 100) == []
