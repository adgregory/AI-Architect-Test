"""Fuzzy matching between extracted names and requested name pairs."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from thefuzz import fuzz

SIMILARITY_THRESHOLD = 90  # percent — required by the specification


@runtime_checkable
class NameMatcher(Protocol):
    def match(self, extracted_names: list[str], query_names: list[dict]) -> list[dict]: ...


class TokenSortNameMatcher:
    """Best extracted name per query name, kept if its score reaches the threshold.

    Scores whole names with token_sort_ratio (typo-tolerant, insensitive to word
    order and case). partial_ratio is deliberately not used: it scores a short
    query as a substring match, so "Jo Sm" would match "John Smith".
    """

    def __init__(self, threshold: int = SIMILARITY_THRESHOLD):
        self._threshold = threshold

    @staticmethod
    def normalize(name: str) -> str:
        return " ".join(name.casefold().split())

    def match(self, extracted_names: list[str], query_names: list[dict]) -> list[dict]:
        matches = []
        for query in query_names:
            query_full = f"{query['first_name']} {query['last_name']}"
            best_match, best_score = None, 0
            for extracted in extracted_names:
                score = fuzz.token_sort_ratio(self.normalize(query_full), self.normalize(extracted))
                if score > best_score:
                    best_match, best_score = extracted, score
            if best_score >= self._threshold:
                matches.append({
                    "extracted_name": best_match,
                    "matched_name": query_full,
                    "score": best_score / 100.0,
                })
        return matches


# Functional API (backwards compatible).
fuzzy_match_names = TokenSortNameMatcher(SIMILARITY_THRESHOLD).match
