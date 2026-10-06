"""Semantic answer cache: reuse an answer when a new question means the same as a cached one.

Embedding similarity alone is not safe for this (spike 04): questions that differ only in a
negation, a number, a name or the order of two names score as high as true paraphrases
(role swaps reach cosine 0.99). A cache hit therefore needs both a high cosine *and* a
QuestionGuard that finds no meaning-changing difference. A false miss only costs an LLM call;
a false hit returns a wrong answer — the guard is tuned for zero false hits.
"""

from __future__ import annotations

import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    Range,
    VectorParams,
)


@dataclass(frozen=True)
class CachedAnswer:
    question: str
    answer: dict
    score: float


@runtime_checkable
class SemanticCache(Protocol):
    def lookup(self, question: str, vector: list[float]) -> CachedAnswer | None: ...

    def store(self, question: str, vector: list[float], answer: dict) -> None: ...


class QuestionGuard:
    """Rejects a candidate cache hit when the two questions differ in a way that changes meaning."""

    NEGATIONS = frozenset(
        {
            "not",
            "no",
            "never",
            "none",
            "nobody",
            "nothing",
            "neither",
            "nor",
            "without",
            "cannot",
            "isn't",
            "wasn't",
            "aren't",
            "weren't",
            "don't",
            "doesn't",
            "didn't",
            "won't",
            "wouldn't",
            "can't",
            "couldn't",
            "shouldn't",
            "hasn't",
            "haven't",
            "hadn't",
            "refused",
            "failed",
        }
    )
    # Opposites that leave the rest of a sentence intact (expand/shrink, promoted/demoted, ...).
    ANTONYMS = (
        ("increase", "decrease"),
        ("increased", "decreased"),
        ("rise", "fall"),
        ("rising", "falling"),
        ("rose", "fell"),
        ("grow", "shrink"),
        ("growing", "shrinking"),
        ("expand", "shrink"),
        ("expand", "reduce"),
        ("up", "down"),
        ("promoted", "demoted"),
        ("promote", "demote"),
        ("hire", "fire"),
        ("hired", "fired"),
        ("approve", "reject"),
        ("approved", "rejected"),
        ("accept", "decline"),
        ("accepted", "declined"),
        ("support", "oppose"),
        ("supported", "opposed"),
        ("open", "close"),
        ("opened", "closed"),
        ("start", "end"),
        ("started", "ended"),
        ("before", "after"),
        ("more", "less"),
        ("most", "least"),
        ("higher", "lower"),
        ("gain", "loss"),
        ("present", "absent"),
        ("win", "lose"),
        ("won", "lost"),
        ("join", "leave"),
        ("joined", "left"),
        ("buy", "sell"),
        ("bought", "sold"),
        ("add", "remove"),
        ("added", "removed"),
        ("include", "exclude"),
        ("included", "excluded"),
        ("first", "last"),
        ("maximum", "minimum"),
    )
    # Number words normalised to digits so "twelve percent" == "12%" and "third quarter" == "Q3".
    NUMBER_WORDS: ClassVar[dict[str, str]] = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
        "ten": "10",
        "eleven": "11",
        "twelve": "12",
        "twenty": "20",
        "thirty": "30",
        "forty": "40",
        "fifty": "50",
        "hundred": "100",
        "thousand": "1000",
        "million": "1000000",
        "first": "1",
        "second": "2",
        "third": "3",
        "fourth": "4",
    }
    TITLES = frozenset({"dr", "prof", "mr", "mrs", "ms", "sir"})
    _TOKEN = re.compile(r"\d[\d.,:]*%?|[A-Za-z][A-Za-z'\-]*")

    def __init__(self) -> None:
        self._opposite: dict[str, set[str]] = {}
        for a, b in self.ANTONYMS:
            self._opposite.setdefault(a, set()).add(b)
            self._opposite.setdefault(b, set()).add(a)

    def _tokens(self, text: str) -> list[str]:
        tokens = []
        for t in self._TOKEN.findall(text):
            t = re.sub(r"['’]s$", "", t)  # possessive: "December's" -> "December"
            if t.casefold().rstrip(".") in self.TITLES:  # "Dr." carries no identity on its own
                continue
            tokens.append(t)
        return tokens

    def _is_proper(self, tokens: list[str], i: int) -> bool:
        t = tokens[i]
        if not t[0].isupper() or t.casefold() in _SENTENCE_STARTERS:
            return False
        if i == 0:  # sentence-initial capital: a name only if the next token is capitalised too
            return i + 1 < len(tokens) and tokens[i + 1][0].isupper()
        return True

    def _number(self, token: str) -> str | None:
        t = token.casefold().rstrip(".,")
        if t[:1].isdigit():
            return t.rstrip("%").replace(",", "")
        return self.NUMBER_WORDS.get(t)

    def _signature(self, text: str) -> dict:
        tokens = self._tokens(text)
        lower = [t.casefold() for t in tokens]
        numbers = {n for n in map(self._number, tokens) if n}
        # "Q3" style tokens: letter prefix + digit -> the digit is the number, the letter is noise.
        return {
            "negations": {t for t in lower if t in self.NEGATIONS or t.endswith("n't")},
            "numbers": numbers,
            "proper": [
                tokens[i].casefold()
                for i in range(len(tokens))
                if self._is_proper(tokens, i) and not self._number(tokens[i])
            ],
            "words": set(lower),
        }

    def same_meaning(self, a: str, b: str) -> bool:
        sa, sb = self._signature(a), self._signature(b)
        if sa["negations"] != sb["negations"]:
            return False
        if sa["numbers"] != sb["numbers"]:
            return False
        if set(sa["proper"]) != set(sb["proper"]):
            return False
        if sa["proper"] != sb["proper"]:  # same names, different order: possible role swap
            return False
        only_a, only_b = sa["words"] - sb["words"], sb["words"] - sa["words"]
        return not any(self._opposite.get(w, set()) & only_b for w in only_a)


_SENTENCE_STARTERS = frozenset(
    {
        "who",
        "what",
        "when",
        "where",
        "which",
        "why",
        "how",
        "is",
        "are",
        "was",
        "were",
        "do",
        "does",
        "did",
        "can",
        "could",
        "will",
        "would",
        "should",
        "has",
        "have",
        "had",
        "the",
        "a",
        "an",
        "in",
        "on",
        "for",
        "to",
        "of",
        "list",
        "name",
        "tell",
        "give",
        "show",
    }
)


class QdrantSemanticCache:
    """Answer cache in its own Qdrant collection.

    Hit = cosine >= threshold AND QuestionGuard agrees AND the entry was stored against the
    current corpus. Entries carry `created_at` (lookups ignore entries older than the TTL — Qdrant
    has no native TTL — and `purge_expired` deletes them) and `corpus_version`: an answer is only
    valid for the documents that existed when it was produced, so indexing new documents
    invalidates every earlier answer.
    """

    def __init__(
        self,
        client: Any,
        collection: str,
        vector_size: int,
        threshold: float,
        ttl_s: int,
        guard: QuestionGuard | None = None,
        clock: Callable[[], float] = time.time,
        corpus_version: Callable[[], int] = lambda: 0,
    ):
        self._client = client
        self._collection = collection
        self._vector_size = vector_size
        self._threshold = threshold
        self._ttl_s = ttl_s
        self._guard = guard or QuestionGuard()
        self._clock = clock
        self._corpus_version = corpus_version
        self._ready = False

    def _ensure(self) -> None:
        if self._ready:
            return
        existing = {c.name for c in self._client.get_collections().collections}
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(size=self._vector_size, distance=Distance.COSINE),
            )
        self._ready = True

    def _valid(self, version: int) -> Filter:
        return Filter(
            must=[
                FieldCondition(key="created_at", range=Range(gte=self._clock() - self._ttl_s)),
                FieldCondition(key="corpus_version", match=MatchValue(value=version)),
            ]
        )

    def lookup(self, question: str, vector: list[float]) -> CachedAnswer | None:
        self._ensure()
        hits = self._client.search(
            collection_name=self._collection,
            query_vector=vector,
            limit=3,
            score_threshold=self._threshold,
            query_filter=self._valid(self._corpus_version()),
        )
        for hit in hits:  # best first; the guard may reject the top one
            if self._guard.same_meaning(question, hit.payload["question"]):
                return CachedAnswer(hit.payload["question"], hit.payload["answer"], hit.score)
        return None

    def store(self, question: str, vector: list[float], answer: dict) -> None:
        self._ensure()
        point_id = str(uuid.uuid5(uuid.NAMESPACE_URL, question.strip().casefold()))
        self._client.upsert(
            collection_name=self._collection,
            points=[
                PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "question": question,
                        "answer": answer,
                        "created_at": self._clock(),
                        "corpus_version": self._corpus_version(),
                    },
                )
            ],
        )

    def purge_expired(self) -> None:
        from qdrant_client.models import FilterSelector

        self._ensure()
        self._client.delete(
            collection_name=self._collection,
            points_selector=FilterSelector(
                filter=Filter(must=[FieldCondition(key="created_at", range=Range(lt=self._clock() - self._ttl_s))])
            ),
        )
