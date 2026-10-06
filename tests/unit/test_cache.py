"""QuestionGuard rules and QdrantSemanticCache behaviour (fake Qdrant client, injected clock)."""

from types import SimpleNamespace

import pytest

from app.services.cache_service import QdrantSemanticCache, QuestionGuard, SemanticCache
from tests.fakes import FakeQdrantClient


class TestQuestionGuard:
    guard = QuestionGuard()

    @pytest.mark.parametrize(
        "a, b",
        [
            ("Who is the new head of engineering?", "who is the new head of engineering, please?"),
            ("What is the overall research budget?", "What's the overall research budget?"),
            ("How much did revenue grow, twelve percent?", "How much did revenue grow, 12%?"),
            ("When is the board meeting again?", "Can you tell me when is the board meeting again?"),
            ("Who reviewed the report of Dr. Olivia Chambers?", "Who reviewed Olivia Chambers's report?"),
        ],
    )
    def test_same_meaning(self, a, b):
        assert self.guard.same_meaning(a, b)

    @pytest.mark.parametrize(
        "a, b, why",
        [
            ("Was the motion approved?", "Was the motion not approved?", "negation"),
            ("Is the NIH grant worth $600K?", "Is the NIH grant worth $750K?", "number"),
            ("Did Robert Chen present at NeurIPS?", "Did James Chen present at NeurIPS?", "entity"),
            (
                "Does Sarah Williams report to James Anderson?",
                "Does James Anderson report to Sarah Williams?",
                "role swap",
            ),
            ("Will the ML division expand in 2025?", "Will the ML division shrink in 2025?", "antonym"),
            ("Did revenue increase last year?", "Did revenue decrease last year?", "antonym"),
        ],
    )
    def test_meaning_changes_are_rejected(self, a, b, why):
        assert not self.guard.same_meaning(a, b), why


class SearchableFakeQdrant(FakeQdrantClient):
    """Adds cosine search with filters to the shared fake."""

    def __init__(self):
        super().__init__()
        self.points = {}
        self.deleted_filters = []

    def upsert(self, **kwargs):
        super().upsert(**kwargs)
        for p in kwargs["points"]:
            self.points[p.id] = p

    @staticmethod
    def _matches(payload, flt) -> bool:
        for cond in flt.must:
            value = payload.get(cond.key)
            if cond.range is not None and not (value is not None and value >= cond.range.gte):
                return False
            if cond.match is not None and value != cond.match.value:
                return False
        return True

    def search(self, collection_name, query_vector, limit, score_threshold, query_filter):
        hits = []
        for p in self.points.values():
            score = sum(x * y for x, y in zip(p.vector, query_vector, strict=True))
            if score >= score_threshold and self._matches(p.payload, query_filter):
                hits.append(SimpleNamespace(payload=p.payload, score=score))
        return sorted(hits, key=lambda h: -h.score)[:limit]

    def delete(self, collection_name, points_selector):
        self.deleted_filters.append(points_selector.filter)


@pytest.fixture
def clock():
    return SimpleNamespace(now=1000.0)


@pytest.fixture
def corpus():
    return SimpleNamespace(version=12)


@pytest.fixture
def cache(clock, corpus):
    return QdrantSemanticCache(
        SearchableFakeQdrant(),
        "answer_cache",
        vector_size=2,
        threshold=0.9,
        ttl_s=60,
        clock=lambda: clock.now,
        corpus_version=lambda: corpus.version,
    )


ANSWER = {"answer": "Robert Chen", "sources": ["chunk"]}


def test_implements_protocol(cache):
    assert isinstance(cache, SemanticCache)


def test_hit_requires_similarity_and_guard(cache):
    cache.store("Who is the new head of engineering?", [1.0, 0.0], ANSWER)
    hit = cache.lookup("who is the new head of engineering, please?", [1.0, 0.0])
    assert hit is not None and hit.answer == ANSWER
    assert cache.lookup("Who is the new head of engineering?", [0.0, 1.0]) is None  # dissimilar vector


def test_guard_blocks_a_similar_but_different_question(cache):
    cache.store("Does Sarah Williams report to James Anderson?", [1.0, 0.0], ANSWER)
    assert cache.lookup("Does James Anderson report to Sarah Williams?", [1.0, 0.0]) is None


def test_expired_entries_are_ignored_and_purged(cache, clock):
    cache.store("Who is the new head of engineering?", [1.0, 0.0], ANSWER)
    clock.now += 61
    assert cache.lookup("Who is the new head of engineering?", [1.0, 0.0]) is None
    cache.purge_expired()
    assert cache._client.deleted_filters[0].must[0].range.lt == clock.now - 60


def test_same_question_overwrites_its_entry(cache):
    cache.store("Who leads design?", [1.0, 0.0], {"answer": "old", "sources": []})
    cache.store("  who leads design?", [1.0, 0.0], {"answer": "new", "sources": []})
    assert len(cache._client.points) == 1
    assert cache.lookup("Who leads design?", [1.0, 0.0]).answer["answer"] == "new"


def test_collection_created_once(cache):
    cache.store("Q?", [1.0, 0.0], ANSWER)
    cache.lookup("Q?", [1.0, 0.0])
    assert len(cache._client.created) == 1


def test_indexing_new_documents_invalidates_earlier_answers(cache, corpus):
    cache.store("What changed in the organisation?", [1.0, 0.0], ANSWER)
    assert cache.lookup("What changed in the organisation?", [1.0, 0.0]) is not None
    corpus.version = 14  # a new document was indexed
    assert cache.lookup("What changed in the organisation?", [1.0, 0.0]) is None
