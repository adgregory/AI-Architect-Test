"""Agent service: answering logic, the AgentCore server contract, and backend pass-through —
all in-process with a scripted fake model (no network, no LLM)."""

import json
from types import SimpleNamespace

import httpx
import pytest
from fastapi.testclient import TestClient
from starlette.testclient import TestClient as StarletteClient

import app.agent.server as server
from app.agent.answering import NO_INFORMATION, AnswerService, build_prompt
from app.api.deps import get_container
from app.main import app as backend_app
from app.services.agent_client import AgentAnswerClient, AgentUnavailableError
from tests.api.test_routes import FakeContainer
from tests.fakes import FakeEmbeddings, InMemoryVectorStore

DOCS = [
    "Robert Chen has been promoted to Vice President of Engineering.",
    "The cloud migration to AWS is expected to be completed by Q3 2024.",
]


class ScriptedAgent:
    """Stands in for a Strands Agent: yields text deltas, then a result with a stop reason."""

    def __init__(self, tokens, stop_reason="end_turn", log=None):
        self.tokens, self.stop_reason, self.log = tokens, stop_reason, log if log is not None else []

    async def stream_async(self, prompt):
        self.log.append(prompt)
        for t in self.tokens:
            yield {"data": t}
        yield {"result": SimpleNamespace(stop_reason=self.stop_reason)}


class DictCache:
    """Exact-question cache (the guard/threshold logic is tested in test_cache.py)."""

    def __init__(self):
        self.entries = {}

    def lookup(self, question, vector):
        hit = self.entries.get(question.lower())
        return SimpleNamespace(question=question, answer=hit, score=1.0) if hit else None

    def store(self, question, vector, answer):
        self.entries[question.lower()] = answer


def make_service(tokens=("Robert ", "Chen."), stop_reason="end_turn", docs=DOCS, cache=None):
    embeddings, store = FakeEmbeddings(), InMemoryVectorStore(score_threshold=0.1)
    if docs:
        store.upsert(docs, embeddings.embed_documents(docs))
    prompts, agents = [], []

    def factory():
        agents.append(ScriptedAgent(tokens, stop_reason, prompts))
        return agents[-1]

    service = AnswerService(embeddings.embed_query, store.search, factory, cache=cache, top_k=1)
    return service, prompts, agents


async def collect(service, question):
    return [e async for e in service.stream(question)]


class TestAnswerService:
    async def test_streams_sources_tokens_then_done(self):
        service, prompts, _ = make_service()
        events = await collect(service, "Who was promoted to Vice President of Engineering?")
        assert [e["type"] for e in events] == ["sources", "token", "token", "done"]
        assert events[0]["sources"] == [DOCS[0]]
        assert events[-1] == {
            "type": "done",
            "answer": "Robert Chen.",
            "sources": [DOCS[0]],
            "cached": False,
            "stop_reason": "end_turn",
        }
        assert "Question: Who was promoted" in prompts[0] and f"[1] {DOCS[0]}" in prompts[0]

    async def test_cache_hit_skips_retrieval_and_model(self):
        cache = DictCache()
        service, prompts, agents = make_service(cache=cache)
        await collect(service, "Who was promoted?")
        events = await collect(service, "who was promoted?")
        assert events[-1]["cached"] is True and events[-1]["answer"] == "Robert Chen."
        assert len(agents) == 1 and len(prompts) == 1

    async def test_truncated_answers_are_not_cached(self):
        cache = DictCache()
        service, _, _ = make_service(stop_reason="max_tokens", cache=cache)
        events = await collect(service, "Who was promoted?")
        assert events[-1]["stop_reason"] == "max_tokens" and cache.entries == {}

    async def test_refusals_are_not_cached(self):
        cache = DictCache()
        service, _, _ = make_service(tokens=("I don't have enough information.",), cache=cache)
        events = await collect(service, "Who has been promoted to Vice President of Engineering")
        assert events[-1]["stop_reason"] == "end_turn"  # the model answered...
        assert events[-1]["answer"] == "I don't have enough information." and cache.entries == {}  # ...not cached

    async def test_no_context_never_calls_the_model(self):
        service, prompts, agents = make_service(docs=[])
        events = await collect(service, "Anything?")
        assert events[-1]["answer"] == NO_INFORMATION and events[-1]["stop_reason"] == "no_context"
        assert agents == [] and prompts == []

    async def test_fresh_agent_per_question(self):
        service, _, agents = make_service()
        await collect(service, "Who was promoted?")
        await collect(service, "When is the migration due?")
        assert len(agents) == 2 and agents[0] is not agents[1]

    async def test_duplicate_passages_are_dropped_before_top_k(self):
        embeddings, store = FakeEmbeddings(), InMemoryVectorStore(score_threshold=0.0)
        docs = [DOCS[0], DOCS[0], "Robert Chen  has been promoted to Vice President of Engineering.", DOCS[1]]
        store.upsert(docs, embeddings.embed_documents(docs))
        service = AnswerService(embeddings.embed_query, store.search, lambda: ScriptedAgent(["ok"]), top_k=2)
        events = await collect(service, "Who has been promoted to Vice President of Engineering")
        assert events[0]["sources"] == [DOCS[0], DOCS[1]]

    def test_prompt_numbers_passages(self):
        prompt = build_prompt("Q?", [{"text": "a"}, {"text": "b"}])
        assert prompt == "Context passages:\n[1] a\n\n[2] b\n\nQuestion: Q?"


@pytest.fixture
def agent_runtime(monkeypatch):
    service, prompts, _ = make_service()
    monkeypatch.setattr(server, "_runtime", SimpleNamespace(answers=service))
    return prompts


def sse_events(body: str) -> list[dict]:
    return [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]


class TestAgentCoreServer:
    def test_ping(self, agent_runtime):
        assert StarletteClient(server.app).get("/ping").status_code == 200

    def test_invocations_stream_answer_events(self, agent_runtime):
        r = StarletteClient(server.app).post("/invocations", json={"question": "Who was promoted?"})
        assert r.headers["content-type"].startswith("text/event-stream")
        events = sse_events(r.text)
        assert [e["type"] for e in events] == ["sources", "token", "token", "done"]

    def test_rejects_empty_question(self, agent_runtime):
        events = sse_events(StarletteClient(server.app).post("/invocations", json={"question": " "}).text)
        assert events == [{"type": "error", "error": "payload must include a non-empty 'question'"}]


@pytest.fixture
def backend(agent_runtime):
    """Backend API whose answering backend is the real AgentCore app, reached over ASGI."""
    client = AgentAnswerClient(
        "http://agent",
        StarletteClient(server.app, base_url="http://agent"),
        httpx.AsyncClient(transport=httpx.ASGITransport(app=server.app), base_url="http://agent"),
    )
    container = FakeContainer()
    container.rag = client
    backend_app.dependency_overrides[get_container] = lambda: container
    yield TestClient(backend_app)
    backend_app.dependency_overrides.clear()


QUESTION = "Who has been promoted to Vice President of Engineering"


class TestBackendToAgent:
    def test_ask_collects_the_agent_stream(self, backend):
        r = backend.post("/api/ask", json={"question": QUESTION})
        assert r.status_code == 200
        assert r.json() == {"answer": "Robert Chen.", "sources": [DOCS[0]]}

    def test_ask_stream_passes_events_through(self, backend):
        with backend.stream("POST", "/api/ask/stream", json={"question": QUESTION}) as r:
            assert r.headers["content-type"].startswith("text/event-stream")
            body = "".join(r.iter_text())
        assert [e["type"] for e in sse_events(body)] == ["sources", "token", "token", "done"]
        assert "event: token" in body

    def test_agent_errors_surface_as_503(self, backend, monkeypatch):
        monkeypatch.setattr(server, "_runtime", None)
        monkeypatch.setattr(server, "runtime", lambda: (_ for _ in ()).throw(RuntimeError("model down")))
        r = backend.post("/api/ask", json={"question": "Who was promoted?"})
        assert r.status_code == 503


def test_client_rejects_streams_without_a_final_answer():
    body = 'data: {"type": "token", "text": "x"}\n\n'
    transport = httpx.MockTransport(lambda req: httpx.Response(200, text=body))
    client = AgentAnswerClient(
        "http://agent", httpx.Client(transport=transport), httpx.AsyncClient(transport=transport)
    )
    with pytest.raises(AgentUnavailableError):
        client.answer("Q?")
