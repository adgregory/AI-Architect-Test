"""Agent service built to the AgentCore Runtime contract (POST /invocations, GET /ping, port 8080).

Locally it runs as a container in docker compose; in AWS the same image deploys to AgentCore
Runtime. Payload: {"question": "..."}; response: server-sent events, one AnswerEvent per line.

    python -m app.agent.server
"""

from __future__ import annotations

import threading
import time
from functools import cached_property

from bedrock_agentcore.runtime import BedrockAgentCoreApp

from app.agent.answering import AnswerService, strands_agent_factory
from app.agent.providers import ModelProviderFactory
from app.core.config import Settings, get_settings
from app.core.container import Container
from app.core.factories import default_qdrant_client
from app.core.logging import configure_logging, get_logger
from app.services.cache_service import QdrantSemanticCache

log = get_logger(__name__)


class AgentRuntime:
    """Builds the answering service once per process (models, clients, cache)."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._container = Container(settings)
        self._lock = threading.Lock()

    @cached_property
    def answers(self) -> AnswerService:
        with self._lock:
            s = self.settings
            cache = None
            if s.answer_cache_enabled:
                cache = QdrantSemanticCache(
                    default_qdrant_client(s),
                    s.answer_cache_collection,
                    vector_size=s.embedding_dim,
                    threshold=s.answer_cache_threshold,
                    ttl_s=s.answer_cache_ttl_s,
                )
            return AnswerService(
                embed_query=self._container.embeddings.embed_query,
                retrieve=lambda vector, top_k: self._container.vector_store.search(vector, top_k),
                agent_factory=strands_agent_factory(ModelProviderFactory.create(s)),
                cache=cache,
                top_k=s.retrieval_top_k,
            )

    def warm_up(self) -> None:
        started = time.perf_counter()
        self._container.embeddings.embed_query("warm-up")
        _ = self.answers  # build the model client and cache
        log.info(
            "agent.ready",
            seconds=round(time.perf_counter() - started, 2),
            provider=self.settings.llm_provider,
            model=self.settings.gemini_model
            if self.settings.llm_provider == "gemini"
            else self.settings.bedrock_model_id,
        )


app = BedrockAgentCoreApp()
_runtime: AgentRuntime | None = None


def runtime() -> AgentRuntime:
    global _runtime
    if _runtime is None:
        _runtime = AgentRuntime(get_settings())
    return _runtime


@app.entrypoint
async def invoke(payload: dict):
    question = (payload or {}).get("question", "").strip()
    if not question:
        yield {"type": "error", "error": "payload must include a non-empty 'question'"}
        return
    async for event in runtime().answers.stream(question):
        yield event


if __name__ == "__main__":
    settings = get_settings()
    configure_logging(settings)
    runtime().warm_up()
    app.run(port=8080, host="0.0.0.0")  # noqa: S104 - container entrypoint; reached via the service network
