"""Retrieval-augmented generation: chunking, retrieval and answer generation."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import httpx

from app.core.config import get_settings
from app.services.embedding_service import get_query_embedding
from app.services.vector_service import search_similar

_settings = get_settings()
OPENAI_API_KEY = _settings.llm_api_key.get_secret_value() if _settings.llm_api_key else None
NO_INFORMATION = "No relevant information found."

PROMPT_TEMPLATE = """Based on the following context, answer the question.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:"""


@runtime_checkable
class LLMClient(Protocol):
    def complete(self, prompt: str) -> str: ...


class OpenAICompatibleChatClient:
    """Chat Completions over HTTP — works with OpenAI and any OpenAI-compatible endpoint."""

    def __init__(self, api_key: str | None, base_url: str, model: str, timeout_s: float):
        self._url = f"{base_url.rstrip('/')}/chat/completions"
        self._model = model
        self._timeout_s = timeout_s
        # No key, no header (local OpenAI-compatible servers often need none).
        self._headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}

    def complete(self, prompt: str) -> str:
        # Module-level httpx.post: the provided tests patch it. The agent service is the
        # production answering path; this client is the in-process reference implementation.
        response = httpx.post(
            self._url,
            headers=self._headers,
            json={"model": self._model, "messages": [{"role": "user", "content": prompt}]},
            timeout=self._timeout_s,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]


class TextChunker:
    """Splits text into chunks of about `chunk_size` characters on word boundaries.

    Boundaries fall after whitespace, so no word is cut and "".join(chunks) == text.
    A single word longer than chunk_size becomes its own chunk.
    """

    def chunk(self, text: str, chunk_size: int = 500) -> list[str]:
        chunks, start = [], 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
                if cut > start:
                    end = cut + 1
                else:  # word longer than chunk_size: extend to its end
                    nxt = min((i for i in (text.find(" ", end), text.find("\n", end)) if i != -1), default=len(text))
                    end = nxt + 1 if nxt < len(text) else len(text)
            chunks.append(text[start:end])
            start = end
        return chunks


class RAGService:
    """Embed the question, retrieve top-k chunks, and ask the LLM to answer from them."""

    def __init__(
        self,
        embed_query: Callable[[str], list[float]],
        retrieve: Callable[..., list[dict]],
        llm: LLMClient,
        top_k: int = 3,
    ):
        self._embed_query = embed_query
        self._retrieve = retrieve
        self._llm = llm
        self._top_k = top_k

    def answer(self, question: str) -> dict:
        chunks = self._retrieve(self._embed_query(question), top_k=self._top_k)
        if not chunks:
            return {"answer": NO_INFORMATION, "sources": []}
        context = "\n\n".join(c["text"] for c in chunks)
        answer = self._llm.complete(PROMPT_TEMPLATE.format(context=context, question=question))
        return {"answer": answer, "sources": [c["text"] for c in chunks]}


# --------------------------------------------------------------------------- #
# Functional API (backwards compatible). The adapter resolves its collaborators
# (embedding, retrieval, API key) at call time.
# --------------------------------------------------------------------------- #
class _ModuleRAG:
    def generate_answer(self, question: str) -> dict:
        llm = OpenAICompatibleChatClient(
            api_key=OPENAI_API_KEY,
            base_url=_settings.llm_base_url,
            model=_settings.llm_model,
            timeout_s=_settings.llm_timeout_s,
        )
        return RAGService(get_query_embedding, search_similar, llm, _settings.retrieval_top_k).answer(question)


chunk_text = TextChunker().chunk
generate_answer = _ModuleRAG().generate_answer
