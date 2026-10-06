"""Backend client for the agent service (AgentCore Runtime contract: POST /invocations, SSE)."""

from __future__ import annotations

import json
from typing import AsyncIterator, Iterable, Iterator

import httpx


class AgentUnavailableError(RuntimeError):
    """The agent service failed or returned no final answer."""


class AgentAnswerClient:
    """Talks to the agent over its streaming /invocations endpoint.

    The HTTP clients are injected and shared per process (connection pooling, limits and
    timeouts configured once), never created per request.
    """

    def __init__(self, base_url: str, http: httpx.Client, ahttp: httpx.AsyncClient):
        self._url = f"{base_url.rstrip('/')}/invocations"
        self._http = http
        self._ahttp = ahttp

    @staticmethod
    def _events(lines: Iterable[str]) -> Iterator[dict]:
        for line in lines:
            if line.startswith("data:"):
                yield json.loads(line[5:].strip())

    @staticmethod
    def _check(event: dict) -> None:
        if event.get("type") == "error" or "error" in event and "type" not in event:
            raise AgentUnavailableError(event.get("error", "agent error"))

    def answer(self, question: str) -> dict:
        """Collect the stream into {answer, sources} (the /api/ask JSON contract)."""
        with self._http.stream("POST", self._url, json={"question": question}) as response:
            response.raise_for_status()
            for event in self._events(response.iter_lines()):
                self._check(event)
                if event.get("type") == "done":
                    return {"answer": event["answer"], "sources": event["sources"]}
        raise AgentUnavailableError("agent stream ended without a final answer")

    async def stream(self, question: str) -> AsyncIterator[dict]:
        """Relay the agent's events as they arrive (pass-through streaming)."""
        async with self._ahttp.stream("POST", self._url, json={"question": question}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data:"):
                    event = json.loads(line[5:].strip())
                    self._check(event)
                    yield event
