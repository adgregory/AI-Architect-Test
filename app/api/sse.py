"""Server-sent events formatting (one place for the wire format)."""

from __future__ import annotations

import json

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


def sse_comment(text: str) -> str:
    return f": {text}\n\n"
