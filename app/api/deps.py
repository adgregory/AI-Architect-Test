"""FastAPI dependencies: read the container built by the app lifespan."""

from __future__ import annotations

import threading

from fastapi import Depends, Request

from app.core.config import Settings, get_settings
from app.core.container import Container
from app.services.extraction_service import ExtractionSession

_fallback_lock = threading.Lock()


def get_container(request: Request) -> Container:
    """The lifespan puts the container on app.state. If the app runs without its
    lifespan (e.g. `TestClient(app)` outside a `with` block), build it once, lazily."""
    container = getattr(request.app.state, "container", None)
    if container is None:
        with _fallback_lock:
            container = getattr(request.app.state, "container", None)
            if container is None:
                container = Container(get_settings())
                request.app.state.container = container
    return container


def get_app_settings(container: Container = Depends(get_container)) -> Settings:
    return container.settings


def get_extraction_session(container: Container = Depends(get_container)) -> ExtractionSession:
    return container.extraction_session()


def get_rag_service(container: Container = Depends(get_container)):
    """The configured answering backend: exposes answer(question) (and stream() for the agent)."""
    return container.rag
