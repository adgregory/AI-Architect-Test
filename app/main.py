import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from app.api.extract import router as extract_router
from app.api.rag import router as rag_router
from app.core.config import get_settings
from app.core.container import Container
from app.core.factories import MissingEngineError
from app.core.logging import configure_logging, get_logger
from app.models.schemas import HealthResponse

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Build the container and load every model once, before serving traffic."""
    settings = get_settings()
    configure_logging(settings)
    container = Container(settings)
    started = time.perf_counter()
    await run_in_threadpool(container.warm_up)  # model loading is blocking: keep it off the event loop
    log.info("startup.ready", seconds=round(time.perf_counter() - started, 2),
             ocr=settings.ocr_engine, ner=settings.ner_engine, embeddings=settings.embedding_engine)
    app.state.container = container
    try:
        yield
    finally:
        container.close()
        log.info("shutdown.done")


app = FastAPI(title="PDF Name Extractor & RAG API", lifespan=lifespan)

app.include_router(extract_router, prefix="/api", tags=["extraction"])
app.include_router(rag_router, prefix="/api", tags=["rag"])


@app.middleware("http")
async def request_context(request: Request, call_next):
    """Bind a request ID to every log line of the request and echo it back."""
    request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id, method=request.method, path=request.url.path)
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception("request.failed")
        raise
    log.info("request.done", status=response.status_code,
             duration_ms=round((time.perf_counter() - started) * 1000, 1))
    response.headers["x-request-id"] = request_id
    return response


@app.exception_handler(MissingEngineError)
async def missing_engine(_: Request, exc: MissingEngineError) -> JSONResponse:
    log.error("engine.unavailable", error=str(exc))
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.get("/health", response_model=HealthResponse, tags=["ops"])
def health() -> HealthResponse:
    """Liveness probe: the process is up and serving requests."""
    return HealthResponse(status="ok")
