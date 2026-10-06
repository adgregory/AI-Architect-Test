# syntax=docker/dockerfile:1.7
#
# PDF Name Extractor & RAG API.
#   STACK=chosen   (default) PaddleOCR ONNX + GLiNER int8 + bge-small via fastembed
#   STACK=fallback           Tesseract + spaCy + sentence-transformers
#   STACK=agent              Strands agent service (AgentCore contract) + bge-small embeddings only
#
#   docker build --build-arg STACK=chosen -t pdf-name-extractor:chosen .

ARG PYTHON_VERSION=3.12
ARG STACK=chosen

# --------------------------------------------------------------------------- #
FROM python:${PYTHON_VERSION}-slim AS base
ARG STACK
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STACK=${STACK}
# opencv (RapidOCR) needs libGL/glib; the fallback stack needs the tesseract binary.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
 && if [ "$STACK" = "fallback" ]; then \
      apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-eng; \
    fi \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /app

# --------------------------------------------------------------------------- #
FROM base AS deps
ARG STACK
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_PROJECT_ENVIRONMENT=/app/.venv
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra "${STACK}"

# --------------------------------------------------------------------------- #
FROM deps AS models
ARG STACK
ENV MODELS_DIR=/app/.models \
    HF_HOME=/app/.models/hf \
    LOG_JSON=true
COPY app ./app
COPY scripts/prepare_models.py ./scripts/
RUN if [ "$STACK" = "fallback" ]; then \
      printf 'OCR_ENGINE=tesseract\nNER_ENGINE=spacy\nEMBEDDING_ENGINE=sentence-transformers\n' > .env; \
    else \
      printf 'OCR_ENGINE=rapidocr\nNER_ENGINE=gliner\nEMBEDDING_ENGINE=fastembed\n' > .env; \
    fi
# Download / export models with the same code path the app's lifespan uses
# (the agent only needs the embedding model).
RUN if [ "$STACK" = "agent" ]; then components=embeddings; else components=ocr,ner,embeddings; fi \
 && /app/.venv/bin/python scripts/prepare_models.py --components "$components" \
 && if [ "$STACK" = "chosen" ]; then rm -rf /app/.models/hf; fi  # GLiNER source weights only needed for export

# --------------------------------------------------------------------------- #
FROM base AS runtime
ARG STACK
ENV PATH=/app/.venv/bin:$PATH \
    MODELS_DIR=/app/.models \
    HF_HOME=/app/.models/hf \
    HF_HUB_OFFLINE=1 \
    LOG_JSON=true
RUN useradd --create-home --uid 10001 app \
 && mkdir -p /data && chown app:app /data   # shared storage volume (api + workers)
COPY --from=deps /app/.venv /app/.venv
COPY --from=models --chown=app:app /app/.models /app/.models
COPY --from=models /app/.env /app/.env
COPY alembic.ini ./
COPY migrations ./migrations
COPY app ./app
USER app
EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=3s --start-period=60s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status != 200)"
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
