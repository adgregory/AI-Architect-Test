# Minimal CPU-only image per OCR engine, used only to measure image size.
# Build context: spikes/01-ocr.   docker build -f docker/Dockerfile --build-arg ENGINE=<engine> .
FROM python:3.12-slim

ARG ENGINE
ARG APT_PACKAGES=""
ENV UV_LINK_MODE=copy UV_COMPILE_BYTECODE=0 PYTHONDONTWRITEBYTECODE=1

RUN if [ -n "$APT_PACKAGES" ]; then \
      apt-get update && apt-get install -y --no-install-recommends $APT_PACKAGES && rm -rf /var/lib/apt/lists/*; \
    fi

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /spike
COPY engines/${ENGINE}/pyproject.toml engines/${ENGINE}/uv.lock engines/${ENGINE}/
RUN cd engines/${ENGINE} && uv sync --frozen --no-cache

COPY common/ common/
COPY engines/${ENGINE}/adapter.py engines/${ENGINE}/
COPY docker/prefetch.py docker/
# Bake models into the image (what a production image would do).
RUN engines/${ENGINE}/.venv/bin/python docker/prefetch.py ${ENGINE}
