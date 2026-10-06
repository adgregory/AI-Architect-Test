# Technical Assessment — PDF Name Extractor & RAG API

## Overview

You are given a **partially implemented** application that extracts names from scanned PDF documents, identifies their bounding box locations, performs fuzzy matching, and answers questions using a RAG (Retrieval-Augmented Generation) pipeline.

The codebase has **bugs, missing features, and poor engineering practices**. Your task is to fix, complete, and improve it.

## What's Already Implemented

```
app/
├── main.py
├── models/
│   └── schemas.py
├── services/
│   ├── ocr_service.py
│   ├── ner_service.py
│   ├── bbox_service.py
│   ├── fuzzy_service.py
│   ├── embedding_service.py
│   ├── vector_service.py
│   └── rag_service.py
├── api/
│   ├── extract.py
│   └── rag.py
tests/
├── conftest.py
├── test_ocr.py
├── test_ner.py
├── test_bbox.py
├── test_fuzzy.py
├── test_vector.py
├── test_rag.py
├── test_api.py
├── test_architecture.py
└── test_integration.py
```

## Your Tasks

### 1. Fix All Bugs

Run the test suite with `pytest -v`. Multiple tests are **failing due to bugs** in the implementation. Find and fix each bug.

### 2. Refactor Architecture

The current codebase has significant design issues. Think about classes, abstractions, and SOLID principles.

### 3. Implement Software Engineering Best Practices

The codebase is missing critical production practices. Think about logging, configuration (look for hardcoded values), resilience, and validation.

### 4. Complete Missing Features

Some expected functionality is not implemented. The tests will guide you.

### 5. Containerization

- Create a `Dockerfile` for the application.
- Create a `docker-compose.yml` that includes all required services.
- Ensure the container can be built and run with `docker compose up`.

### 6. Design Document

Include a `DESIGN.md` (max 3 pages) covering:
- Component diagram showing all services and dependencies.
- Sequence diagram for both API endpoints.
- Technology choice justifications and tradeoffs.
- Scaling strategy for 1000+ PDFs/hour.
- Failure modes and mitigations.
- LLM routing strategy (if you were to add model selection based on query complexity).

## Evaluation Criteria

| Criteria | Weight |
|----------|--------|
| All tests passing (bugs fixed correctly) | 20% |
| Architecture and design patterns | 20% |
| Software engineering best practices | 20% |
| Code quality and decisions | 15% |
| Design document | 10% |
| Containerization | 10% |
| Bonus: additional tests you write | 5% |

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Run tests (many will fail — that's expected)
pytest -v

# Start Qdrant (required for integration tests)
docker run -p 6333:6333 qdrant/qdrant

# Run the app
uvicorn app.main:app --reload
```

## Findings Report

As part of your submission, fill in the tables below documenting every bug you found and every improvement you made. This helps us understand your debugging process and engineering judgment.

### Bugs Found & Fixed

Baseline ([docs/baseline-test-failures.md](docs/baseline-test-failures.md)): 30 of the 49 provided tests failed before any
app change: 17 from 11 bugs (rows 1–5, 7–10), 5 from missing features (rows 11–12) and 8 from architecture checks
(Architecture table, rows 1–2). Rows 6, 13 and 14 were found outside the provided tests.
Now: all 49 pass, plus 166 added tests (215 total).

| # | File | Bug Description | How You Fixed It |
|---|------|-----------------|------------------|
| 1 | `app/services/ocr_service.py` | `range(1, len(doc))` skipped page 0, so single-page PDFs returned no text (also caused 4 integration failures) | Iterate every page; per-page OCR now shared by every engine (`0cc6b52`) |
| 2 | `app/services/ocr_service.py` | `fitz` documents never closed (resource leak) | Close in `finally` / context manager (`0cc6b52`) |
| 3 | `app/services/ocr_service.py` | Word boxes returned in 150-DPI pixel space, not PDF points | Scale by 72/DPI; text and boxes rendered at the same DPI (`0cc6b52`) |
| 4 | `app/services/ner_service.py` | `PERSON` **and** `ORG` entities kept, so companies were returned as people (both functions) | Person-only filter; later replaced spaCy `sm` with GLiNER, which has 0 false positives on the spike set (`b6a7c2e`) |
| 5 | `app/services/bbox_service.py` | Case-sensitive comparison (`JOHN` ≠ `John`), and punctuation broke matches | Case-fold and strip edge punctuation (`0b97a82`) |
| 6 | `app/services/bbox_service.py` | *(not covered by a test)* Name parts matched independently anywhere in the document, so a box could span unrelated words; an early `break` returned only the first occurrence | Match consecutive words on the same page; return every occurrence in order (`0b97a82`) |
| 7 | `app/services/fuzzy_service.py` | Threshold 70 instead of 90 | Threshold 90, configurable (`7bee72e`) |
| 8 | `app/services/fuzzy_service.py` | `partial_ratio` scores substrings, so partial names passed and full-name typos were under-scored | `token_sort_ratio` on normalised names (`7bee72e`) |
| 9 | `app/services/rag_service.py` | Escaped `{{question}}` in the f-string, so the LLM never saw the question | Interpolate the question (`4e5235a`) |
| 10 | `app/services/vector_service.py` | Point IDs = chunk index, so each document overwrote the previous one | UUID5(document_id:chunk) with the document ID in the payload (`f5fcb60`); document IDs later made content-addressed so re-uploads don't duplicate (`de05dc1`) |
| 11 | `app/api/extract.py` | Non-PDF upload raised `FileDataError` → 500; bad `names` JSON → 500 | 400 on missing `%PDF-` signature, 413 over the size limit, 422 on bad names (`c9e70ee`) |
| 12 | `app/api/*`, `app/models/schemas.py` | Missing features: no `fuzzy_matches`, no `page_number` on boxes, no `sources` on answers, no `/health` | Added to the schemas and routes (`c9e70ee`, `e4af849`) |
| 13 | `pyproject.toml` | *(environment)* NumPy 2 ABI break with spaCy 3.7 stopped the suite from importing | Pinned `numpy<2` while spaCy was in use (`fbf3763`) |
| 14 | agent (new code) | *(found while testing live)* Duplicate passages filled the top-k; a refusal cached before a document was indexed kept being served | Dedupe before top-k (`9b7850c`); never cache refusals, and scope cache entries to the corpus version (`4e113cf`) |

### Architecture & Design Improvements

| # | What You Changed | Why |
|---|------------------|-----|
| 1 | Every service is a class behind a `Protocol` (`OCRService`, `NERService`, `NameLocator`, `NameMatcher`, `EmbeddingService`, `VectorStore`, `LLMClient`, `SemanticCache`, `JobRepository`, `ObjectStorage`, `JobOrchestrator`) | Swappable engines and test doubles without patching (Dependency Inversion / Open-Closed) |
| 2 | Composition root (`app/core/container.py`) built in the FastAPI lifespan; routes get dependencies via `Depends` | Models and clients load once per process, not per request; tests use `dependency_overrides` |
| 3 | Factory registries + uv extras `chosen` / `fallback` (`app/core/factories.py`) | Engine choice is configuration; the fallback stack (Tesseract, spaCy, MiniLM) installs without the chosen one |
| 4 | Engines chosen by measured spikes: PaddleOCR on ONNX (RapidOCR), GLiNER int8, bge-small via fastembed | Evidence in `spikes/01-04`: e.g. OCR CER 14.9% → 2.4%; NER false positives 178 → 0 |
| 5 | `ExtractionSession`: OCR each PDF once per request; NER → box location → fuzzy match share that result | The original code OCR'd the same document several times per request |
| 6 | Async jobs on Temporal (`/api/jobs` → 202, SSE events), separate `cpu` / `io` task queues, per-page parallel OCR, reconciler schedule | Large and batch PDFs no longer block HTTP; durable retries; CPU workers scale without growing DB connections |
| 7 | Agent service (Strands, AgentCore contract) with a Qdrant semantic answer cache + `QuestionGuard` | Answering scales and deploys independently; provider is per environment (Gemini/Vertex locally, Bedrock in AWS); cache has 0 false hits in spike 04 |
| 8 | One `DocumentIndexer`; content-addressed document IDs; `/api/extract` indexes in the background | No duplicated chunk/embed/store code; each PDF indexed once; anything extracted is answerable |
| 9 | Postgres with pooled psycopg + LISTEN/NOTIFY; only `api` and the `io` worker own pools | A fixed, predictable connection budget |
| 10 | `infra/` Pulumi (Python): VPC, RDS, S3, ECS Fargate, AgentCore, outbox → Debezium → Kinesis → Lambda | Production target, validated offline (mocked unit tests, CrossGuard policies, previews) |

### Engineering Best Practices Added

| # | Practice | Where / How You Implemented It |
|---|----------|-------------------------------|
| 1 | Typed configuration, no hard-coded values | `app/core/config.py` (pydantic-settings, env / `.env`, validated ranges) |
| 2 | Structured logging with request IDs | structlog JSON (`app/core/logging.py`); request-ID middleware binds context to every log line |
| 3 | Input validation | `app/api/uploads.py`: PDF signature, size limit (413), names schema (422); `max_pages` rejected as non-retryable |
| 4 | Resilience | Temporal retry policies (transient vs non-retryable), heartbeats, reconciler; HTTP/Qdrant/LLM timeouts; bounded DB pools → 503 + `Retry-After`; background work is best effort |
| 5 | Error handling | Dependency failures (engine missing, agent down, DB pool exhausted) mapped to 503 + `Retry-After` in `app/main.py`; anything else is a generic 500 with the detail in the logs only |
| 6 | Tests (bonus) | 166 added: unit (fakes), API (dependency overrides), contract (every engine against the same expectations), workflows (Temporal time-skipping server), Postgres repository, end-to-end pipeline, Pulumi mocks |
| 7 | Dependency management | uv + `pyproject.toml` with a lockfile; extras per stack; `uv.lock` checked in pre-commit |
| 8 | Static checks | pre-commit: ruff (lint + format), mypy (pydantic plugin), detect-secrets, hygiene hooks |
| 9 | Containerisation | Multi-stage `Dockerfile` (`STACK` build arg, models baked in, non-root, healthcheck, offline); `Dockerfile.agent`; `docker-compose.yml` with the full stack and migrations |
| 10 | Database migrations | Alembic (`migrations/`), run by a one-shot compose service |
| 11 | Secrets hygiene | ADC mounted read-only, never in images; no GCP credentials in AWS; least-privilege IAM enforced by policy |
| 12 | Developer workflow | `Taskfile.dist.yaml` (`setup`, `test:*`, `lint`, `stack:up/down`, `job:demo`, `infra:check`); ADR + spike write-ups + `DESIGN.md` |

## Submission

Create a branch and make a pull request to this repository.
