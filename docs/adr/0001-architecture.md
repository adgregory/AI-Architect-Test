# ADR 0001 — Solution architecture

- **Status:** accepted (model choices decided by spikes 01–04)
- **Date:** 2026-10-05

## Context

The service extracts person names and their bounding boxes from scanned PDFs,
fuzzy-matches them against names supplied in the request, indexes document
text for retrieval, and answers questions with RAG. Constraints:

- Runs locally with Docker Compose; the AWS target is modelled in `infra/` (Pulumi).
- Production targets AWS: the agent runs on AgentCore Runtime with a Bedrock model. The only
  account available for local testing is GCP, so locally the agent can use Gemini on Vertex AI
  via Application Default Credentials (ADC); without credentials it answers without an LLM.
- The test suite fixes part of the contract: a synchronous `POST /api/extract`
  response shape, and a Qdrant-backed `VectorStore`.
- Every backend sits behind an interface so it can be swapped (e.g. hybrid search).

## Decision

### Services (docker compose)

| Container | Responsibility |
|-----------|----------------|
| `api` | FastAPI backend: validation, sync extraction, job API, SSE, relays agent stream |
| `worker-cpu` | Temporal worker on the `cpu` queue: OCR, NER, box location, fuzzy matching, embedding — no DB connections |
| `worker-io` | Temporal worker on the `io` queue: job status/result writes, `NOTIFY`, Qdrant upserts — sole owner of the worker-side DB pool |
| `agent` | Strands agent, AgentCore runtime contract (Gemini on Vertex locally; Bedrock on AgentCore in AWS) |
| `postgres` | `jobs`, Temporal persistence |
| `temporal`, `temporal-ui` | Workflow orchestration |
| `qdrant` | Document-chunk vectors and the semantic answer cache |

### API

- `POST /api/extract` — synchronous extraction (kept: test contract, small files).
- `POST /api/jobs` → `202 {job_id}`; `GET /api/jobs/{id}`; `GET /api/jobs/{id}/events` (SSE).
- `POST /api/ask` — JSON `{answer, sources}`; `POST /api/ask/stream` — the same answer as SSE
  (`sources`, `token`…, `done`).
- `POST /api/ingest` — synchronous OCR + indexing of a PDF for `/api/ask`.
- `GET /health`.

Sync and async extraction call the same service classes; only the entry point differs.

### Async extraction: direct orchestration (implemented) — outbox (production target)

**Implemented (local / assessment scope):**

1. `POST /api/jobs` validates the upload, stores the PDF, inserts the `jobs` row
   (`status = queued`), **starts `ExtractNamesWorkflow` directly** with `workflow_id = job_id`,
   and returns `202 {job_id}`.
2. If the start call fails (Temporal unreachable), the job stays `queued`. A **reconciler**
   (a Temporal schedule) periodically re-starts `queued` jobs older than a threshold; the
   workflow-ID uniqueness makes re-starts idempotent. This covers the "row written, workflow
   never started" gap without extra infrastructure.
3. The workflow's last activity writes the result to `jobs.result` (jsonb), sets the status,
   and issues `NOTIFY job_events` (sent on every status change); open SSE connections push it.

**Production target (modelled in `infra/` Pulumi, not run locally):** transactional outbox —
the job row and an `outbox` row in one transaction; **Debezium Server** reads the outbox via
logical replication and publishes to **Kinesis**; a **dispatcher Lambda** (partial batch
failures, bisect-on-error, SQS DLQ + alarm) starts the workflow idempotently. It removes the
dual-write window entirely and lets other consumers subscribe to job events. Its main risk is
WAL retention when Debezium stalls (mitigations: `max_slot_wal_keep_size`, slot-lag and disk
alarms, durable Debezium offsets). Deferred from the local build because it adds four
components whose guarantee the assessment doesn't exercise; the reconciler gives most of the
safety at a fraction of the cost.

### Temporal workflow

```
ExtractNamesWorkflow(job_id)
  1. prepare_document                  (cpu)  validate the PDF, count pages, content-addressed document id
  2. mark_running                      (io)
  3. ocr_page × N, in parallel         (cpu)  render + OCR one page: text + word boxes → stored ref
  4. extract_and_match                 (cpu)  NER → name boxes (PDF space) → fuzzy match (≥ 90%)
  5. complete_job                      (io)   jobs.result, status, NOTIFY   (any failure → fail_job)
  child IndexDocumentWorkflow(index-<document id>)  (abandon on parent close)  chunk + embed (cpu) → upsert Qdrant (io)
```

- **References, not payloads:** page images, OCR words and intermediate results are
  stored in object storage; activities exchange paths (Temporal payloads ≈ 2 MB max).
- **Task queues:** `cpu` (OCR/NER/embedding) and `io` (DB/Qdrant) are served by separate
  worker deployments that scale independently. Models load once per `cpu` worker process;
  only `io` workers hold database connections (see Connection management).
- **Retries:** transient errors retry with backoff; invalid input (corrupt PDF, page
  limit) is non-retryable and fails fast. OCR activities heartbeat.
- **Failure:** on terminal failure the workflow sets `status = failed` with a reason and
  still notifies, so clients never wait forever.
- **Indexing** runs as a child workflow so RAG indexing can't delay or fail extraction.

### Agent service and streaming

- Strands Agents; the model provider is selected by configuration, per environment:

  | Environment | Provider | Credentials |
  |-------------|----------|-------------|
  | Local (compose) | `GeminiModel` (`strands-agents[gemini]`) with a pre-built `google.genai.Client(vertexai=True, project, location="global")` | Opt-in: ADC mounted read-only via `docker-compose.gcp.yml`. Without credentials, `LLM_PROVIDER=auto` answers with the retrieved sentence closest to the question (no LLM) |
  | AWS (AgentCore Runtime) | `BedrockModel` — the production model | AgentCore runtime execution role (`bedrock:InvokeModel`, `bedrock:InvokeModelWithResponseStream` scoped to the configured model ARN) |

  No GCP credentials exist in AWS (no Workload Identity Federation needed). Prompts and
  answer quality are validated against both providers since local and production models
  differ. The Bedrock model ID is stack configuration.
- Built to the **AgentCore Runtime contract** (`POST /invocations`, `GET /ping`, port
  8080) so the same image can be deployed to AgentCore later.
- Inside the agent: embed the question → **semantic cache** (cosine ≥ 0.90 + QuestionGuard,
  spike 04) → retrieve top-k chunks from Qdrant → stream a grounded answer from a fresh Strands
  `Agent` per request → cache complete answers. **Retrieve-then-generate** rather than retrieval
  as a Strands tool: one model call, predictable latency, sources = the retrieved chunks; tool-based
  agentic retrieval is the upgrade path for multi-step questions. The embedding service (`app/services/embedding_service.py`) is shared by the
  worker (indexing) and the agent (queries) so vectors stay identical.
- **Streaming is passed through:** client ← SSE ← `api` ← streamed HTTP ← `agent`.
  No broker. Trade-off: no resume after disconnect; one open connection per answer.
  A broker (Valkey/Redis Streams) behind an `EventStream` interface is the upgrade path
  for resumable streams or cross-replica fan-out.
- Known Strands Gemini issues, not yet handled beyond caching only `end_turn` answers:
  `RECITATION`/missing finish reasons surfaced as `end_turn`, and some 429s not retried.
  See `docs/research/strands-vertex-gemini.md`.

### Data and storage

- **Postgres:** `jobs` (status, timestamps, attempts as columns; result as
  `jsonb`) and Temporal's persistence — one database engine (plus `outbox` in the production target).
  In AWS (outbox target) logical replication is enabled for Debezium via an RDS parameter
  group (`rds.logical_replication=1`).
- **Qdrant:** `pdf_documents` collection (cosine) and `answer_cache` collection. The cache stores
  `created_at` in the payload and lookups filter out expired entries (Qdrant has no TTL;
  `purge_expired` exists but isn't scheduled yet). Hybrid (sparse + dense with fusion) is available when needed.
- **Object storage:** local volume behind an `ObjectStorage` interface (MinIO/S3 later).

### Connection management

No per-request connections anywhere. Every client is created once per process in the
app lifespan / worker startup and injected:

**Single owner of worker-side database access.** Temporal workers never connect to
Temporal's persistence — only the Temporal server does, through its own pool
(`maxConns`, `maxIdleConns`, `maxConnLifetime`). For application data, all activities that
touch Postgres (`mark_running`, `complete_job`, `mark_failed`, `NOTIFY`) run on the `io`
task queue, served by a dedicated `io` worker (1–2 replicas) that holds the only
worker-side pool. `cpu` workers (OCR, NER, matching, embedding) hold **no** database
connections, so they scale horizontally without changing the connection count; when
the `io` worker is busy, tasks wait durably in Temporal's queue instead of on Postgres.
A single shared connection was rejected: one connection runs one transaction at a time
(head-of-line blocking) and is a single point of failure — a small pool behind one owner
gives the same predictability with concurrency.

| Resource | Approach |
|----------|----------|
| Postgres pools | Async pool per process (`psycopg_pool.AsyncConnectionPool`) in `api` and the `io` worker only, with `min_size`/`max_size`, acquire `timeout`, `max_waiting`, `max_lifetime`, `max_idle` and a reset/health check on checkout — all from config |
| Postgres budget | `api` replicas × (`api` pool + 1 `LISTEN`) + `io` replicas × `io` pool (+ 1 Debezium replication connection in the outbox target) + Temporal server `maxConns` + admin headroom < `max_connections`; independent of `cpu` worker count |
| Query hygiene | `statement_timeout` and `idle_in_transaction_session_timeout` on the app role; connections acquired late and released early — never held across OCR, NER or LLM calls |
| Backpressure | Pool exhaustion → fast `503` + `Retry-After` from the API (bounded `max_waiting`); async jobs absorb bursts in Temporal's task queues |
| Observability | Not implemented yet: export `pool.get_stats()` (in use, idle, waiting, acquire wait, timeouts) with the service metrics |
| `LISTEN/NOTIFY` | `api` only (job-completion SSE): one dedicated long-lived connection per process, outside the pool (LISTEN is session-scoped) |
| PgBouncer | Added in transaction mode when replica count makes the budget tight; `LISTEN` connections bypass it |
| Qdrant | One client per process (pooled HTTP keep-alive / gRPC channel), reused |
| `api` → `agent` | One shared `httpx.AsyncClient` with connection limits and timeouts |
| Vertex (`google-genai`) | One client per event loop (Strands' documented constraint) |
| Temporal client | One per process, reused |

Pools are closed on shutdown; pool sizes, timeouts and limits come from configuration.

### Interfaces

`OCRService`, `NERService`, `NameLocator`, `NameMatcher`, `EmbeddingService`,
`VectorStore` (`QdrantVectorStore`), `SemanticCache`, `LLMClient`, `JobRepository`,
`ObjectStorage`, `JobOrchestrator`.

## Model choices (decided by spikes)

| Choice | Decision | Evidence |
|--------|----------|----------|
| OCR engine | **PaddleOCR PP-OCRv5 mobile on ONNX Runtime CPU (RapidOCR)** | spike 01 |
| NER model | **GLiNER small v2.1, ONNX int8 on CPU, threshold 0.3 (configurable)** | spike 02 |
| Embedding model | **bge-small-en-v1.5 (384-d), ONNX Runtime CPU via fastembed** | spike 03 |
| Answer-cache rule | **cosine ≥ 0.90 + QuestionGuard** | spike 04 |
| Gemini model (local) | **gemini-3.8-flash** (`GEMINI_MODEL`) | verified on Vertex |
| Bedrock model (AWS) | stack configuration | — |

## Consequences

- More moving parts than a single FastAPI app, but each concern scales and fails
  independently, and the sync endpoint keeps the simple path and the test contract.
- One database engine (Postgres) and one vector engine (Qdrant) — the stack matches the
  brief's suggested technologies, so deviations need no special justification.
- The agent service can move to AgentCore without code changes to its contract.
