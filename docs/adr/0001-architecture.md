# ADR 0001 — Solution architecture

- **Status:** accepted (model choices pending spikes 01–04)
- **Date:** 2026-10-05

## Context

The service extracts person names and their bounding boxes from scanned PDFs,
fuzzy-matches them against names supplied in the request, indexes document
text for retrieval, and answers questions with RAG. Constraints:

- Runs locally with Docker Compose for now; infrastructure-as-code comes later.
- The only cloud account available is GCP, so the LLM is Gemini on Vertex AI
  via Application Default Credentials (ADC).
- The test suite fixes part of the contract: a synchronous `POST /api/extract`
  response shape, and a Qdrant-backed `VectorStore`.
- A later AWS deployment (AgentCore for the agent runtime) must stay possible.
- Every backend sits behind an interface so it can be swapped (e.g. hybrid search).

## Decision

### Services (docker compose)

| Container | Responsibility |
|-----------|----------------|
| `api` | FastAPI backend: validation, sync extraction, job API, SSE, relays agent stream |
| `outbox-relay` | Dispatches outbox rows to Temporal |
| `worker-cpu` | Temporal worker on the `cpu` queue: OCR, NER, box location, fuzzy matching, embedding — no DB connections |
| `worker-io` | Temporal worker on the `io` queue: job status/result writes, `NOTIFY`, Qdrant upserts — sole owner of the worker-side DB pool |
| `agent` | Strands agent (Gemini on Vertex via ADC), AgentCore runtime contract |
| `postgres` | `jobs`, `outbox`, Temporal persistence |
| `temporal`, `temporal-ui` | Workflow orchestration |
| `qdrant` | Document-chunk vectors and the semantic answer cache |

### API

- `POST /api/extract` — synchronous extraction (kept: test contract, small files).
- `POST /api/jobs` → `202 {job_id}`; `GET /api/jobs/{id}`; `GET /api/jobs/{id}/events` (SSE).
- `POST /api/ask` — streamed answer (SSE) with sources.
- `GET /health`.

Sync and async extraction call the same service classes; only the entry point differs.

### Async extraction: transactional outbox + Temporal

1. `POST /api/jobs` validates the upload, stores the PDF, and in **one transaction**
   inserts the `jobs` row and an `outbox` row, then returns `202`.
2. `outbox-relay` claims unsent rows with `SELECT … FOR UPDATE SKIP LOCKED`, starts
   `ExtractNamesWorkflow` with `workflow_id = job_id`, and marks the row dispatched.
   It is woken by `LISTEN/NOTIFY` with a short poll as fallback. Delivery is
   at-least-once; Temporal's workflow-ID uniqueness makes duplicates harmless.
   No CDC (Debezium/Kafka) — a polling relay is enough at this scale; CDC is the
   upgrade path if other consumers need the same events.
3. The workflow's last activity writes the result to `jobs.result` (jsonb), sets
   the status, and issues `NOTIFY job_done`; open SSE connections push the result.

### Temporal workflow

```
ExtractNamesWorkflow(job_id)
  1. mark_running                      (io)
  2. prepare_document → page refs      (cpu)  load PDF, validate, render pages, store
  3. ocr_page × N, in parallel         (cpu)  text + word boxes per page → stored ref
  4. extract_names                     (cpu)  NER → person spans
  5. locate_and_match                  (cpu)  name → word boxes (PDF space) + fuzzy match (≥ 90%)
  6. complete_job                      (io)   jobs.result, status, NOTIFY
  child IndexDocumentWorkflow(job_id)  (abandon on parent close)  chunk + embed (cpu) → upsert Qdrant (io)
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

- Strands Agents with `GeminiModel` (`strands-agents[gemini]`) given a pre-built
  `google.genai.Client(vertexai=True, project, location="global")`; credentials via ADC.
- Built to the **AgentCore Runtime contract** (`POST /invocations`, `GET /ping`, port
  8080) so the same image can be deployed to AgentCore later.
- Retrieval (embed question → Qdrant search → semantic cache) is a Strands tool inside
  the agent. The embedding model lives in a shared internal package used by both the
  worker (indexing) and the agent (queries) so vectors stay identical.
- **Streaming is passed through:** client ← SSE ← `api` ← streamed HTTP ← `agent`.
  No broker. Trade-off: no resume after disconnect; one open connection per answer.
  A broker (Valkey/Redis Streams) behind an `EventStream` interface is the upgrade path
  for resumable streams or cross-replica fan-out.
- Known Strands Gemini issues to handle explicitly: `RECITATION`/missing finish reasons
  surfaced as `end_turn` (inspect `stop_reason`), and some 429s not retried (own retry).
  See `docs/research/strands-vertex-gemini.md`.

### Data and storage

- **Postgres:** `jobs` (status, timestamps, owner, retry count as columns; result as
  `jsonb`), `outbox`, and Temporal's persistence — one database engine.
- **Qdrant:** `chunks` collection (cosine) and `answer_cache` collection. The cache stores
  `created_at` in the payload; lookups filter expired entries and a periodic job deletes
  them (Qdrant has no TTL). Hybrid (sparse + dense with fusion) is available when needed.
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
| Postgres pools | Async pool per process (`psycopg_pool.AsyncConnectionPool` or SQLAlchemy async engine) in `api`, `outbox-relay` and the `io` worker only, with `min_size`/`max_size`, acquire `timeout`, `max_waiting`, `max_lifetime`, `max_idle` and a reset/health check on checkout — all from config |
| Postgres budget | `api` replicas × `api` pool + `io` replicas × `io` pool + relay pool + `LISTEN` connections + Temporal server `maxConns` + admin headroom < `max_connections`; independent of `cpu` worker count |
| Query hygiene | `statement_timeout` and `idle_in_transaction_session_timeout` on the app role; connections acquired late and released early — never held across OCR, NER or LLM calls |
| Backpressure | Pool exhaustion → fast `503` + `Retry-After` from the API (bounded `max_waiting`); async jobs absorb bursts in the outbox and Temporal queues |
| Observability | Pool stats (in use, idle, waiting, acquire wait, timeouts) exported with the service metrics |
| `LISTEN/NOTIFY` | One dedicated long-lived connection per process, outside the pool (LISTEN is session-scoped) |
| PgBouncer | Added in transaction mode when replica count makes the budget tight; `LISTEN` connections bypass it |
| Qdrant | One client per process (pooled HTTP keep-alive / gRPC channel), reused |
| `api` → `agent` | One shared `httpx.AsyncClient` with connection limits and timeouts |
| Vertex (`google-genai`) | One client per event loop (Strands' documented constraint) |
| Temporal client | One per process, reused |

Pools are closed on shutdown; pool sizes, timeouts and limits come from configuration.

### Interfaces

`OCREngine`, `NERModel`, `EmbeddingModel`, `VectorStore` (`QdrantVectorStore`),
`SemanticCache`, `JobRepository`, `ObjectStorage`, `AgentClient`.

## Pending (decided by spikes)

| Choice | Spike |
|--------|-------|
| OCR engine | 01 |
| NER model | 02 |
| Embedding model | 03 |
| Answer-cache similarity threshold | 04 |
| Gemini model ID | confirm in the Vertex console |

## Consequences

- More moving parts than a single FastAPI app, but each concern scales and fails
  independently, and the sync endpoint keeps the simple path and the test contract.
- One database engine (Postgres) and one vector engine (Qdrant) — the stack matches the
  brief's suggested technologies, so deviations need no special justification.
- The agent service can move to AgentCore without code changes to its contract.
