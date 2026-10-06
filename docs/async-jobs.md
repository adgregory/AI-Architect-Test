# Async jobs — sequence diagrams

How large and batch PDFs are processed: `POST /api/jobs` returns `202` immediately, and a
Temporal workflow runs the same extraction as `/api/extract`, split across worker queues.
This is the path the scaling plan in [DESIGN.md](../DESIGN.md#4-scaling-to-1000-pdfshour)
relies on. Everything below is implemented and runs in `docker compose`.

The production write path (transactional outbox → Debezium → Kinesis → Lambda) is designed
in [ADR 0001](adr/0001-architecture.md) and modelled in `infra/`, but not built, so it isn't
diagrammed here.

**Participants**
- **api**: FastAPI. Validates the upload, records the job, starts the workflow, serves SSE.
- **Storage**: the PDF and intermediate artifacts (a local volume; S3 in AWS). Activities
  pass storage keys, never document contents.
- **Postgres**: the `jobs` table. Every status change sends `NOTIFY job_events` in the same
  transaction.
- **Temporal**: runs the workflows and dispatches activities to two task queues.
- **cpu worker** (`extraction-cpu`, scales horizontally): OCR, NER, matching, embedding.
  Holds no database connections.
- **io worker** (`extraction-io`): the only worker with a Postgres pool; also writes to Qdrant.

## 1. Submit a job and follow it

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant A as api
  participant S as Storage
  participant P as Postgres
  participant T as Temporal
  C->>A: POST /api/jobs (PDF + names)
  A->>A: validate (PDF signature, size, names JSON) → 400 / 413 / 422
  A->>S: put jobs/<job_id>/input.pdf
  A->>P: INSERT job (status = queued)
  A->>T: start ExtractNamesWorkflow (workflow_id = job_id)
  Note over A,T: If Temporal is unreachable, the job stays queued and the<br/>reconciler starts it later (section 3). The client still gets its id.
  A-->>C: 202 {job_id, links.self, links.events}

  C->>A: GET /api/jobs/<job_id>/events (SSE)
  A->>P: read job
  A-->>C: event: current status (queued, or already running)
  loop until the job is terminal (or 10 min pass, then the client reconnects)
    P--)A: NOTIFY job_events (one LISTEN connection per api process)
    A->>P: re-read job
    A-->>C: event: running / succeeded / failed (full job view)
    Note over A,C: No notification within 5 s → re-read anyway and send a<br/>keep-alive comment, so a missed NOTIFY only adds latency.
  end
```

`GET /api/jobs/<job_id>` returns the same job view as a one-off JSON read.

## 2. The workflow: extraction across the two queues

```mermaid
sequenceDiagram
  autonumber
  participant T as Temporal<br/>ExtractNamesWorkflow
  participant CPU as cpu worker ×N
  participant IO as io worker
  participant S as Storage
  participant P as Postgres
  participant I as Temporal<br/>IndexDocumentWorkflow
  participant Q as Qdrant

  T->>CPU: prepare_document(job_id)
  CPU->>S: read input.pdf
  CPU-->>T: page_count, document_id = uuid5(sha256(pdf))
  T->>IO: mark_running(job_id, page_count)
  IO->>P: status = running + NOTIFY
  par one activity per page, in parallel
    T->>CPU: ocr_page(job_id, 0)
    CPU->>S: write pages/0000.json (text + word boxes, PDF points)
  and
    T->>CPU: ocr_page(job_id, n)
    CPU->>S: write pages/<n>.json
  end
  Note over T,CPU: Each page activity heartbeats (30 s timeout): a crashed worker's page<br/>is retried on another worker, and finished pages are not redone.
  T->>CPU: extract_and_match(job_id, pages, query_names)
  CPU->>S: read pages → NER → name boxes → fuzzy match (≥ 90%)
  CPU->>S: write result.json
  CPU-->>T: names, matches (counts only)

  alt every activity succeeded
    T->>IO: complete_job(job_id)
    IO->>S: read result.json
    IO->>P: status = succeeded, result (jsonb) + NOTIFY
    T-)I: start child (id = index-<document_id>, abandoned)
    Note over T,I: Same PDF already being indexed → WorkflowAlreadyStarted, ignored.<br/>The job is already succeeded: indexing can't delay or fail it.
    I->>CPU: chunk_and_embed (bge-small)
    CPU->>S: write chunks.json (texts + vectors)
    I->>IO: upsert_chunks
    IO->>Q: upsert into pdf_documents (point ids per document + chunk)
  else an activity fails for good (InvalidDocument, or retries exhausted)
    T->>IO: fail_job(job_id, root-cause message)
    IO->>P: status = failed, error + NOTIFY
    Note over T: No indexing for a failed job
  end
```

**Timeouts and retries** (`app/workflows/workflows.py`)

| Activity | Queue | Start-to-close | Retries |
|---|---|---|---|
| `prepare_document` | cpu | 60 s | up to 3, backoff 1 s → 30 s; invalid input not retried |
| `ocr_page` × N | cpu | 3 min, heartbeat 30 s | up to 3 |
| `extract_and_match` | cpu | 2 min | up to 3 |
| `chunk_and_embed` | cpu | 5 min | up to 3 |
| `mark_running`, `complete_job`, `fail_job` | io | 15 s | up to 10, backoff 1 s → 20 s |
| `upsert_chunks` | io | 60 s | up to 10 |

`prepare_document` fails fast on an unreadable PDF, a PDF with no pages, or more pages than
`MAX_PAGES`: those raise `InvalidDocument`, which is never retried. Indexing runs as an
abandoned child workflow, so the client sees `succeeded` as soon as `complete_job` commits.

## 3. Reconciler: jobs whose workflow never started

```mermaid
sequenceDiagram
  autonumber
  participant Sch as Temporal schedule<br/>(every 60 s, overlap: skip)
  participant R as ReconcileQueuedJobsWorkflow
  participant IO as io worker
  participant P as Postgres
  Sch->>R: start
  R->>IO: find_stale_jobs(older than 120 s)
  IO->>P: SELECT jobs WHERE status = queued AND created_at < now() - 120 s
  IO-->>R: [job_id, filename, query_names] …
  loop each stale job
    R->>R: start child ExtractNamesWorkflow (workflow_id = job_id)
    Note over R: Already running → WorkflowAlreadyStarted, skipped.<br/>The fixed workflow id makes restarts idempotent.
  end
```

The io worker creates this schedule at startup. Its interval and the staleness threshold are
the `RECONCILE_INTERVAL_S` and `RECONCILE_STALE_AFTER_S` settings.

## 4. Background indexing from the sync endpoint

`POST /api/extract` runs extraction in the api process (see DESIGN.md), then reuses the same
indexing workflow, so a document extracted there can be queried with `/api/ask`:

```mermaid
sequenceDiagram
  autonumber
  participant C as Client
  participant A as api
  participant S as Storage
  participant T as Temporal
  C->>A: POST /api/extract
  A->>A: OCR once → NER → boxes → fuzzy match (in process)
  A-->>C: 200 {extracted_names, fuzzy_matches}
  A->>S: write jobs/<document_id>/pages/0000.json (the OCR result, reused)
  A->>T: start IndexDocumentWorkflow (id = index-<document_id>)
  Note over A,T: After the response, best effort: failures are logged,<br/>never returned. Off with INDEX_ON_EXTRACT=false.
  Note over T: Then chunk_and_embed (cpu) → upsert_chunks (io), as in section 2
```
