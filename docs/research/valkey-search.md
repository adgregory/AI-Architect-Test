# Research — Valkey as vector store, semantic cache and event stream

_Desk research, 2026-10-05. Nothing installed or run._

**Outcome:** not adopted for now — Qdrant (vectors + answer cache) and Postgres (job
events) cover the same needs with fewer services and match the brief and test suite
(ADR 0001). Valkey/Redis Streams remains the upgrade path for resumable token streams
or cross-replica fan-out.

## Findings

- **valkey-search 1.2.1** (2026-07-07; needs Valkey 9.0.1+), BSD-3-Clause, GA since 2025-05.
  - `HNSW` and `FLAT`; `L2` / `IP` / `COSINE`; `FLOAT32` only in 1.2.x.
  - HASH or JSON documents; `TAG` and `NUMERIC` pre-filters; KNN via
    `FT.SEARCH idx "*=>[KNN k @vec $v AS score]" PARAMS 2 v <blob> DIALECT 2`.
  - `TEXT` fields since 1.2.0 but no released relevance scoring. BM25 (#1287) and
    `FT.HYBRID` with RRF (#1083) merged on `main` only, with open scoring bugs (#1410–1416).
  - Largely RediSearch-compatible, with documented `FT.SEARCH` differences (#1353).
  - COSINE score is a **distance** (`1 − cos`, lower is better).
- **Docker:** `valkey/valkey-bundle:9.1.3` (Valkey 9.1.2 + Search 1.2.1 + JSON + Bloom);
  vector indexes persist in RDB; protected mode off — set a password.
- **Python:** `valkey` (valkey-py) — sync + asyncio, redis-py-style `ft()` API, used by
  valkey-search's own tests. `redis` 8.x likely works; `valkey-glide` has an `ft` module.
- **Semantic cache:** RedisVL `SemanticCache` likely fails on 1.2.1 (uses `VECTOR_RANGE`);
  LiteLLM has a native `valkey-semantic` backend; a custom ~50-line KNN cache is simplest.
- **Streaming:** Pub/Sub is fire-and-forget (lost on reconnect). Streams (`XADD`,
  `XREAD BLOCK` from `Last-Event-ID`) support resumable SSE.
- **vs Qdrant:** all in RAM (~3.5–7 KB per 768-d vector, unverified); no released hybrid
  search; snapshots via RDB; active but young (1.2.1 was largely crash fixes, ~250 open
  issues).

## Sources

- https://github.com/valkey-io/valkey-search (releases, issues #1353, #1410–1416, #989)
- https://valkey.io/commands/ft.create/ · https://valkey.io/commands/ft.search/ · https://valkey.io/topics/search/
- https://hub.docker.com/r/valkey/valkey-bundle
- https://glide.valkey.io/how-to/modules-api/search-module/
- https://docs.litellm.ai/blog/valkey_semantic_caching
