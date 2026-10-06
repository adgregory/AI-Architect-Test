# Research — Strands Agents with Gemini on Vertex AI (ADC)

_Desk research, 2026-10-05. Nothing installed or run; verify model IDs and prices in the console._

## Findings

- **Repo:** `strands-agents/sdk-python` now redirects to the `strands-agents/harness-sdk`
  monorepo (Python under `strands-py/`); latest `python/v1.58.0` (2026-10-05).
- **Provider:** `from strands.models.gemini import GeminiModel`, installed with
  `strands-agents[gemini]` (`google-genai>=1.67,<3`). Uses `google-genai`, not
  `google-generativeai`.
- **Vertex + ADC works** without workarounds — the provider passes `client` or
  `client_args` straight to `genai.Client`:
  - `client=genai.Client(vertexai=True, project=..., location=...)`, or
  - `client_args={"vertexai": True, "project": ..., "location": ...}`, or
  - env vars `GOOGLE_GENAI_USE_VERTEXAI=true`, `GOOGLE_CLOUD_PROJECT`, `GOOGLE_CLOUD_LOCATION`.
  - In `google-genai` 2.x, `vertexai` is a legacy alias for `enterprise=True`; it still works.
  - ADC via `google.auth.default(cloud-platform scope)`; `quota_project_id` sent as
    `x-goog-user-project`; location defaults to `global` without an API key.
- **Streaming:** `agent.stream_async(prompt)` yields dicts; text chunks in `event["data"]`,
  final `event["result"]` (with `stop_reason`). Use `callback_handler=None` with
  `stream_async`. The Gemini provider streams natively via
  `client.aio.models.generate_content_stream`.
- **Known open issues** (harness-sdk tracker):
  - #4723 `RECITATION` finish mapped to `end_turn` — relevant for RAG answers.
  - #4722 stream with no finish reason reported as normal `end_turn`.
  - #4523 429 with "Too Many Requests" status not retried (only `RESOURCE_EXHAUSTED`/`UNAVAILABLE`).
  - #3639 built-in + function tools → 400; #1121 structured output (native JSON schema not streamed).
  - A `genai.Client` should not be shared across asyncio event loops.
- **Models (unverified status/suffixes):** `gemini-3.5-flash-lite` (cheap),
  `gemini-3.8-flash` (balanced), `gemini-3.1-pro` (strong); `gemini-2.5-flash` stable
  fallback. `location="global"` for widest availability.
- **ADC in Docker:** `gcloud auth application-default login` (ideally
  `--impersonate-service-account=<SA with roles/aiplatform.user>`), set the quota project,
  mount the ADC file read-only and set `GOOGLE_APPLICATION_CREDENTIALS`. Never bake it
  into an image; avoid service-account key files.

## Decision impact

Strands chosen for AWS/AgentCore portability, observability (OpenTelemetry) and memory;
the issues above are handled explicitly in the agent service (see ADR 0001).

## Minimal shape

```python
from google import genai
from strands import Agent
from strands.models.gemini import GeminiModel

client = genai.Client(vertexai=True, project=PROJECT, location="global")  # ADC
model = GeminiModel(client=client, model_id=MODEL_ID, params={"temperature": 0.2})

async def answer(question: str, context: str):
    agent = Agent(model=model, system_prompt=f"Answer only from:\n{context}", callback_handler=None)
    async for ev in agent.stream_async(question):
        if "data" in ev:
            yield ev["data"]
        elif "result" in ev:
            stop = ev["result"].stop_reason  # check for recitation / truncation
```

## Sources

- https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/models/gemini.py
- https://github.com/strands-agents/harness-sdk/blob/main/site/src/content/docs/user-guide/sdk/model-providers/google.mdx
- https://github.com/strands-agents/harness-sdk/blob/main/site/src/content/docs/user-guide/sdk/streaming/async-iterators.mdx
- https://github.com/googleapis/python-genai/blob/main/google/genai/client.py
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/models
- https://docs.cloud.google.com/vertex-ai/generative-ai/docs/learn/locations
- https://pypi.org/project/strands-agents/
