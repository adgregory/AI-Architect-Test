# Infrastructure (AWS, Pulumi + Python + uv)

Production-shaped AWS deployment for the stack described in
[`docs/adr/0001-architecture.md`](../docs/adr/0001-architecture.md). Its own uv project
(`pyproject.toml`, `uv.lock`); Pulumi runs it with `toolchain: uv`.

```
infra/
├── Pulumi.yaml / Pulumi.dev.yaml / Pulumi.staging.yaml   per-environment parameters
├── config.py        typed (pydantic) stack config — rejects invalid sizes / open admin CIDRs
├── stack.py         assembles the components (shared by __main__.py and the tests)
├── components/      network · data · fargate_service · platform · app_services · outbox · observability
├── policies/        CrossGuard pack (rules.py = plain functions, also run by the tests)
└── tests/           pytest with Pulumi mocks — no AWS access
```

| | dev | staging |
|---|---|---|
| Networking | public subnets, no NAT | private subnets + NAT |
| RDS Postgres 16 | single-AZ, no deletion protection | Multi-AZ, deletion protection |
| Worker pools (CPU workers) | 1 × 2 vCPU / 8 GB | 2 × 4 vCPU / 16 GB |
| Job dispatch | direct start + reconciler (as local) | **outbox**: Debezium → Kinesis → Lambda |
| Resources | 166 | 193 |

## What it creates

- **Network:** VPC, subnets per AZ, optional NAT, S3 gateway endpoint; security groups that encode the
  ADR's access rules — only the API, IO workers, Temporal and Debezium reach Postgres (CPU workers never),
  only the API is behind the public listener, the Temporal UI only from `adminCidrs`.
- **Data:** ECR (scan on push), private TLS-only artefacts bucket with lifecycle, RDS Postgres (RDS-managed
  master secret, logical replication with `max_slot_wal_keep_size`), EFS for Qdrant data and Debezium offsets.
- **Platform:** ECS Fargate (ARM64) + Cloud Map service discovery; ALB; Temporal server (one replica,
  persistence in RDS) + UI + schema task; Qdrant.
- **App:** API, CPU and IO worker pools (machine size and counts per stack), migration task.
- **Agent:** Strands agent on **AgentCore Runtime** (VPC mode) with a **Bedrock** model; its role may invoke
  only the configured model. No GCP credentials exist in AWS.
- **Outbox (staging):** Kinesis, Debezium Server, dispatcher Lambda (partial batch failures,
  bisect-on-error, bounded retries, SQS DLQ).
- **Alarms:** RDS storage / connections / CPU, ALB 5xx, replication-slot lag, dispatcher errors, DLQ depth,
  Kinesis iterator age.

## Validate offline (no AWS account)

```bash
task infra:setup      # uv env + local-backend stacks (state in infra/.pulumi, git-ignored)
task infra:check      # unit tests + offline previews of dev and staging with the policy pack
```

Offline mode (`INFRA_OFFLINE=1`) uses an explicit AWS provider with a placeholder credential and skips
credential validation, account-ID lookup and the metadata API. The program uses **no data sources**
(AZs and similar come from config), so a full plan needs no AWS access. Three layers:

1. **`pulumi preview`** — the full resource graph type-checks against the AWS provider schema
   (this caught ALB/target-group names over the 32-character limit).
2. **CrossGuard policies** during preview — S3 public access, RDS encryption/privacy, internet ingress
   only on 80/443, ECR scanning, IAM least privilege, stream consumers with partial failures + DLQ.
   IAM documents that embed not-yet-created ARNs are *unknown* during a preview, so Pulumi reports
   that rule as "can't run" there; the **unit tests** evaluate the same rule functions on fully known
   mock values for every resource of both stacks.
3. **Unit tests with Pulumi mocks** — access rules, data protection, agent permissions, outbox
   resilience, NAT per stack, sizes from config, config validation, and each rule catching a violation.

## Deploying for real

`unset INFRA_OFFLINE`, use real AWS credentials and a real backend, then `pulumi up --stack dev`.
Build and push the `app`, `agent` and `dispatcher` images to the ECR repositories first, then run the
`temporal-schema` and `migrate` task definitions once (`aws ecs run-task`).

## App changes needed to run in AWS (not yet implemented)

The infrastructure passes these settings, but the app doesn't support them yet, so it won't start in AWS until they exist (`ANSWER_BACKEND=agentcore` fails settings validation; the DB settings are ignored):

| Setting | Needed in the app |
|---|---|
| `STORAGE_BACKEND=s3`, `STORAGE_BUCKET` | an `S3Storage` implementation of `ObjectStorage` |
| `DB_HOST`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | build `DATABASE_URL` from parts (RDS-managed secret) |
| `ANSWER_BACKEND=agentcore`, `AGENT_RUNTIME_ARN` | an answering client that calls AgentCore's `InvokeAgentRuntime` |
| `outboxEnabled: true` | an outbox write mode in `POST /api/jobs` and the dispatcher Lambda image |
