# Deployment runbook

This runbook describes the repository's Docker Compose reference deployment.
It is a local or operator-controlled deployment boundary, not evidence of a
production compliance approval. Image digests and dependency versions are
reviewed inputs; credentials, backups, network policy, and legal/compliance
sign-off remain deployment-owner responsibilities.

## Before deploying

1. Use a clean checkout or review the exact diff being released.
2. Copy `.env.example` to `.env` only for local setup. For any shared
   environment, provide secrets through the deployment secret manager or an
   environment file that is not committed.
3. Replace every development/default credential, especially
   `POSTGRES_PASSWORD`, `NEO4J_AUTH`, and `FINGRAPH_NEO4J_PASSWORD`. Set an
   application token (`FINGRAPH_API_TOKEN`) when the API is reachable by more
   than a single trusted local user.
4. Confirm that the host has Docker Engine and Compose, and that the published
   ports do not conflict. The reference Compose file binds published ports to
   `127.0.0.1`; put an authenticated, TLS-terminating proxy in front of the API
   before exposing it outside the host.

## Validate and start the dependency stack

Run these commands from the repository root:

```bash
docker compose -f docker/compose.yaml config --quiet
docker compose -f docker/compose.yaml pull postgres redis neo4j qdrant
docker compose -f docker/compose.yaml up -d postgres redis neo4j qdrant
uv run python scripts/check_health.py --required postgres redis neo4j qdrant
```

The Compose health checks gate container startup where a dependency is used by
another service. Qdrant's image does not include a general-purpose HTTP client,
so its container check covers storage initialization and
`scripts/check_health.py` performs the host-facing `/readyz` check.

## Start application profiles

Build and start the API after the dependency checks pass:

```bash
docker compose -f docker/compose.yaml build api
docker compose -f docker/compose.yaml --profile api up -d api
docker compose -f docker/compose.yaml ps
curl --fail --silent http://127.0.0.1:8000/health/live
curl --fail --silent http://127.0.0.1:8000/health/ready
```

Optional, repository-owned verification profiles are deliberately opt-in:

```bash
docker compose -f docker/compose.yaml --profile synthetic up -d synthetic-api
docker compose -f docker/compose.yaml --profile ingestion run --rm fixture-worker
docker compose -f docker/compose.yaml --profile evaluation run --rm evaluation
docker compose -f docker/compose.yaml --profile observability up -d phoenix
```

The worker and evaluation services are one-shot jobs. They are not a durable
queue or a production scheduler.

Local checkpoint/state data under `.fingraph-state/` and generated reports under
`reports/` are ignored by Git because they may contain request or evidence
data. Ignoring them is not deletion or retention enforcement: operators must
set and verify storage retention, access controls, backup encryption, and
deletion schedules for any shared deployment.

## Release gates

Run the gates in this order and retain their output with the release record:

```bash
uv run ruff format --check .
uv run ruff check .
timeout 60 uv run pytest -m unit --no-cov
docker compose -f docker/compose.yaml config --quiet
uv run pytest -m e2e --no-cov
```

The e2e test starts a local synthetic-mode API and does not prove that external
provider credentials, a public financial corpus, or a production deployment is
working. A successful Compose config check likewise does not prove that images
were pulled, credentials are valid, storage is backed up, or compliance review
is complete.

## Isolated PostgreSQL backup/restore drill

Run this drill with two temporary named volumes and the pinned PostgreSQL image;
it does not stop or reuse the application stack's volumes. Create one marker
row in the source database, dump it, start a fresh database on the restore
volume, restore the dump, and verify the marker:

```bash
DRILL=fingraph-backup-drill-$RANDOM
docker volume create "$DRILL-source"
docker volume create "$DRILL-restore"
docker run -d --name "$DRILL-source-db" \
  -e POSTGRES_DB=drill -e POSTGRES_USER=drill -e POSTGRES_PASSWORD=drill-pass \
  -v "$DRILL-source":/var/lib/postgresql/data \
  postgres:16.4-alpine@sha256:5660c2cbfea50c7a9127d17dc4e48543eedd3d7a41a595a2dfa572471e37e64c
docker exec "$DRILL-source-db" pg_isready -U drill -d drill
docker exec "$DRILL-source-db" psql -v ON_ERROR_STOP=1 -U drill -d drill \
  -c "CREATE TABLE release_drill (id integer PRIMARY KEY, marker text NOT NULL); INSERT INTO release_drill VALUES (1, 'backup-restore-ok');"
docker exec "$DRILL-source-db" pg_dump -U drill -d drill --format=plain --no-owner --no-privileges > backup.sql
docker stop "$DRILL-source-db" && docker rm "$DRILL-source-db"
docker run -d --name "$DRILL-restore-db" \
  -e POSTGRES_DB=drill -e POSTGRES_USER=drill -e POSTGRES_PASSWORD=drill-pass \
  -v "$DRILL-restore":/var/lib/postgresql/data \
  postgres:16.4-alpine@sha256:5660c2cbfea50c7a9127d17dc4e48543eedd3d7a41a595a2dfa572471e37e64c
docker exec "$DRILL-restore-db" pg_isready -U drill -d drill
docker exec -i "$DRILL-restore-db" psql -v ON_ERROR_STOP=1 -U drill -d drill < backup.sql
docker exec "$DRILL-restore-db" psql -At -U drill -d drill \
  -c "SELECT id || '|' || marker FROM release_drill;"
```

Expected result: `1|backup-restore-ok`. Remove only the two drill containers,
the two drill volumes, and the temporary `backup.sql` after recording the
result (for example, `docker stop "$DRILL-restore-db"; docker rm
"$DRILL-restore-db"; docker volume rm "$DRILL-source" "$DRILL-restore"; rm
backup.sql`). This validates dump/restore mechanics for this image; it does not prove
that an operator's real backup schedule, encryption, retention, permissions,
or recovery-time objective is adequate.

Release evidence from 2026-10-26: the drill passed with two isolated volumes
(`fingraph-backup-drill-31948-source` and `fingraph-backup-drill-31948-restore`),
marker `1|backup-restore-ok`, and a 1,068-byte dump. The existing Compose
database services were not restarted or recreated.

## Synthetic API smoke boundary

When the API image and dependencies are stable, start only the synthetic
profile and verify `/health/live`, `/health/ready`, an unauthenticated request,
and an authenticated SSE query:

```bash
FINGRAPH_API_TOKEN=replace-for-smoke \
  docker compose -f docker/compose.yaml --profile synthetic up -d synthetic-api
curl --fail --silent http://127.0.0.1:8001/health/live
curl --fail --silent http://127.0.0.1:8001/health/ready
curl --silent --output /dev/null --write-out '%{http_code}\n' \
  -X POST http://127.0.0.1:8001/v1/query \
  -H 'Content-Type: application/json' \
  --data '{"question":"How does Parent affect Supplier FY2025 as filed?"}'
curl --fail --silent --max-time 15 \
  -X POST http://127.0.0.1:8001/v1/query \
  -H 'Authorization: Bearer replace-for-smoke' \
  -H 'Content-Type: application/json' \
  --data '{"question":"How does Parent affect Supplier FY2025 as filed?"}'
docker compose -f docker/compose.yaml --profile synthetic stop synthetic-api
```

The expected auth-negative status is `401`; the successful SSE response should
contain `accepted`, `verification`, `answer`, and `done` events. On 2026-10-26,
the rebuilt API image passed with live `200`, readiness `200`, unauthenticated
query `401`, and authenticated SSE `200`; the stream contained all four events
and the synthetic answer marker. The earlier readiness failure was caused by an
older image with an incompatible `aiosqlite`; always rebuild from the current
lock before smoke testing. No Neo4j/Qdrant-to-workflow E2E claim is made here.

## Phoenix telemetry smoke

Set `FINGRAPH_OTLP_ENDPOINT` to the Phoenix OTLP/HTTP trace endpoint and, when
needed for local forensic review, set `FINGRAPH_TRACE_PATH` to a writable path.
The Compose API and synthetic services pass both values through explicitly.
Run the application smoke against a real local Phoenix container, then verify
both the redacted JSONL artifact and Phoenix's project API:

```bash
FINGRAPH_OTLP_ENDPOINT=http://phoenix:6006/v1/traces \
  docker compose -f docker/compose.yaml --profile observability up -d phoenix
# Start the synthetic profile with the same endpoint on its Compose network,
# run the readiness/auth/SSE checks above, and retain the trace artifact.
curl --fail --silent 'http://127.0.0.1:6006/v1/projects/default/traces?limit=100&include_spans=true'
```

Release evidence from 2026-10-26 used the pinned Phoenix image in an isolated
temporary container on host port `16006` (the existing host port `6006` was in
use). A real synthetic-mode API process exported to
`http://127.0.0.1:16006/v1/traces`; readiness and authenticated SSE passed, the
safe JSONL artifact contained application spans such as `query.classify`, and
Phoenix's `/v1/projects/default/traces?include_spans=true` response contained
the application spans. The artifact was checked for query text and bearer
token absence. The temporary Phoenix container and volume were removed after
the evidence was recorded. This proves exporter reachability and redaction for
the local setup; it does not prove retention, alerting, access policy, or
production observability SLOs.

## Rollback and shutdown

Keep the previously approved checkout and image digests available. To roll back
an application release, stop the profile, check out the approved revision, then
rebuild and start it again:

```bash
docker compose -f docker/compose.yaml --profile api stop api
docker compose -f docker/compose.yaml build api
docker compose -f docker/compose.yaml --profile api up -d api
docker compose -f docker/compose.yaml ps
```

Do not remove named volumes as part of an application rollback. They contain
state for PostgreSQL, Neo4j, Qdrant, checkpoints, and reports. For a complete
local teardown (data loss), explicitly confirm the volume list first and then
run:

```bash
docker compose -f docker/compose.yaml down
docker volume ls --filter label=com.docker.compose.project=fingraph-rag
```

Volume deletion, backup restoration, schema migration rollback, and incident
response require an operator-specific procedure and are intentionally not
automated here.

## Evidence to record

For each release, record the Git revision, image digests, Compose config output,
release-gate results, health responses, and the exact environment/secret source
(never the secret values). Also record any skipped gate and its owner-approved
reason. This runbook cannot independently establish credential safety,
availability targets, backup recoverability, vulnerability status, or
legal/compliance approval.
