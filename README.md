# FinGraph-RAG

FinGraph-RAG is an observable, guardrailed GraphRAG system for messy,
interconnected financial data. The current slice includes typed source-backed
contracts, bounded hybrid retrieval, deterministic calculation and citation
verification, a replayable FastAPI/SSE workflow, synthetic fixtures, and a
local PostgreSQL/Redis/Neo4j/Qdrant runtime. External model/provider and public
corpus activation remain explicit deployment-owner concerns.

## Local setup

```bash
cp .env.example .env
uv sync --extra orchestration
uv run ruff format .
uv run ruff check .
uv run pytest -m unit --no-cov
docker compose -f docker/compose.yaml config
docker compose -f docker/compose.yaml up -d postgres redis neo4j qdrant
uv run python scripts/check_health.py --required postgres redis neo4j qdrant
```

For a locked release check, use `uv sync --locked --extra orchestration --dev`
followed by `uv run pytest --no-cov`, then run the Docker build and Compose
gates in [the deployment runbook](docs/runbooks/deployment.md). The Compose
file binds local ports to loopback and pins dependency images by digest.

## Runtime modes

The API uses `FINGRAPH_RUNTIME_MODE=live` by default and fails closed when live
model, embedding, graph, or alias configuration is incomplete. Set
`FINGRAPH_RUNTIME_MODE=synthetic` plus `FINGRAPH_FIXTURE` only for repository
fixtures; synthetic mode is forbidden in `prod`. The local CLI exercises the
same workflow and verifier without provider credentials:

```bash
uv run python scripts/query.py \
  --fixture tests/fixtures/graph/sample.json \
  --question 'How does Parent affect Supplier FY2025 as filed?' \
  --require-citations
```

The CLI emits ordered workflow events and returns non-zero when the request
abstains, errors, or violates the requested citation/abstention expectation.
It is a synthetic verification boundary, not evidence of a live financial-data
deployment.

## Architecture boundary

`domain` contains typed contracts, policies, and errors. `application` owns
use-case boundaries and unit-of-work protocols. `graph` and `vector` expose
storage ports, while adapters provide Neo4j, Qdrant, embeddings, and fixture
implementations. The API composition root selects live or synthetic mode;
concrete integrations depend inward on the contracts. See
[data lineage](docs/data-lineage.md), [threat model](docs/threat-model.md),
and [deployment/release operations](docs/runbooks/deployment.md) for the
boundaries and evidence requirements.
