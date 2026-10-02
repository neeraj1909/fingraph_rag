# FinGraph-RAG

FinGraph-RAG is an observable, guardrailed GraphRAG system for messy,
interconnected financial data. This repository currently contains Milestone 1:
typed domain contracts, vendor-neutral ports, configuration, and a local
PostgreSQL/Redis/Neo4j/Qdrant runtime.

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

The orchestration extra is optional in this foundation slice. Domain and
application code must remain free of vendor SDK imports; adapters are added in
later milestones.

## Architecture boundary

`domain` contains typed contracts, policies, and errors. `application` owns
use-case boundaries and unit-of-work protocols. `infrastructure` provides
vendor-neutral gateways and clocks. `graph` and `vector` expose storage ports.
Concrete integrations must depend inward on these contracts.
