.PHONY: format lint test check compose-config up down health

format:
	uv run ruff format .

lint:
	uv run ruff check .

test:
	uv run pytest -m unit --no-cov

check: format lint test compose-config

compose-config:
	docker compose -f docker/compose.yaml config

up:
	docker compose -f docker/compose.yaml up -d postgres redis neo4j qdrant

down:
	docker compose -f docker/compose.yaml down

health:
	uv run python scripts/check_health.py --required postgres redis neo4j qdrant
