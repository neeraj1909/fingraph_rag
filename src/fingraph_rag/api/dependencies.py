"""Explicit synthetic runtime or configured live vendor adapters; no implicit demo data."""

import asyncio
import json
import os
from pathlib import Path
from urllib.request import Request, urlopen

from fingraph_rag.application.query_service import QueryService
from fingraph_rag.graph.neo4j_store import Neo4jGraphStore
from fingraph_rag.guardrails.pipeline import GuardrailPipeline
from fingraph_rag.retrieval.contracts import RetrievalBudget
from fingraph_rag.retrieval.fixture_adapter import FixtureGraph, FixtureVector, load_fixture
from fingraph_rag.retrieval.service import RetrievalService
from fingraph_rag.settings import get_settings
from fingraph_rag.vector.qdrant_store import QdrantVectorStore
from fingraph_rag.workflow.generators import ExtractiveGenerationAdapter, PydanticGenerationAdapter

__all__ = ["FixtureGraph", "FixtureVector", "configured_service"]


def _load_entity_aliases(raw: str | None, *, required: bool = False) -> dict[str, str]:
    """Load the trusted surface-form to canonical-ID map for live retrieval.

    Query classification is deliberately deterministic and does not perform open-ended
    entity search. Keeping this map explicit makes the graph lookup boundary auditable
    and prevents user text from becoming a canonical graph identifier.
    """
    if not raw:
        if required:
            raise ValueError("live runtime requires FINGRAPH_ENTITY_ALIASES")
        return {}
    try:
        values = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError("FINGRAPH_ENTITY_ALIASES must be a JSON object") from exc
    if not isinstance(values, dict) or not values:
        raise ValueError("FINGRAPH_ENTITY_ALIASES must be a non-empty JSON object")
    aliases: dict[str, str] = {}
    seen: dict[str, str] = {}
    for alias, entity_id in values.items():
        if (
            not isinstance(alias, str)
            or not alias.strip()
            or len(alias) > 256
            or not isinstance(entity_id, str)
            or not entity_id.strip()
            or len(entity_id) > 256
        ):
            raise ValueError("FINGRAPH_ENTITY_ALIASES contains an invalid alias")
        normalized = alias.casefold()
        prior = seen.get(normalized)
        if prior is not None and prior != entity_id:
            raise ValueError("FINGRAPH_ENTITY_ALIASES contains an ambiguous alias")
        seen[normalized] = entity_id
        aliases[alias] = entity_id
    return aliases


class OpenAICompatibleEmbeddings:
    """Configured embeddings HTTP boundary; no fallback to synthetic vectors."""

    def __init__(self, url, model, api_key):
        self.url, self.model, self.api_key = url, model, api_key

    async def embed(self, texts):
        def request():
            req = Request(
                self.url,
                data=json.dumps({"model": self.model, "input": list(texts)}).encode(),
                headers={
                    "Authorization": "Bearer " + self.api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urlopen(req, timeout=15) as response:
                result = json.load(response)
            rows = sorted(result["data"], key=lambda row: row["index"])
            if [row["index"] for row in rows] != list(range(len(texts))):
                raise ValueError("embedding response indices mismatch")
            return [row["embedding"] for row in rows]

        return await asyncio.to_thread(request)


def configured_service():
    settings = get_settings()
    mode = os.getenv("FINGRAPH_RUNTIME_MODE", "live")
    production = settings.app_env == "prod"
    if production and not os.getenv("FINGRAPH_API_TOKEN"):
        raise ValueError("production requires API authentication")
    probe = None
    if mode == "synthetic":
        if production:
            raise ValueError("synthetic runtime is forbidden in prod")
        path = os.getenv("FINGRAPH_FIXTURE")
        if not path:
            raise ValueError("synthetic runtime requires FINGRAPH_FIXTURE")
        fixture = load_fixture(Path(path))
        retrieval = RetrievalService(
            fixture.graph(),
            fixture.vector(),
            aliases=fixture.aliases,
            budget=RetrievalBudget(
                max_hops=settings.max_hops,
                max_context_tokens=settings.max_context_tokens,
            ),
        )
        generator = ExtractiveGenerationAdapter()
    elif mode == "live":
        aliases = _load_entity_aliases(os.getenv("FINGRAPH_ENTITY_ALIASES"), required=True)
        required = [
            "FINGRAPH_MODEL",
            "FINGRAPH_EMBEDDING_URL",
            "FINGRAPH_EMBEDDING_MODEL",
            "FINGRAPH_EMBEDDING_API_KEY",
            "FINGRAPH_NEO4J_READ_PASSWORD" if production else "FINGRAPH_NEO4J_PASSWORD",
        ]
        if any(not os.getenv(key) for key in required):
            raise ValueError("live runtime is unconfigured")
        if production and (
            not os.getenv("FINGRAPH_NEO4J_READ_USER")
            or not os.getenv("FINGRAPH_NEO4J_READ_PASSWORD")
            or os.getenv("FINGRAPH_NEO4J_READ_USER") == os.getenv("FINGRAPH_NEO4J_USER", "neo4j")
        ):
            raise ValueError("production requires separate graph reader credentials")
        graph = Neo4jGraphStore(
            os.getenv("FINGRAPH_NEO4J_HTTP_URL", "http://localhost:7474"),
            username=os.environ["FINGRAPH_NEO4J_READ_USER"]
            if production
            else os.getenv("FINGRAPH_NEO4J_USER", "neo4j"),
            password=os.environ["FINGRAPH_NEO4J_READ_PASSWORD"]
            if production
            else os.environ["FINGRAPH_NEO4J_PASSWORD"],
            read_username=os.getenv("FINGRAPH_NEO4J_READ_USER"),
            read_password=os.getenv("FINGRAPH_NEO4J_READ_PASSWORD"),
        )
        embeddings = OpenAICompatibleEmbeddings(
            os.environ["FINGRAPH_EMBEDDING_URL"],
            os.environ["FINGRAPH_EMBEDDING_MODEL"],
            os.environ["FINGRAPH_EMBEDDING_API_KEY"],
        )
        vector = QdrantVectorStore(
            embeddings,
            url=os.getenv("FINGRAPH_QDRANT_URL", "http://localhost:6333"),
            dimensions=int(os.getenv("FINGRAPH_EMBEDDING_DIMENSIONS", "384")),
        )

        async def probe():
            # Neo4j rejects an empty entity_ids list. Probe the same bounded,
            # parameterized read used by requests with a configured canonical ID.
            await graph.execute_read(
                "entity_lookup", {"entity_ids": [next(iter(aliases.values()))], "limit": 1}
            )
            await vector._request("GET", vector._path)
            if production:
                rows = await asyncio.to_thread(
                    graph._transaction, [("SHOW CURRENT USER", {})], read=True
                )
                roles = set(rows[0][0].get("roles", [])) if rows and rows[0] else set()
                if "reader" not in roles or roles - {"reader", "PUBLIC"}:
                    raise ValueError("graph credentials are not restricted to built-in reader role")

        retrieval = RetrievalService(
            graph,
            vector,
            aliases=aliases,
            budget=RetrievalBudget(
                max_hops=settings.max_hops,
                max_context_tokens=settings.max_context_tokens,
            ),
        )
        generator = PydanticGenerationAdapter(os.environ["FINGRAPH_MODEL"])
    else:
        raise ValueError("unknown runtime mode")
    return QueryService(
        retrieval,
        generator,
        GuardrailPipeline(),
        state_directory=os.getenv("FINGRAPH_STATE_DIRECTORY", ".fingraph-state"),
        timeout_seconds=settings.max_query_seconds,
        max_repairs=settings.cypher_repair_retries,
        dependency_probe=probe,
    )
