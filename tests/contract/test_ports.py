from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from fingraph_rag.graph.ports import GraphStore
from fingraph_rag.infrastructure.clock import Clock
from fingraph_rag.infrastructure.embedding_gateway import EmbeddingGateway
from fingraph_rag.infrastructure.model_gateway import ModelGateway
from fingraph_rag.ingestion.fetch import FetchedArtifact, SourceFetcher
from fingraph_rag.ingestion.sources import SourceDescriptor, SourceRegistry
from fingraph_rag.vector.ports import VectorStore


class FakeGraph:
    async def execute_read(
        self, template_id: str, parameters: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]:
        return []

    async def upsert_entities(self, entities: Sequence[Mapping[str, Any]]) -> int:
        return len(entities)

    async def upsert_relationships(self, relationships: Sequence[Mapping[str, Any]]) -> int:
        return len(relationships)


class FakeVector:
    async def upsert(self, records: Sequence[dict[str, object]]) -> int:
        return len(records)

    async def search(self, query: str, limit: int) -> Sequence[dict[str, object]]:
        return []


class FakeModel:
    async def generate(self, request: Any) -> Any:
        return request


class FakeEmbedding:
    async def embed(self, texts: Sequence[str]) -> Sequence[Sequence[float]]:
        return [[0.0] for _ in texts]


class FakeClock:
    def now(self):
        from datetime import UTC, datetime

        return datetime.now(UTC)


class FakeSourceRegistry:
    async def resolve(self, source_id: str) -> SourceDescriptor:
        return SourceDescriptor(
            source_id=source_id,
            source_uri="fixture://source.txt",
            media_type="text/plain",
            license_reference="synthetic-fixture",
        )


class FakeSourceFetcher:
    async def fetch(self, source: SourceDescriptor) -> FetchedArtifact:
        raise NotImplementedError


@pytest.mark.contract
def test_fake_adapters_satisfy_vendor_neutral_ports() -> None:
    assert isinstance(FakeGraph(), GraphStore)
    assert isinstance(FakeVector(), VectorStore)
    assert isinstance(FakeModel(), ModelGateway)
    assert isinstance(FakeEmbedding(), EmbeddingGateway)
    assert isinstance(FakeClock(), Clock)
    assert isinstance(FakeSourceRegistry(), SourceRegistry)
    assert isinstance(FakeSourceFetcher(), SourceFetcher)
