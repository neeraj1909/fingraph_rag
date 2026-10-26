"""Real Qdrant branch proves period prefiltering through the application service."""

import asyncio
import os
from datetime import date
from uuid import uuid4

import pytest

from fingraph_rag.domain.contracts import EvidenceLocator, TemporalContext
from fingraph_rag.retrieval import RetrievalBudget, RetrievalService
from fingraph_rag.vector.qdrant_store import QdrantVectorStore, VectorChunk

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("FINGRAPH_RUN_STORAGE_INTEGRATION") != "1", reason="requires local Qdrant"
    ),
]


class Embeddings:
    async def embed(self, texts):
        return [[1.0, 0.1] for _ in texts]


class NoGraph:
    async def execute_read(self, template_id, parameters):
        return []


def test_period_filtered_before_top_one_and_source_linked_fusion():
    async def run():
        store = QdrantVectorStore(
            Embeddings(),
            dimensions=2,
            collection="test_retrieval_" + uuid4().hex,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        try:
            await store.initialize()
            records = []
            for year in (2023, 2024, 2025):
                records.append(
                    VectorChunk(
                        evidence_id=f"revenue-{year}",
                        text=f"Acme revenue FY{year} 42 million",
                        locator=EvidenceLocator(
                            doc_id=f"synthetic-{year}",
                            filing_date=date(year + 1, 2, 1),
                            page=1,
                            table_id="income",
                            cell_range="B2",
                            quote_hash="synthetic1234",
                        ),
                        temporal=TemporalContext(
                            fiscal_year=year, filing_date=date(year + 1, 2, 1)
                        ),
                        kind="table",
                        entity_ids=["acme"],
                        metric_key="revenue",
                    )
                )
            await store.upsert_chunks(records)
            context = await RetrievalService(
                NoGraph(),
                store,
                aliases={"Acme": "acme"},
                budget=RetrievalBudget(candidate_limit=1),
            ).retrieve("Acme FY2024 revenue as filed")
            assert [e.evidence_id for e in context.table_evidence] == ["revenue-2024"]
            assert context.table_evidence[0].locator.cell_range == "B2"
            assert not context.warnings
        finally:
            await store._request("DELETE", store._path)

    asyncio.run(run())
