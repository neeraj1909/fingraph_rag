"""Opt-in real Qdrant 1.13 test; only its unique collection is removed."""

import asyncio
import os
from datetime import date
from uuid import uuid4

import pytest

from fingraph_rag.domain.contracts import EvidenceLocator, TemporalContext
from fingraph_rag.vector.qdrant_store import QdrantVectorStore, VectorChunk

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("FINGRAPH_RUN_STORAGE_INTEGRATION") != "1", reason="requires local Qdrant"
    ),
]


class FixtureEmbeddings:
    async def embed(self, texts):
        return [[1.0, 0.1] if "revenue" in text.lower() else [0.1, 1.0] for text in texts]


def test_real_qdrant_replay_hybrid_filters_and_provenance():
    async def run():
        store = QdrantVectorStore(
            FixtureEmbeddings(),
            dimensions=2,
            collection="test_fingraph_" + uuid4().hex,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        try:
            await store.initialize()
            await store.initialize()  # schema initialization is idempotent
            chunks = [
                VectorChunk(
                    evidence_id=f"revenue-{year}",
                    text=f"Revenue {year} was 42 million dollars",
                    locator=EvidenceLocator(
                        doc_id=f"doc-{year}",
                        page=1,
                        table_id="t1",
                        cell_range="B2",
                        filing_date=date(year + 1, 2, 1),
                        quote_hash="12345678",
                    ),
                    temporal=TemporalContext(
                        fiscal_year=year, fiscal_quarter="FY", filing_date=date(year + 1, 2, 1)
                    ),
                    kind="table",
                    entity_ids=["acme"],
                )
                for year in (2023, 2024)
            ]
            assert await store.upsert_chunks(chunks) == 2
            assert await store.upsert_chunks(chunks) == 2
            count = await store._request("POST", store._path + "/points/count", {"exact": True})
            assert count["count"] == 2
            results = await store.hybrid_search(
                "revenue", 1, filters={"fiscal_year": 2024, "kind": "table"}
            )
            assert len(results) == 1
            assert results[0]["evidence_id"] == "revenue-2024"
            assert results[0]["locator"]["cell_range"] == "B2"
            assert results[0]["temporal"]["fiscal_year"] == 2024
            assert await store.hybrid_search("revenue", 1, filters={"fiscal_year": 2022}) == []
        finally:
            await store._request("DELETE", store._path)

    asyncio.run(run())


def test_real_temporal_range_and_global_latest_before_top_one():
    async def run():
        store = QdrantVectorStore(
            FixtureEmbeddings(),
            dimensions=2,
            collection="test_temporal_" + uuid4().hex,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        try:
            await store.initialize()
            records = []
            for version in (0, 1):
                filed = date(2025, 2 + version, 1)
                records.append(
                    VectorChunk(
                        evidence_id=f"version-{version}",
                        text="revenue revenue revenue" if version == 0 else "corrected amount",
                        locator=EvidenceLocator(
                            doc_id=f"filing-{version}",
                            filing_date=filed,
                            page=1,
                            quote_hash="12345678",
                        ),
                        temporal=TemporalContext(
                            fiscal_year=2024,
                            fiscal_quarter="FY",
                            filing_date=filed,
                            period_start=date(2024, 1, 1),
                            period_end=date(2024, 12, 31),
                            restatement_version=version,
                        ),
                        entity_ids=["acme"],
                        metric_key="revenue",
                    )
                )
            await store.upsert_chunks(records)
            historical = await store.hybrid_search(
                "revenue",
                1,
                filters={
                    "as_of_date": "2025-02-15",
                    "period_start": "2024-01-01",
                    "period_end": "2024-12-31",
                },
                answer_mode="latest_restatement",
            )
            assert [r["evidence_id"] for r in historical] == ["version-0"]
            latest = await store.hybrid_search(
                "revenue", 1, filters={"entity_ids": ["acme"]}, answer_mode="latest_restatement"
            )
            assert [r["evidence_id"] for r in latest] == ["version-1"]
            assert (
                await store.hybrid_search("revenue", 1, filters={"period_start": "2024-06-01"})
                == []
            )
            empty = await store.hybrid_search(
                "revenue", 1, filters={"fiscal_year": 2022}, answer_mode="latest_restatement"
            )
            assert empty == []
        finally:
            await store._request("DELETE", store._path)

    asyncio.run(run())
