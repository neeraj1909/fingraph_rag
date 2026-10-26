import asyncio
from datetime import date

import pytest

from fingraph_rag.domain.contracts import EvidenceLocator, TemporalContext
from fingraph_rag.vector.bm25 import Bm25Index, sparse_vector
from fingraph_rag.vector.fusion import RrfFusion
from fingraph_rag.vector.qdrant_store import QdrantVectorStore, VectorChunk

pytestmark = pytest.mark.unit


class Embeddings:
    async def embed(self, texts):
        return [[1.0, 0.5] for _ in texts]


def chunk(identity="chunk-1", **overrides):
    return VectorChunk(
        evidence_id=identity,
        text="Revenue increased to 42 million dollars",
        locator=EvidenceLocator(
            doc_id="doc-1", filing_date=date(2025, 2, 1), page=1, quote_hash="12345678"
        ),
        temporal=TemporalContext(filing_date=date(2025, 2, 1), fiscal_year=2024),
        **overrides,
    )


def test_rrf_rank_not_raw_score_and_no_duplicate_vote():
    fused = RrfFusion().fuse(
        [
            [{"evidence_id": "a", "score": 0.1}, {"evidence_id": "a", "score": 999}],
            [{"evidence_id": "b", "score": 999999}, {"evidence_id": "a", "score": 0.01}],
        ],
        2,
    )
    assert [item["evidence_id"] for item in fused] == ["a", "b"]
    assert fused[0]["score"] == pytest.approx(1 / 61 + 1 / 62)


def test_bm25_and_sparse_identity():
    index = Bm25Index({"a": "revenue revenue", "b": "supply chain", "c": "revenue"})
    assert [item["evidence_id"] for item in index.search("revenue")] == ["a", "c"]
    assert sparse_vector("Revenue revenue") == sparse_vector("revenue Revenue")
    assert len(sparse_vector("revenue revenue")["indices"]) == 1
    assert Bm25Index({}).search("revenue") == []


def test_vector_validation_and_query_budget():
    store = QdrantVectorStore(Embeddings(), dimensions=2)
    with pytest.raises(ValueError, match="table_id"):
        chunk(kind="table")
    with pytest.raises(ValueError, match="dimension"):
        store._dense([1.0])
    with pytest.raises(ValueError, match="finite"):
        store._dense([float("nan"), 1.0])
    with pytest.raises(ValueError, match="zero"):
        store._dense([0, 0])
    with pytest.raises(ValueError, match="unsupported"):
        store._filter({"arbitrary": "x"})
    with pytest.raises(ValueError, match="budget"):
        asyncio.run(store.search("revenue", 101))


def test_stable_point_identity_and_provenance():
    store = QdrantVectorStore(Embeddings(), dimensions=2)
    calls = []

    async def request(method, path, body):
        calls.append((method, body))

    store._request = request
    asyncio.run(store.upsert_chunks([chunk()]))
    asyncio.run(store.upsert_chunks([chunk()]))
    writes = [body for method, body in calls if method == "PUT"]
    assert writes[0] == writes[1]
    assert writes[0]["points"][0]["payload"]["locator"]["page"] == 1
    assert writes[0]["points"][0]["payload"]["fiscal_year"] == 2024
    with pytest.raises(ValueError, match="duplicate"):
        asyncio.run(store.upsert_chunks([chunk(), chunk()]))


def test_same_evidence_id_with_changed_provenance_is_rejected():
    store = QdrantVectorStore(Embeddings(), dimensions=2)
    existing = None

    async def request(method, path, body):
        nonlocal existing
        if method == "POST" and path.endswith("/points"):
            return [existing] if existing else []
        if method == "PUT":
            existing = body["points"][0]

    store._request = request
    asyncio.run(store.upsert_chunks([chunk()]))
    changed = chunk()
    changed.text = "A different source-backed fact"
    with pytest.raises(ValueError, match="different content or provenance"):
        asyncio.run(store.upsert_chunks([changed]))


def test_temporal_ranges_and_as_of_are_payload_prefilters():
    store = QdrantVectorStore(Embeddings(), dimensions=2)
    conditions = store._filter(
        {"as_of_date": "2025-02-01", "period_start": "2024-01-01", "period_end": "2024-12-31"}
    )["must"]
    assert conditions[0] == {"key": "filing_date", "range": {"lte": "2025-02-01T00:00:00Z"}}
    assert conditions[1]["should"][0] == {"is_empty": {"key": "valid_from"}}
    assert conditions[3] == {"key": "period_start", "range": {"gte": "2024-01-01T00:00:00Z"}}
    with pytest.raises(ValueError):
        store._filter({"as_of_date": "invalid"})


def test_latest_scan_fails_closed_when_eligibility_is_incomplete():
    store = QdrantVectorStore(Embeddings(), dimensions=2, max_version_scan=1)

    async def request(method, path, body):
        return {"points": [{"id": "a"}], "next_page_offset": "b"}

    store._request = request
    with pytest.raises(ValueError, match="scan budget"):
        asyncio.run(store.hybrid_search("revenue", 1, answer_mode="latest_restatement"))
