"""Qdrant REST adapter with deterministic identity and explicit provenance.

Uses the Qdrant 1.13 REST surface; vendor transport stays outside the domain.
Embeddings are supplied by the caller's gateway, never silently synthesized.
"""

import asyncio
import hashlib
import json
import math
import re
from collections.abc import Sequence
from datetime import date
from typing import Any, Literal
from urllib.error import HTTPError
from urllib.request import Request, urlopen
from uuid import NAMESPACE_URL, uuid5

from pydantic import Field, model_validator

from fingraph_rag.domain.contracts import ContractModel, EvidenceLocator, TemporalContext
from fingraph_rag.infrastructure.embedding_gateway import EmbeddingGateway
from fingraph_rag.vector.bm25 import sparse_vector
from fingraph_rag.vector.fusion import RrfFusion


class VectorChunk(ContractModel):
    evidence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    locator: EvidenceLocator
    temporal: TemporalContext
    kind: Literal["text", "table"] = "text"
    metric_key: str | None = Field(default=None, min_length=1)
    entity_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provenance(self) -> "VectorChunk":
        if self.temporal.filing_date != self.locator.filing_date:
            raise ValueError("temporal and locator filing dates differ")
        if self.kind == "table" and not self.locator.table_id:
            raise ValueError("table evidence requires table_id")
        return self


class QdrantVectorStore:
    supports_temporal_ranges = True
    supports_answer_mode = True
    supports_global_latest = True

    def __init__(
        self,
        embeddings: EmbeddingGateway,
        *,
        url: str = "http://localhost:6333",
        collection: str = "fingraph_chunks",
        dimensions: int = 384,
        api_key: str | None = None,
        timeout: float = 10,
        max_results: int = 100,
        max_version_scan: int = 1000,
    ):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", collection):
            raise ValueError("invalid collection name")
        if dimensions < 1 or max_results < 1 or timeout <= 0 or max_version_scan < 1:
            raise ValueError("dimensions, max_results and timeout must be positive")
        self.embeddings = embeddings
        self.url, self.collection = url.rstrip("/"), collection
        self.dimensions, self.api_key = dimensions, api_key
        self.timeout, self.max_results = timeout, max_results
        self.max_version_scan = max_version_scan

    async def _request(self, method: str, path: str, body: Any = None) -> Any:
        def perform():
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["api-key"] = self.api_key
            request = Request(
                self.url + path,
                data=json.dumps(body).encode() if body is not None else None,
                method=method,
                headers=headers,
            )
            with urlopen(request, timeout=self.timeout) as response:
                data = json.load(response)
            return data["result"]

        return await asyncio.to_thread(perform)

    @property
    def _path(self) -> str:
        return f"/collections/{self.collection}"

    async def initialize(self) -> None:
        try:
            info = await self._request("GET", self._path)
        except HTTPError as error:
            if error.code != 404:
                raise
            await self._request(
                "PUT",
                self._path,
                {
                    "vectors": {"dense": {"size": self.dimensions, "distance": "Cosine"}},
                    "sparse_vectors": {"sparse": {"index": {"on_disk": False}}},
                },
            )
        else:
            config = info["config"]["params"]
            dense = config["vectors"].get("dense", {})
            if dense.get("size") != self.dimensions or dense.get("distance") != "Cosine":
                raise ValueError("existing Qdrant dense schema does not match")
            if "sparse" not in config.get("sparse_vectors", {}):
                raise ValueError("existing Qdrant collection lacks sparse vectors")
        for field, kind in {
            "doc_id": "keyword",
            "entity_ids": "keyword",
            "kind": "keyword",
            "fiscal_year": "integer",
            "fiscal_quarter": "keyword",
            "restatement_version": "integer",
            "filing_date": "datetime",
            "period_start": "datetime",
            "period_end": "datetime",
            "valid_from": "datetime",
            "valid_to": "datetime",
            "metric_key": "keyword",
        }.items():
            await self._request(
                "PUT", self._path + "/index?wait=true", {"field_name": field, "field_schema": kind}
            )

    def _dense(self, vector: Sequence[float]) -> list[float]:
        result = list(vector)
        if len(result) != self.dimensions or not all(math.isfinite(v) for v in result):
            raise ValueError("embedding dimension or finite-value mismatch")
        if not any(result):
            raise ValueError("zero embeddings are not meaningful cosine vectors")
        return result

    async def upsert_chunks(self, chunks: Sequence[VectorChunk]) -> int:
        if not chunks:
            return 0
        if len({chunk.evidence_id for chunk in chunks}) != len(chunks):
            raise ValueError("duplicate evidence identities in one batch")
        vectors = await self.embeddings.embed([chunk.text for chunk in chunks])
        if len(vectors) != len(chunks):
            raise ValueError("embedding result count mismatch")
        points = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            payload = chunk.model_dump(mode="json")
            payload.update(
                {
                    "doc_id": chunk.locator.doc_id,
                    "fiscal_year": chunk.temporal.fiscal_year,
                    "fiscal_quarter": chunk.temporal.fiscal_quarter,
                    "restatement_version": chunk.temporal.restatement_version,
                    "filing_date": chunk.temporal.filing_date.isoformat() + "T00:00:00Z",
                }
            )
            for field in ("period_start", "period_end", "valid_from", "valid_to"):
                value = getattr(chunk.temporal, field)
                if value is not None:
                    payload[field] = value.isoformat() + "T00:00:00Z"
            identity_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            payload["_identity_digest"] = hashlib.sha256(identity_payload.encode()).hexdigest()
            points.append(
                {
                    "id": str(uuid5(NAMESPACE_URL, "fingraph:" + chunk.evidence_id)),
                    "vector": {"dense": self._dense(vector), "sparse": sparse_vector(chunk.text)},
                    "payload": payload,
                }
            )
        existing = await self._request(
            "POST",
            self._path + "/points",
            {
                "ids": [point["id"] for point in points],
                "with_payload": True,
                "with_vector": False,
            },
        )
        for point in existing or []:
            payload = point.get("payload") or {}
            if payload.get("_identity_digest") != next(
                candidate["payload"]["_identity_digest"]
                for candidate in points
                if candidate["id"] == str(point.get("id"))
            ):
                raise ValueError(
                    "evidence identity already exists with different content or provenance"
                )
        await self._request("PUT", self._path + "/points?wait=true", {"points": points})
        return len(points)

    async def upsert(self, records: Sequence[dict[str, object]]) -> int:
        return await self.upsert_chunks([VectorChunk.model_validate(item) for item in records])

    def _filter(self, filters: dict[str, Any] | None) -> dict[str, Any] | None:
        if not filters:
            return None
        allowed = {
            "doc_id",
            "entity_ids",
            "kind",
            "fiscal_year",
            "fiscal_quarter",
            "restatement_version",
            "as_of_date",
            "period_start",
            "period_end",
        }
        if filters.keys() - allowed:
            raise ValueError("unsupported payload filter")
        conditions = []
        for key, value in filters.items():
            if key in {"as_of_date", "period_start", "period_end"}:
                parsed = value if isinstance(value, date) else date.fromisoformat(value)
                timestamp = parsed.isoformat() + "T00:00:00Z"
                if key == "as_of_date":
                    conditions.append({"key": "filing_date", "range": {"lte": timestamp}})
                    for field, operation in (("valid_from", "lte"), ("valid_to", "gte")):
                        conditions.append(
                            {
                                "should": [
                                    {"is_empty": {"key": field}},
                                    {"key": field, "range": {operation: timestamp}},
                                ]
                            }
                        )
                else:
                    conditions.append(
                        {
                            "key": key,
                            "range": {"gte" if key == "period_start" else "lte": timestamp},
                        }
                    )
                continue
            values = value if isinstance(value, list) else [value]
            if not values or any(type(item) not in (str, int) for item in values):
                raise ValueError("filter values must be nonempty string/integer scalars or lists")
            match = {"any": values} if isinstance(value, list) else {"value": value}
            conditions.append({"key": key, "match": match})
        return {"must": conditions}

    async def _latest_filter(self, payload_filter: dict[str, Any] | None) -> dict[str, Any]:
        # Complete bounded eligibility scan precedes ranking: top-k cannot prove latest.
        points = []
        offset = None
        while True:
            body: dict[str, Any] = {
                "limit": min(100, self.max_version_scan + 1 - len(points)),
                "with_payload": True,
                "with_vector": False,
            }
            if payload_filter:
                body["filter"] = payload_filter
            if offset is not None:
                body["offset"] = offset
            result = await self._request("POST", self._path + "/points/scroll", body)
            points.extend(result["points"])
            offset = result.get("next_page_offset")
            if len(points) > self.max_version_scan or (
                len(points) == self.max_version_scan and offset is not None
            ):
                raise ValueError("latest_restatement eligibility scan budget exceeded")
            if offset is None:
                break
        latest: dict[tuple, int] = {}
        grouped = []
        for point in points:
            chunk = VectorChunk.model_validate(
                {
                    key: value
                    for key, value in point["payload"].items()
                    if key in VectorChunk.model_fields
                }
            )
            temporal = chunk.temporal
            key = (
                (
                    tuple(sorted(chunk.entity_ids)),
                    chunk.metric_key,
                    temporal.fiscal_year,
                    temporal.fiscal_quarter,
                    temporal.period_start,
                    temporal.period_end,
                )
                if chunk.metric_key
                else (chunk.evidence_id,)
            )
            latest[key] = max(latest.get(key, -1), temporal.restatement_version)
            grouped.append((point["id"], key, temporal.restatement_version))
        selected = [identity for identity, key, version in grouped if version == latest[key]]
        return {"must": [*(payload_filter or {}).get("must", []), {"has_id": selected}]}

    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        *,
        filters: dict[str, Any] | None = None,
        answer_mode: Literal["latest_restatement", "both", "reported_as_filed"] = "both",
    ) -> list[dict[str, Any]]:
        if not query.strip() or len(query) > 20000:
            raise ValueError("query must contain 1..20000 characters")
        if not 1 <= limit <= self.max_results:
            raise ValueError("result budget exceeded")
        if answer_mode not in {"latest_restatement", "both", "reported_as_filed"}:
            raise ValueError("unsupported answer mode")
        if answer_mode == "reported_as_filed":
            filters = {**(filters or {}), "restatement_version": 0}
        payload_filter = self._filter(filters)
        if answer_mode == "latest_restatement":
            payload_filter = await self._latest_filter(payload_filter)
        vectors = await self.embeddings.embed([query])
        if len(vectors) != 1:
            raise ValueError("embedding result count mismatch")
        dense = self._dense(vectors[0])
        candidate_limit = min(limit * 4, self.max_results)

        async def search(name, vector):
            body = {"query": vector, "using": name, "limit": candidate_limit, "with_payload": True}
            if payload_filter:
                body["filter"] = payload_filter
            result = await self._request("POST", self._path + "/points/query", body)
            records = []
            for point in result["points"]:
                chunk = VectorChunk.model_validate(
                    {
                        key: value
                        for key, value in point["payload"].items()
                        if key in VectorChunk.model_fields
                    }
                )
                record = chunk.model_dump(mode="json")
                record["score"] = point["score"]
                if chunk.locator.table_id:
                    record.update(
                        table_id=chunk.locator.table_id, cell_range=chunk.locator.cell_range
                    )
                records.append(record)
            return records

        rankings = await asyncio.gather(
            search("dense", dense), search("sparse", sparse_vector(query))
        )
        return RrfFusion().fuse(rankings, limit)

    async def search(self, query: str, limit: int) -> list[dict[str, Any]]:
        return await self.hybrid_search(query, limit)
