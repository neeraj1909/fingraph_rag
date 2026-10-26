"""Opt-in real graph/vector through the compiled production workflow."""

import asyncio
import os
from pathlib import Path
from uuid import uuid4

import pytest
from tests.unit.test_graph_serializer import graph_fixture

from fingraph_rag.application.query_service import QueryService
from fingraph_rag.domain.contracts import MetricObservation
from fingraph_rag.graph.graph_serializer import GraphWriter
from fingraph_rag.graph.neo4j_store import Neo4jGraphStore
from fingraph_rag.guardrails.pipeline import GuardrailPipeline
from fingraph_rag.retrieval.service import RetrievalService
from fingraph_rag.vector.qdrant_store import QdrantVectorStore, VectorChunk
from fingraph_rag.workflow.generators import ExtractiveGenerationAdapter

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.getenv("FINGRAPH_RUN_NEO4J_INTEGRATION") != "1"
        or os.getenv("FINGRAPH_RUN_STORAGE_INTEGRATION") != "1",
        reason="requires local Neo4j and Qdrant",
    ),
]


class FixtureEmbeddings:
    async def embed(self, texts):
        return [[1.0, 0.1] for _ in texts]


def test_real_storage_to_compiled_workflow_answer_and_replay(tmp_path: Path):
    async def run():
        prefix = "workflow-storage:" + uuid4().hex
        graph = Neo4jGraphStore(
            os.getenv("NEO4J_HTTP_URL", "http://localhost:7474"),
            password=os.getenv("NEO4J_PASSWORD", "change-me-before-use"),
        )
        vector = QdrantVectorStore(
            FixtureEmbeddings(),
            dimensions=2,
            collection="test_workflow_" + uuid4().hex,
            url=os.getenv("QDRANT_URL", "http://localhost:6333"),
        )
        entities, envelope = graph_fixture(prefix)
        metric = MetricObservation.model_validate_json(envelope.metrics[0].model_dump_json())
        writer = GraphWriter(graph)
        await graph.initialize_schema()
        await vector.initialize()
        try:
            await writer.write_envelope(envelope, entities, entities[0].entity_id)
            await vector.upsert_chunks(
                [
                    VectorChunk(
                        evidence_id=prefix + ":revenue",
                        text="Parent FY2024 revenue was USD 125.",
                        locator=metric.locator,
                        temporal=metric.temporal,
                        entity_ids=[entities[0].entity_id],
                        metric_key="revenue",
                    )
                ]
            )
            retrieval = RetrievalService(
                graph,
                vector,
                aliases={"Parent": entities[0].entity_id},
            )
            service = await QueryService(
                retrieval,
                ExtractiveGenerationAdapter(),
                GuardrailPipeline(),
                state_directory=str(tmp_path),
            ).open()
            try:
                request_id = service.submit(
                    "How does Parent revenue FY2024 as filed through Supplier?"
                )
                events = [event async for event in service.stream(request_id)]
                assert events[-1].payload["outcome"] == "answer", [e.model_dump() for e in events]
                answer = next(event for event in events if event.event_type == "answer")
                claims = answer.payload["claims"]
                assert claims and all(claim["citations"] for claim in claims)
                assert claims[0]["citations"][0]["document"]
                assert claims[0]["citations"][0]["graph_path_id"]
                assert "125" in claims[0]["text"]
            finally:
                await service.close()
            reopened = await QueryService(
                retrieval,
                ExtractiveGenerationAdapter(),
                GuardrailPipeline(),
                state_directory=str(tmp_path),
            ).open()
            try:
                replay = [
                    event async for event in reopened.stream(request_id, after=events[-2].sequence)
                ]
                assert [event.event_type for event in replay] == ["done"]
            finally:
                await reopened.close()
        finally:
            await vector._request("DELETE", vector._path)
            graph._transaction(
                [
                    (
                        "MATCH (n) WHERE n.entity_id STARTS WITH $prefix DETACH DELETE n",
                        {"prefix": prefix},
                    )
                ]
            )

    asyncio.run(run())
