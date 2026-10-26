"""Opt-in real Neo4j transactions with test-owned IDs and cleanup only."""

import asyncio
import os
from datetime import date
from decimal import Decimal
from uuid import uuid4

import pytest
from tests.unit.test_graph_serializer import graph_fixture

from fingraph_rag.domain.errors import DependencyUnavailable, ValidationError
from fingraph_rag.graph.graph_serializer import GraphWriter, envelope_records
from fingraph_rag.graph.neo4j_store import Neo4jGraphStore

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("FINGRAPH_RUN_NEO4J_INTEGRATION") != "1",
        reason="set FINGRAPH_RUN_NEO4J_INTEGRATION=1 for local Neo4j transactions",
    ),
]


def test_real_neo4j_replay_paths_restatements_and_atomic_conflict():
    async def run():
        prefix = "graph-test:" + uuid4().hex
        store = Neo4jGraphStore(
            os.environ.get("NEO4J_HTTP_URL", "http://localhost:7474"),
            password=os.environ.get("NEO4J_PASSWORD", "change-me-before-use"),
        )
        await store.initialize_schema()
        entities, envelope = graph_fixture(prefix)
        for assertion in envelope.relationships:
            assertion.temporal.period_start = date(2024, 1, 1)
            assertion.temporal.period_end = date(2024, 12, 31)
        envelope.metrics[0].temporal.period_start = date(2024, 1, 1)
        envelope.metrics[0].temporal.period_end = date(2024, 12, 31)
        writer = GraphWriter(store)
        cleanup_entity_ids = [entity.entity_id for entity in entities]
        cleanup_observation_ids = []
        try:
            nodes, _, metrics = envelope_records(envelope, entities, entities[0].entity_id)
            cleanup_entity_ids.extend(str(item["entity_id"]) for item in nodes)
            cleanup_observation_ids.extend(str(item["observation_id"]) for item in metrics)
            result = await writer.write_envelope(envelope, entities, entities[0].entity_id)
            assert (result.entities, result.relationships, result.metrics) == (4, 3, 1)
            await writer.write_envelope(envelope, entities, entities[0].entity_id)
            counts = store._transaction(
                [
                    (
                        "MATCH (a:Entity)-[r:ASSERTION]->() WHERE a.entity_id IN $ids "
                        "RETURN count(r) AS count",
                        {"ids": cleanup_entity_ids},
                    )
                ]
            )[0]
            assert counts == [{"count": 3}]
            paths = await store.execute_read(
                "bounded_paths",
                {
                    "entity_ids": [entities[0].entity_id],
                    "max_hops": 2,
                    "limit": 10,
                    "fiscal_year": 2024,
                },
            )
            assert any(
                row["entity_ids"][-1] == entities[2].entity_id and row["hop_count"] == 2
                for row in paths
            )
            assert (
                await store.execute_read(
                    "bounded_paths",
                    {
                        "entity_ids": [entities[0].entity_id],
                        "max_hops": 2,
                        "limit": 1,
                        "period_start": "2025-01-01",
                        "period_end": "2025-12-31",
                    },
                )
                == []
            )
            assert any(row["relationships"][0]["predicate"] == "REPORTED_METRIC" for row in paths)
            assert all(row["path_id"].startswith("path:") for row in paths)
            assert all(
                row["relationships"][0]["locators"][0]["doc_id"] == envelope.document_id
                for row in paths
            )
            assert (
                await store.execute_read(
                    "temporal_facts",
                    {"entity_ids": [entities[0].entity_id], "as_of_date": "2024-01-01"},
                )
                == []
            )
            assert (
                await store.execute_read(
                    "temporal_facts", {"entity_ids": [entities[0].entity_id], "fiscal_year": 2023}
                )
                == []
            )
            labels = store._transaction(
                [
                    (
                        "MATCH (n:Entity {entity_id: $id}) RETURN labels(n) AS labels",
                        {"id": entities[0].entity_id},
                    )
                ]
            )[0][0]["labels"]
            assert "Company" in labels
            # A restatement remains a separate source/version-scoped observation.
            restated = envelope.model_copy(deep=True)
            restated.relationships = []
            restated.metrics[0].temporal.restatement_version = 1
            restated.metrics[0].value_decimal = Decimal("130")
            nodes, _, metrics = envelope_records(restated, entities, entities[0].entity_id)
            cleanup_entity_ids.extend(str(item["entity_id"]) for item in nodes)
            cleanup_observation_ids.extend(str(item["observation_id"]) for item in metrics)
            await writer.write_envelope(restated, entities, entities[0].entity_id)
            observations = await store.execute_read(
                "metric_observations", {"entity_ids": [entities[0].entity_id]}
            )
            assert sorted(row["payload"]["value_decimal"] for row in observations) == ["125", "130"]
            # A conflicting payload at the same source/version identity rolls back
            # changes to canonical entities in that same server transaction.
            conflict = restated.model_copy(deep=True)
            conflict.metrics[0].value_decimal = Decimal("999")
            changed_entities = [entity.model_copy(deep=True) for entity in entities]
            changed_entities[0].canonical_name = "must roll back"
            with pytest.raises(DependencyUnavailable):
                await writer.write_envelope(conflict, changed_entities, entities[0].entity_id)
            persisted = await store.execute_read(
                "entity_lookup", {"entity_ids": [entities[0].entity_id]}
            )
            assert persisted[0]["payload"]["canonical_name"] == "parent"
            # Missing endpoints are rejected before a transaction is attempted.
            nodes, edges, metrics = envelope_records(envelope, entities, entities[0].entity_id)
            with pytest.raises(ValidationError):
                await store.write_records(nodes[:1], edges, metrics)
        finally:
            store._transaction(
                [
                    (
                        "MATCH (n:Entity) WHERE n.entity_id IN $ids DETACH DELETE n",
                        {"ids": cleanup_entity_ids},
                    ),
                    (
                        "MATCH (n:MetricObservation) WHERE n.observation_id IN $ids DELETE n",
                        {"ids": cleanup_observation_ids},
                    ),
                ]
            )

    asyncio.run(run())
