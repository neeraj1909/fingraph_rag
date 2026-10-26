"""Security entry point retained for the project validation runbook."""

import asyncio

import pytest

from fingraph_rag.domain.errors import PolicyViolation, ValidationError
from fingraph_rag.graph.neo4j_store import Neo4jGraphStore

pytestmark = pytest.mark.security


@pytest.mark.parametrize(
    "params",
    [
        {"entity_ids": ["x"], "limit": 101},
        {"entity_ids": ["x"], "limit": True},
        {"entity_ids": ["x"], "max_hops": 5},
        {"entity_ids": ["x"], "max_hops": 0},
        {"entity_ids": ["x"], "cypher": "DELETE n"},
        {"entity_ids": []},
        {"entity_ids": "x"},
    ],
)
def test_invalid_reads_are_rejected_before_network(params):
    store = Neo4jGraphStore("http://localhost:1", password="test")
    with pytest.raises((PolicyViolation, ValidationError)):
        asyncio.run(store.execute_read("bounded_paths", params))


def test_arbitrary_query_cannot_be_used_as_template_id():
    store = Neo4jGraphStore("http://localhost:1", password="test")
    with pytest.raises(PolicyViolation):
        asyncio.run(store.execute_read("MATCH (n) RETURN n", {"entity_ids": ["x"]}))
