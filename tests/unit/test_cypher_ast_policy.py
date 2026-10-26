import pytest

from fingraph_rag.domain.errors import PolicyViolation
from fingraph_rag.graph.cypher_ast_policy import CypherAstPolicy
from fingraph_rag.graph.cypher_templates import (
    METRIC_OBSERVATIONS,
    TRUSTED_READ_QUERIES,
    path_template,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("query", TRUSTED_READ_QUERIES)
def test_registered_query_structures_are_accepted(query):
    parsed = CypherAstPolicy().validate(query)
    assert "entity_ids" in parsed.parameters
    assert "limit" in parsed.parameters
    assert CypherAstPolicy().validate("\n  " + query.replace("\n", "  \n")).tree == parsed.tree


@pytest.mark.parametrize(
    "query",
    [
        "MATCH (n) DETACH DELETE n",
        "MATCH (n) RETURN n LIMIT 10",
        TRUSTED_READ_QUERIES[0] + "; CREATE (n)",
        TRUSTED_READ_QUERIES[0] + " UNION MATCH (n) RETURN n",
        TRUSTED_READ_QUERIES[0].replace("$entity_ids", "['unsafe']"),
        TRUSTED_READ_QUERIES[0].replace("$limit", "100000"),
        TRUSTED_READ_QUERIES[0] + " // comment",
        TRUSTED_READ_QUERIES[0] + " /* comment */",
        TRUSTED_READ_QUERIES[0] + " CALL apoc.load.json($url)",
        TRUSTED_READ_QUERIES[2].replace("*1..1", "*"),
        TRUSTED_READ_QUERIES[2].replace("*1..1", "*1..999"),
        TRUSTED_READ_QUERIES[0].replace("Entity", "`Entity`"),
        TRUSTED_READ_QUERIES[0].replace("MATCH", "МATCH"),
        TRUSTED_READ_QUERIES[0].replace("RETURN", "RETURN } CALL { CREATE (x) } RETURN"),
        "",
        "MATCH (n] RETURN n",
        "MATCH (n RETURN n",
    ],
)
def test_unregistered_or_unsafe_queries_fail_closed(query):
    with pytest.raises(PolicyViolation):
        CypherAstPolicy().validate(query)


def test_as_of_templates_enforce_effective_validity_window():
    assert "n.valid_from" in METRIC_OBSERVATIONS
    assert "n.valid_to" in METRIC_OBSERVATIONS
    query = path_template(1)
    assert "r.valid_from" in query
    assert "r.valid_to" in query
    assert "r.period_start" in query
    assert "r.period_end" in query
