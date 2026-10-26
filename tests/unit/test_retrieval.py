"""Behavioral retrieval checks over adversarial source-linked fixtures."""

import asyncio
import json
from pathlib import Path

import pytest

from fingraph_rag.domain.contracts import PeriodConstraint, QuerySpec
from fingraph_rag.retrieval import GraphPath, RetrievalBudget, RetrievalService, TextEvidence
from fingraph_rag.retrieval.classifier import QueryClassifier
from fingraph_rag.retrieval.subgraph_pruner import SubgraphPruner

pytestmark = pytest.mark.unit


def fixture():
    return json.loads((Path(__file__).parents[1] / "fixtures/graph/sample.json").read_text())


class Graph:
    def __init__(self, rows):
        self.rows, self.calls = rows, []

    async def execute_read(self, template, parameters):
        self.calls.append((template, parameters))
        return self.rows


class Vector:
    def __init__(self, rows):
        self.rows, self.filters = rows, []

    async def search(self, query, limit):
        return self.rows

    async def hybrid_search(self, query, limit, *, filters=None):
        self.filters.append(filters)
        return self.rows


@pytest.mark.parametrize(
    ("question", "route"),
    [
        ("Parent FY2025 revenue?", "fact_table"),
        ("How does Supplier affect Parent through Subsidiary?", "multi_hop"),
        ("Compare Parent FY2025 revenue versus FY2024", "cross_period"),
        ("Parent risk landscape", "thematic"),
    ],
)
def test_four_routes(question, route):
    data = fixture()
    graph, vector = Graph(data["graph_paths"]), Vector(data["evidence"])
    context = asyncio.run(
        RetrievalService(graph, vector, aliases=data["aliases"]).retrieve(question)
    )
    assert context.query.route == route
    assert context.table_evidence[0].locator.doc_id == "synthetic-parent-fy2025"
    # A two-edge path is never truncated into misleading evidence for one-hop routes.
    assert bool(context.graph_paths) == (route == "multi_hop")
    assert all(call[1]["max_hops"] <= 4 for call in graph.calls)


def test_filters_applied_before_search_and_again_after_search():
    data = fixture()
    vector = Vector(data["evidence"])
    context = asyncio.run(
        RetrievalService(Graph([]), vector, aliases=data["aliases"]).retrieve(
            "Parent FY2024 revenue"
        )
    )
    assert vector.filters == [{"entity_ids": ["parent"], "fiscal_year": 2024}]
    assert not context.table_evidence
    assert context.rejected[0].reason == "temporal_mismatch"


def test_global_budget_counts_all_serialized_provenance():
    data = fixture()
    context = asyncio.run(
        RetrievalService(
            Graph(data["graph_paths"]),
            Vector(data["evidence"]),
            budget=RetrievalBudget(max_context_tokens=1),
        ).retrieve("supplier through parent")
    )
    assert not context.graph_paths and not context.table_evidence
    assert context.estimated_tokens == 0
    assert {item.reason for item in context.rejected} == {"token_budget"}


def test_bad_order_and_missing_provenance_rejected():
    data = fixture()
    data["graph_paths"][0]["entity_ids"] = ["supplier", "parent", "subsidiary"]
    del data["evidence"][0]["locator"]
    context = asyncio.run(
        RetrievalService(Graph(data["graph_paths"]), Vector(data["evidence"])).retrieve(
            "through parent"
        )
    )
    assert not context.graph_paths and not context.table_evidence
    assert {r.reason for r in context.rejected} == {
        "invalid_graph_provenance",
        "invalid_document_provenance",
    }


def test_restatement_modes_keep_latest_both_and_as_filed():
    data = fixture()["evidence"][0]
    original = TextEvidence.model_validate(data)
    revised = original.model_copy(deep=True)
    revised.evidence_id = "revised"
    revised.temporal.restatement_version = 1
    for mode, expected in [
        ("latest_restatement", ["revised"]),
        ("both", [original.evidence_id, "revised"]),
        ("reported_as_filed", [original.evidence_id]),
    ]:
        result = SubgraphPruner().prune(
            QuerySpec(question="revenue", answer_mode=mode),
            [],
            [original, revised],
            RetrievalBudget(),
        )
        assert [e.evidence_id for e in result.table_evidence] == expected


def test_independent_branch_timeout_returns_explicit_warning():
    class SlowGraph(Graph):
        async def execute_read(self, template, parameters):
            await asyncio.sleep(0.05)
            return []

    data = fixture()
    context = asyncio.run(
        RetrievalService(
            SlowGraph([]),
            Vector(data["evidence"]),
            budget=RetrievalBudget(branch_timeout_seconds=0.001),
        ).retrieve("revenue")
    )
    assert context.table_evidence
    assert "graph_branch_failed" in context.warnings


def test_no_unknown_entity_invention_and_invalid_graph_hops():
    assert QueryClassifier().classify("Unknown Corp revenue").entities == []
    data = fixture()["graph_paths"][0]
    path = GraphPath.model_validate(data)
    result = SubgraphPruner().prune(
        QuerySpec(question="through supplier", periods=[PeriodConstraint(fiscal_year=2024)]),
        [path],
        [],
        RetrievalBudget(),
    )
    assert not result.graph_paths
    assert result.rejected[0].reason == "temporal_or_status"


def test_latest_graph_assertion_version_and_both_mode():
    original = GraphPath.model_validate(fixture()["graph_paths"][0])
    revised = original.model_copy(deep=True)
    revised.path_id = "revised-path"
    for relationship in revised.relationships:
        relationship.assertion_id += "-v1"
        relationship.temporal.restatement_version = 1
    context = SubgraphPruner().prune(
        QuerySpec(question="through parent"), [original, revised], [], RetrievalBudget()
    )
    assert [p.path_id for p in context.graph_paths] == ["revised-path"]
    both = SubgraphPruner().prune(
        QuerySpec(question="through parent", answer_mode="both"),
        [original, revised],
        [],
        RetrievalBudget(),
    )
    assert len(both.graph_paths) == 2


def test_unanchored_question_is_explicit_and_empty_is_not_answer():
    context = asyncio.run(RetrievalService(Graph([]), Vector([])).retrieve("sector risk overview"))
    assert context.query.route == "thematic"
    assert context.warnings == ["unresolved_entity_anchor"]
    assert not context.graph_paths and not context.text_evidence and not context.table_evidence


def test_graph_node_edge_source_budgets_are_global():
    path = GraphPath.model_validate(fixture()["graph_paths"][0])
    for budget in [RetrievalBudget(max_nodes=2), RetrievalBudget(max_edges=1)]:
        context = SubgraphPruner().prune(QuerySpec(question="through parent"), [path], [], budget)
        assert not context.graph_paths
        assert context.rejected[0].reason == "graph_budget"
