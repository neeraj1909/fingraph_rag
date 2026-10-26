"""Apply one shared budget to complete paths and source-backed documents."""

from collections.abc import Sequence

from fingraph_rag.domain.contracts import QuerySpec
from fingraph_rag.retrieval.contracts import (
    FusedContext,
    GraphPath,
    RejectedEvidence,
    RetrievalBudget,
    TableEvidence,
    TextEvidence,
)
from fingraph_rag.retrieval.temporal import compatible


def token_upper_bound(item: GraphPath | TextEvidence) -> int:
    # UTF-8 bytes upper-bound tokenizer units, including every serialized locator.
    return len(item.model_dump_json().encode("utf-8"))


class SubgraphPruner:
    def prune(
        self,
        query: QuerySpec,
        paths: Sequence[GraphPath],
        evidence: Sequence[TextEvidence],
        budget: RetrievalBudget,
    ) -> FusedContext:
        context = FusedContext(query=query)
        nodes: set[str] = set()
        edges: set[str] = set()
        sources: set[str] = set()
        seen: set[str] = set()
        latest_relations: dict[tuple, int] = {}

        def relation_key(relationship):
            temporal = relationship.temporal
            return (
                relationship.subject_entity_id,
                relationship.predicate,
                relationship.properties.get("metric_id", relationship.object_entity_id)
                if relationship.predicate == "REPORTED_METRIC"
                else relationship.object_entity_id,
                temporal.fiscal_year,
                temporal.fiscal_quarter,
                temporal.period_start,
                temporal.period_end,
            )

        if query.answer_mode == "latest_restatement":
            for path in paths:
                for relationship in path.relationships:
                    if (
                        compatible(relationship.temporal, query)
                        and relationship.status == "asserted"
                    ):
                        key = relation_key(relationship)
                        latest_relations[key] = max(
                            latest_relations.get(key, -1), relationship.temporal.restatement_version
                        )
        for path in paths:
            docs = {r.source_doc_id for r in path.relationships}
            next_nodes = nodes | set(path.entity_ids)
            next_edges = edges | {r.assertion_id for r in path.relationships}
            reason = None
            if path.path_id in seen:
                reason = "duplicate"
            elif len(path.relationships) > min(
                budget.max_hops, query.required_hops or budget.max_hops
            ):
                reason = "hop_budget"
            elif query.answer_mode == "latest_restatement" and any(
                r.temporal.restatement_version < latest_relations.get(relation_key(r), -1)
                for r in path.relationships
            ):
                reason = "superseded_assertion"
            elif any(
                not compatible(r.temporal, query) or r.status != "asserted"
                for r in path.relationships
            ):
                reason = "temporal_or_status"
            elif (
                len(context.graph_paths) >= budget.max_paths
                or len(next_nodes) > budget.max_nodes
                or len(next_edges) > budget.max_edges
            ):
                reason = "graph_budget"
            elif len(sources | docs) > budget.max_sources:
                reason = "source_budget"
            elif context.estimated_tokens + token_upper_bound(path) > budget.max_context_tokens:
                reason = "token_budget"
            if reason:
                context.rejected.append(RejectedEvidence(evidence_id=path.path_id, reason=reason))
                continue
            context.graph_paths.append(path)
            context.estimated_tokens += token_upper_bound(path)
            nodes, edges, sources = next_nodes, next_edges, sources | docs
            seen.add(path.path_id)
        latest: dict[tuple, int] = {}
        if query.answer_mode == "latest_restatement":
            for item in evidence:
                if item.metric_key and compatible(item.temporal, query):
                    key = (
                        tuple(sorted(item.entity_ids)),
                        item.metric_key,
                        item.temporal.fiscal_year,
                        item.temporal.fiscal_quarter,
                        item.temporal.period_start,
                        item.temporal.period_end,
                    )
                    latest[key] = max(latest.get(key, -1), item.temporal.restatement_version)
        evidence_seen: set[str] = set()
        for item in evidence:
            reason = None
            if item.evidence_id in evidence_seen:
                reason = "duplicate"
            elif (
                item.metric_key
                and query.answer_mode == "latest_restatement"
                and item.temporal.restatement_version
                < latest.get(
                    (
                        tuple(sorted(item.entity_ids)),
                        item.metric_key,
                        item.temporal.fiscal_year,
                        item.temporal.fiscal_quarter,
                        item.temporal.period_start,
                        item.temporal.period_end,
                    ),
                    -1,
                )
            ):
                reason = "superseded_observation"
            elif not compatible(item.temporal, query):
                reason = "temporal_mismatch"
            elif query.entities and not set(item.entity_ids) & set(query.entities):
                reason = "entity_mismatch"
            elif len(context.text_evidence) + len(context.table_evidence) >= budget.max_evidence:
                reason = "evidence_budget"
            elif len(sources | {item.locator.doc_id}) > budget.max_sources:
                reason = "source_budget"
            elif context.estimated_tokens + token_upper_bound(item) > budget.max_context_tokens:
                reason = "token_budget"
            if reason:
                context.rejected.append(
                    RejectedEvidence(evidence_id=item.evidence_id, reason=reason)
                )
                continue
            if item.kind == "table":
                context.table_evidence.append(TableEvidence.model_validate(item.model_dump()))
            else:
                context.text_evidence.append(item)
            context.estimated_tokens += token_upper_bound(item)
            sources.add(item.locator.doc_id)
            evidence_seen.add(item.evidence_id)
        return context
