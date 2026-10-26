"""Parallel bounded retrieval over vendor-neutral graph and vector ports."""

import asyncio
from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from fingraph_rag.domain.contracts import QuerySpec
from fingraph_rag.graph.ports import GraphStore
from fingraph_rag.observability import get_telemetry
from fingraph_rag.retrieval.classifier import QueryClassifier
from fingraph_rag.retrieval.context_fusion import ContextFusion
from fingraph_rag.retrieval.contracts import (
    FusedContext,
    GraphPath,
    RejectedEvidence,
    RetrievalBudget,
    TableEvidence,
    TextEvidence,
)
from fingraph_rag.retrieval.decomposer import QueryDecomposer
from fingraph_rag.vector.ports import VectorStore


class RetrievalService:
    def __init__(
        self,
        graph: GraphStore,
        vector: VectorStore,
        *,
        aliases: Mapping[str, str] | None = None,
        budget: RetrievalBudget | None = None,
        fusion: ContextFusion | None = None,
    ) -> None:
        self.graph, self.vector = graph, vector
        self.classifier = QueryClassifier(aliases)
        self.budget = budget or RetrievalBudget()
        self.fusion = fusion or ContextFusion()

    async def retrieve(self, query: str | QuerySpec) -> FusedContext:
        spec = (
            self.classifier.classify(query)
            if isinstance(query, str)
            else query.model_copy(deep=True)
        )
        if spec.route is None:
            classified = self.classifier.classify(spec.question)
            spec.route = classified.route
        if len(spec.periods) > 8:
            raise ValueError("at most eight explicit periods per request")
        plans = QueryDecomposer().decompose(spec, self.budget)

        async def graph_branch() -> list[Mapping[str, Any]]:
            rows = []
            for plan in plans:
                rows.extend(await self.graph.execute_read(plan.template_id, plan.parameter_values))
            return rows

        async def vector_branch() -> list[dict[str, object]]:
            hybrid = getattr(self.vector, "hybrid_search", None)
            if hybrid is None:
                return list(await self.vector.search(spec.question, self.budget.candidate_limit))
            rows = []
            for period in spec.periods or [None]:
                filters: dict[str, Any] = {}
                if spec.entities:
                    filters["entity_ids"] = spec.entities
                if period:
                    filters.update(
                        {
                            key: value
                            for key, value in period.model_dump(exclude_none=True).items()
                            if key
                            in (
                                {
                                    "fiscal_year",
                                    "fiscal_quarter",
                                    "restatement_version",
                                    "as_of_date",
                                    "period_start",
                                    "period_end",
                                }
                                if getattr(self.vector, "supports_temporal_ranges", False)
                                else {"fiscal_year", "fiscal_quarter", "restatement_version"}
                            )
                        }
                    )
                if spec.answer_mode == "reported_as_filed":
                    filters["restatement_version"] = 0
                rows.extend(
                    await hybrid(
                        spec.question,
                        self.budget.candidate_limit,
                        filters=filters,
                        **(
                            {"answer_mode": spec.answer_mode}
                            if getattr(self.vector, "supports_answer_mode", False)
                            else {}
                        ),
                    )
                )
            return rows

        async def bounded(coro: Any, is_graph: bool = False) -> Any:
            with get_telemetry().span(
                "retrieval.graph" if is_graph else "retrieval.vector",
                {
                    "route.class": spec.route or "fact_table",
                    "retrieval.top_k": self.budget.candidate_limit,
                },
            ):
                return await asyncio.wait_for(coro, self.budget.branch_timeout_seconds)

        results = await asyncio.gather(
            bounded(graph_branch(), True),
            bounded(vector_branch()),
            return_exceptions=True,
        )
        paths, evidence, rejected, warnings = [], [], [], []
        if isinstance(results[0], BaseException):
            warnings.append("graph_branch_failed")
        else:
            for index, row in enumerate(results[0]):
                try:
                    paths.append(
                        GraphPath.model_validate(
                            {key: row[key] for key in ("path_id", "entity_ids", "relationships")}
                        )
                    )
                except (ValidationError, KeyError, TypeError):
                    rejected.append(
                        RejectedEvidence(
                            evidence_id=str(row.get("path_id", f"graph-row-{index}")),
                            reason="invalid_graph_provenance",
                        )
                    )
        if isinstance(results[1], BaseException):
            warnings.append("vector_branch_failed")
        else:
            for index, row in enumerate(results[1]):
                try:
                    model = TableEvidence if row.get("kind") == "table" else TextEvidence
                    payload = {
                        key: value for key, value in row.items() if key in model.model_fields
                    }
                    evidence.append(model.model_validate(payload))
                except (ValidationError, TypeError):
                    rejected.append(
                        RejectedEvidence(
                            evidence_id=str(row.get("evidence_id", f"vector-row-{index}")),
                            reason="invalid_document_provenance",
                        )
                    )
        context = self.fusion.fuse(spec, paths, evidence, self.budget)
        context.rejected.extend(rejected)
        context.warnings.extend(warnings)
        if getattr(self.vector, "hybrid_search", None) is None and (spec.entities or spec.periods):
            context.warnings.append("temporal_prefilter_unavailable")
        if not getattr(self.vector, "supports_temporal_ranges", False) and any(
            p.as_of_date or p.period_start or p.period_end for p in spec.periods
        ):
            context.warnings.append("range_prefilter_unavailable")
        if spec.answer_mode == "latest_restatement" and any(
            not isinstance(result, BaseException)
            and len(result) >= self.budget.candidate_limit
            and not (index == 1 and getattr(self.vector, "supports_global_latest", False))
            for index, result in enumerate(results)
        ):
            context.warnings.append("latest_candidate_completeness_unproven")
            # A truncated candidate pool cannot prove that a lower-ranked revision is absent.
            context.graph_paths.clear()
            context.text_evidence.clear()
            context.table_evidence.clear()
            context.estimated_tokens = 0
        if not spec.entities:
            context.warnings.append("unresolved_entity_anchor")
        return context
