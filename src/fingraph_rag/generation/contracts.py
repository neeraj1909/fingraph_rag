"""Candidates remain unverified until the output policy accepts them."""

from pydantic import Field

from fingraph_rag.domain.contracts import (
    CalculationRequest,
    CalculationResult,
    ContractModel,
    MetricObservation,
)
from fingraph_rag.retrieval.contracts import FusedContext


class CandidateClaim(ContractModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1, max_length=12000)
    evidence_ids: list[str] = Field(default_factory=list, max_length=24)
    graph_path_ids: list[str] = Field(default_factory=list, max_length=16)
    calculation_ids: list[str] = Field(default_factory=list, max_length=8)


class GenerationContext(ContractModel):
    context: FusedContext
    metrics: dict[str, MetricObservation] = Field(default_factory=dict)
    calculations: dict[str, CalculationResult] = Field(default_factory=dict)

    calculation_error: str | None = None
    calculation_requests: dict[str, CalculationRequest] = Field(default_factory=dict)

    @classmethod
    def from_context(cls, context: FusedContext) -> "GenerationContext":
        metrics = {}
        evidence = {e.evidence_id: e for e in context.text_evidence + context.table_evidence}
        for path in context.graph_paths:
            for relationship in path.relationships:
                raw = relationship.properties.get("metric_observation")
                if raw is None:
                    continue
                try:
                    metric = MetricObservation.model_validate(raw)
                except ValueError:
                    continue
                for evidence_id, item in evidence.items():
                    if item.locator == metric.locator and metric.locator in relationship.locators:
                        metrics[evidence_id] = metric
        return cls(context=context, metrics=metrics)

    def evidence(self):
        return {e.evidence_id: e for e in self.context.text_evidence + self.context.table_evidence}
