"""Source-backed retrieval values and explicit execution budgets."""

from typing import Literal

from pydantic import Field, model_validator

from fingraph_rag.domain.contracts import (
    ContractModel,
    EvidenceLocator,
    QuerySpec,
    RelationshipAssertion,
    TemporalContext,
)


class RetrievalBudget(ContractModel):
    max_hops: int = Field(default=4, ge=0, le=4)
    max_nodes: int = Field(default=64, ge=1, le=1000)
    max_edges: int = Field(default=64, ge=1, le=1000)
    max_paths: int = Field(default=16, ge=1, le=100)
    max_evidence: int = Field(default=24, ge=1, le=100)
    max_sources: int = Field(default=12, ge=1, le=100)
    max_context_tokens: int = Field(default=12000, ge=1, le=100000)
    candidate_limit: int = Field(default=40, ge=1, le=100)
    branch_timeout_seconds: float = Field(default=10, gt=0, le=30)


class GraphPath(ContractModel):
    path_id: str = Field(min_length=1)
    entity_ids: list[str] = Field(min_length=2)
    relationships: list[RelationshipAssertion] = Field(min_length=1)

    @model_validator(mode="after")
    def ordered_path(self) -> "GraphPath":
        if len(self.entity_ids) != len(self.relationships) + 1:
            raise ValueError("path must contain one more entity than relationships")
        for index, relationship in enumerate(self.relationships):
            pair = {self.entity_ids[index], self.entity_ids[index + 1]}
            if pair != {relationship.subject_entity_id, relationship.object_entity_id}:
                raise ValueError("relationship endpoints must match ordered path")
            if any(
                loc.doc_id != relationship.source_doc_id
                or loc.filing_date != relationship.temporal.filing_date
                for loc in relationship.locators
            ):
                raise ValueError("relationship locator must identify its source document")
        return self


class TextEvidence(ContractModel):
    evidence_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    locator: EvidenceLocator
    temporal: TemporalContext
    entity_ids: list[str] = Field(default_factory=list)
    score: float = 0.0
    kind: Literal["text", "table"] = "text"
    metric_key: str | None = None
    table_id: str | None = None
    cell_range: str | None = None

    @model_validator(mode="after")
    def consistent_source(self) -> "TextEvidence":
        if self.locator.filing_date != self.temporal.filing_date:
            raise ValueError("source and temporal filing dates disagree")
        if self.kind == "table" and not (self.table_id or self.locator.table_id):
            raise ValueError("table evidence requires table identity")
        return self


class TableEvidence(TextEvidence):
    kind: Literal["table"] = "table"


class RejectedEvidence(ContractModel):
    evidence_id: str
    reason: str


class FusedContext(ContractModel):
    query: QuerySpec
    graph_paths: list[GraphPath] = Field(default_factory=list)
    text_evidence: list[TextEvidence] = Field(default_factory=list)
    table_evidence: list[TableEvidence] = Field(default_factory=list)
    rejected: list[RejectedEvidence] = Field(default_factory=list)
    estimated_tokens: int = Field(default=0, ge=0)
    warnings: list[str] = Field(default_factory=list)
