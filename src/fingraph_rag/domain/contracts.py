"""Strict, vendor-neutral contracts for FinGraph-RAG.

These models deliberately separate source temporal metadata from a user's
query period constraint. A query can ask for FY2025 before retrieval discovers
the filing date that supports the answer.
"""

from datetime import UTC, date, datetime
from decimal import Decimal
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ContractModel(BaseModel):
    """Base model with strict extra-field and assignment behavior."""

    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        str_strip_whitespace=True,
    )


FiscalQuarter = Literal["Q1", "Q2", "Q3", "Q4", "FY"]
EntityType = Literal[
    "Company",
    "Subsidiary",
    "Executive",
    "Supplier",
    "Customer",
    "FinancialMetric",
    "RiskFactor",
    "DebtInstrument",
    "GeographicRegion",
    "FilingPeriod",
]
RouteClass = Literal["fact_table", "multi_hop", "cross_period", "thematic"]
AnswerMode = Literal["reported_as_filed", "latest_restatement", "both"]


class EvidenceLocator(ContractModel):
    doc_id: str = Field(min_length=1)
    filing_date: date
    section: str | None = None
    page: int | None = Field(default=None, ge=1)
    table_id: str | None = None
    cell_range: str | None = None
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    quote_hash: str = Field(min_length=8)

    @model_validator(mode="after")
    def validate_character_range(self) -> "EvidenceLocator":
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must be greater than or equal to char_start")
        if not any(
            (self.section, self.page, self.table_id, self.cell_range, self.char_start is not None)
        ):
            raise ValueError(
                "an evidence locator needs a section, page, table, cell, or character offset"
            )
        return self


class TemporalContext(ContractModel):
    """Temporal metadata attached to a source-backed observation/assertion."""

    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    fiscal_quarter: FiscalQuarter | None = None
    filing_date: date
    as_of_date: date | None = None
    period_start: date | None = None
    period_end: date | None = None
    valid_from: date | None = None
    valid_to: date | None = None
    restatement_version: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_ranges(self) -> "TemporalContext":
        if self.period_start and self.period_end and self.period_end < self.period_start:
            raise ValueError("period_end must be greater than or equal to period_start")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to must be greater than or equal to valid_from")
        return self


class PeriodConstraint(ContractModel):
    """A query-time period constraint that does not require source metadata."""

    fiscal_year: int | None = Field(default=None, ge=1900, le=2200)
    fiscal_quarter: FiscalQuarter | None = None
    period_start: date | None = None
    period_end: date | None = None
    as_of_date: date | None = None
    restatement_version: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_period(self) -> "PeriodConstraint":
        if self.period_start and self.period_end and self.period_end < self.period_start:
            raise ValueError("period_end must be greater than or equal to period_start")
        if not any(
            (
                self.fiscal_year,
                self.fiscal_quarter,
                self.period_start,
                self.period_end,
                self.as_of_date,
                self.restatement_version is not None,
            )
        ):
            raise ValueError("a period constraint must specify at least one temporal field")
        return self


class EntityMention(ContractModel):
    surface_form: str = Field(min_length=1)
    entity_type: EntityType
    aliases: list[str] = Field(default_factory=list)
    identifiers: dict[str, str] = Field(default_factory=dict)
    locator: EvidenceLocator
    confidence: float = Field(ge=0, le=1)


class CanonicalEntity(ContractModel):
    entity_id: str = Field(min_length=1)
    canonical_name: str = Field(min_length=1)
    entity_type: EntityType
    identifiers: dict[str, str] = Field(default_factory=dict)
    merged_surface_forms: list[str] = Field(min_length=1)
    resolution_method: Literal["identifier", "exact", "embedding", "llm_verified", "human_review"]
    confidence: float = Field(ge=0, le=1)


class MetricObservation(ContractModel):
    metric_id: str = Field(min_length=1)
    metric_name: str = Field(min_length=1)
    value_decimal: Decimal
    unit: str = Field(min_length=1)
    currency: str | None = None
    scale: Literal["units", "thousands", "millions", "billions"] = "units"
    statement: Literal[
        "income_statement",
        "balance_sheet",
        "cash_flow",
        "segment",
        "non_gaap",
        "other",
    ]
    temporal: TemporalContext
    locator: EvidenceLocator
    is_estimate: bool = False


RelationshipType = Literal[
    "OWNS_STAKE_IN",
    "SUPPLIES_TO",
    "REPORTED_METRIC",
    "EXPOSED_TO_RISK",
    "HAS_COVENANT",
    "COMPETES_WITH",
    "AFFECTS",
    "LOCATED_IN",
]


class RelationshipAssertion(ContractModel):
    assertion_id: str = Field(min_length=1)
    subject_entity_id: str = Field(min_length=1)
    predicate: RelationshipType
    object_entity_id: str = Field(min_length=1)
    properties: dict[str, Any] = Field(default_factory=dict)
    temporal: TemporalContext
    source_doc_id: str = Field(min_length=1)
    extraction_run_id: str = Field(min_length=1)
    locators: list[EvidenceLocator] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    status: Literal["asserted", "superseded", "disputed"] = "asserted"


class ExtractionEnvelope(ContractModel):
    document_id: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    ontology_version: str = Field(min_length=1)
    extraction_run_id: str = Field(min_length=1)
    entities: list[EntityMention] = Field(default_factory=list)
    metrics: list[MetricObservation] = Field(default_factory=list)
    relationships: list[RelationshipAssertion] = Field(default_factory=list)
    unresolved_mentions: list[str] = Field(default_factory=list)


class EntityResolutionDecision(ContractModel):
    mention: EntityMention
    candidates: list[CanonicalEntity] = Field(default_factory=list)
    selected_entity_id: str | None = None
    decision: Literal["merge", "new", "defer", "reject"]
    reason_code: str = Field(min_length=1)
    evidence: list[EvidenceLocator] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def validate_selection(self) -> "EntityResolutionDecision":
        if self.decision == "merge" and not self.selected_entity_id:
            raise ValueError("merge decisions require selected_entity_id")
        if self.decision in {"defer", "reject"} and not self.reason_code:
            raise ValueError("defer/reject decisions require reason_code")
        return self


class QuerySpec(ContractModel):
    question: str = Field(min_length=1)
    route: RouteClass | None = None
    entities: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)
    periods: list[PeriodConstraint] = Field(default_factory=list)
    required_hops: int = Field(default=0, ge=0, le=4)
    needs_calculation: bool = False
    answer_mode: AnswerMode = "latest_restatement"


class CypherQueryPlan(ContractModel):
    template_id: str = Field(min_length=1)
    parameter_values: dict[str, Any] = Field(default_factory=dict)
    expected_labels: list[str] = Field(default_factory=list)
    expected_relationships: list[str] = Field(default_factory=list)
    max_hops: int = Field(default=0, ge=0, le=4)
    read_only: Literal[True] = True


FormulaID = Literal[
    "yoy_growth",
    "qoq_growth",
    "cagr",
    "margin",
    "leverage",
    "coverage",
    "sum",
    "difference",
    "ratio",
]


class CalculationRequest(ContractModel):
    formula_id: FormulaID
    inputs: list[str] = Field(min_length=1)
    output_unit: str = Field(min_length=1)
    precision: int = Field(default=2, ge=0, le=8)
    rounding: Literal["half_even", "half_up", "floor", "ceil"] = "half_even"


class CalculationResult(ContractModel):
    formula_id: FormulaID
    output_value: Decimal
    output_unit: str = Field(min_length=1)
    input_evidence_ids: list[str] = Field(min_length=1)
    calculator_version: str = Field(min_length=1)
    reproducibility_hash: str = Field(min_length=8)


class Citation(ContractModel):
    citation_id: str = Field(min_length=1)
    document: EvidenceLocator
    graph_path_id: str = Field(min_length=1)
    relation_path: str = Field(min_length=1)
    calculation_id: str | None = None


class VerifiedClaim(ContractModel):
    claim_id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    numeric_tokens: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    verification: Literal["verified", "abstained"]
    reason_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def require_grounding_or_abstention_reason(self) -> "VerifiedClaim":
        if self.verification == "verified" and not self.citations:
            raise ValueError("verified claims require document and graph citations")
        if self.verification == "abstained" and not self.reason_codes:
            raise ValueError("abstained claims require reason_codes")
        return self


class GuardrailDecision(ContractModel):
    stage: Literal["input", "execution", "output", "compliance"]
    action: Literal["allow", "redact", "reject", "abstain"]
    reason_codes: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    redaction_count: int = Field(default=0, ge=0)
    policy_version: str = Field(min_length=1)


class GraphRAGState(TypedDict, total=False):
    """LangGraph-compatible state shape without importing LangGraph."""

    request_id: str
    question: str
    query_spec: QuerySpec
    graph_paths: list[dict[str, Any]]
    evidence_ids: list[str]
    claims: list[VerifiedClaim]
    errors: list[str]


EventType = Literal[
    "accepted",
    "guardrail",
    "route",
    "retrieval",
    "graph_path",
    "calculation",
    "verification",
    "answer",
    "error",
    "done",
]


class EventEnvelope(ContractModel):
    schema_version: str = Field(min_length=1)
    event_type: EventType
    request_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    occurred_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_utc_timestamp(self) -> "EventEnvelope":
        if self.occurred_at.tzinfo is None:
            raise ValueError("occurred_at must include timezone information")
        return self
