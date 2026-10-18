from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError

from fingraph_rag.domain.contracts import (
    Citation,
    EntityMention,
    EvidenceLocator,
    MetricObservation,
    PeriodConstraint,
    QuerySpec,
    TemporalContext,
    VerifiedClaim,
)


def locator() -> EvidenceLocator:
    return EvidenceLocator(
        doc_id="DOC-1",
        filing_date=date(2025, 2, 1),
        section="Item 7",
        page=12,
        quote_hash="a" * 16,
    )


def temporal() -> TemporalContext:
    return TemporalContext(
        fiscal_year=2024,
        fiscal_quarter="FY",
        filing_date=date(2025, 2, 1),
        period_start=date(2024, 1, 1),
        period_end=date(2024, 12, 31),
    )


@pytest.mark.unit
def test_financial_metric_mentions_are_typed_and_bounded() -> None:
    mention = EntityMention(
        surface_form="operating margin",
        entity_type="FinancialMetric",
        locator=locator(),
        confidence=0.91,
    )

    assert mention.entity_type == "FinancialMetric"

    with pytest.raises(ValidationError):
        EntityMention(
            surface_form="Revenue",
            entity_type="FinancialMetric",
            locator=locator(),
            confidence=1.01,
        )


@pytest.mark.unit
def test_temporal_ranges_and_query_periods_are_validated_separately() -> None:
    with pytest.raises(ValidationError):
        TemporalContext(
            fiscal_year=2024,
            filing_date=date(2025, 2, 1),
            period_start=date(2024, 12, 31),
            period_end=date(2024, 1, 1),
        )

    query = QuerySpec(
        question="What was FY2024 revenue?",
        entities=["Acme"],
        metrics=["revenue"],
        periods=[PeriodConstraint(fiscal_year=2024, fiscal_quarter="FY")],
    )
    assert query.periods[0].fiscal_year == 2024


@pytest.mark.unit
def test_metric_observation_requires_source_temporal_and_locator() -> None:
    observation = MetricObservation(
        metric_id="METRIC-1",
        metric_name="revenue",
        value_decimal=Decimal("123.40"),
        unit="USD",
        statement="income_statement",
        temporal=temporal(),
        locator=locator(),
    )

    assert observation.value_decimal == Decimal("123.40")


@pytest.mark.unit
def test_verified_claim_requires_dual_grounding() -> None:
    with pytest.raises(ValidationError):
        VerifiedClaim(
            claim_id="CLAIM-1",
            text="Revenue increased.",
            verification="verified",
        )

    citation = Citation(
        citation_id="CITE-1",
        document=locator(),
        graph_path_id="PATH-1",
        relation_path="(Company)-[REPORTED_METRIC]->(FinancialMetric)",
    )
    claim = VerifiedClaim(
        claim_id="CLAIM-1",
        text="Revenue increased.",
        citations=[citation],
        verification="verified",
    )
    assert claim.citations[0].graph_path_id == "PATH-1"


@pytest.mark.unit
def test_abstention_requires_a_reason() -> None:
    with pytest.raises(ValidationError):
        VerifiedClaim(
            claim_id="CLAIM-1",
            text="Unsupported claim.",
            verification="abstained",
        )
