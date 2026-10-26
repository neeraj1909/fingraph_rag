from datetime import date
from decimal import Decimal

import pytest

from fingraph_rag.domain.contracts import (
    CanonicalEntity,
    EvidenceLocator,
    ExtractionEnvelope,
    MetricObservation,
    RelationshipAssertion,
    TemporalContext,
)
from fingraph_rag.domain.errors import ValidationError
from fingraph_rag.graph.graph_serializer import envelope_records

pytestmark = pytest.mark.unit


def graph_fixture(prefix="test"):
    entities = [
        CanonicalEntity(
            entity_id=f"{prefix}:{name}",
            canonical_name=name,
            entity_type="Company",
            identifiers={},
            merged_surface_forms=[name],
            resolution_method="identifier",
            confidence=1,
        )
        for name in ("parent", "supplier", "child")
    ]
    locator = EvidenceLocator(
        doc_id=f"{prefix}:doc", filing_date=date(2025, 2, 1), page=1, quote_hash="abc12345"
    )
    temporal = TemporalContext(
        filing_date=locator.filing_date, fiscal_year=2024, fiscal_quarter="FY"
    )
    relationships = [
        RelationshipAssertion(
            assertion_id=f"{prefix}:edge:{i}",
            subject_entity_id=entities[a].entity_id,
            object_entity_id=entities[b].entity_id,
            predicate="SUPPLIES_TO",
            temporal=temporal,
            source_doc_id=locator.doc_id,
            extraction_run_id=f"{prefix}:run",
            locators=[locator],
            confidence=1,
        )
        for i, (a, b) in enumerate(((0, 1), (1, 2)))
    ]
    metric = MetricObservation(
        metric_id="revenue",
        metric_name="Revenue",
        value_decimal=Decimal("125"),
        unit="USD",
        statement="income_statement",
        temporal=temporal,
        locator=locator,
    )
    envelope = ExtractionEnvelope(
        document_id=locator.doc_id,
        parser_version="1",
        ontology_version="1",
        extraction_run_id=f"{prefix}:run",
        relationships=relationships,
        metrics=[metric],
    )
    return entities, envelope


def test_serialized_envelope_is_lossless_and_replayable():
    entities, envelope = graph_fixture()
    first = envelope_records(envelope, entities, entities[0].entity_id)
    assert first == envelope_records(envelope, entities, entities[0].entity_id)
    assert (
        RelationshipAssertion.model_validate_json(first[1][0]["payload"])
        == envelope.relationships[0]
    )
    assert MetricObservation.model_validate_json(first[2][0]["payload"]) == envelope.metrics[0]
    changed = envelope.model_copy(deep=True)
    changed.metrics[0].temporal.restatement_version = 1
    assert (
        first[2][0]["observation_id"]
        != envelope_records(changed, entities, entities[0].entity_id)[2][0]["observation_id"]
    )


def test_missing_issuer_or_unresolved_endpoints_fail_before_writes():
    entities, envelope = graph_fixture()
    with pytest.raises(ValidationError):
        envelope_records(envelope, entities)
    with pytest.raises(ValidationError):
        envelope_records(envelope, entities[:1], entities[0].entity_id)
    invalid = envelope.model_copy(deep=True)
    invalid.relationships[0].source_doc_id = "wrong-doc"
    with pytest.raises(ValidationError):
        envelope_records(invalid, entities, entities[0].entity_id)
