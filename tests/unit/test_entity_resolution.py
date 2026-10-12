from datetime import UTC, date, datetime

import pytest

from fingraph_rag.domain.contracts import CanonicalEntity, EntityMention, EvidenceLocator
from fingraph_rag.ingestion.entity_resolution import (
    EntityIndex,
    EntityResolver,
    normalize_identifier,
    normalize_surface_form,
)


def locator() -> EvidenceLocator:
    return EvidenceLocator(
        doc_id="doc-1",
        filing_date=date(2025, 2, 28),
        section="Item 7",
        page=1,
        quote_hash="f" * 16,
    )


def entity() -> CanonicalEntity:
    return CanonicalEntity(
        entity_id="company-acme",
        canonical_name="Synthetic Acme Corporation",
        entity_type="Company",
        identifiers={"CIK": "0000001", "ticker": "SAC"},
        merged_surface_forms=["Acme", "SAC"],
        resolution_method="identifier",
        confidence=1.0,
    )


@pytest.mark.unit
def test_identifier_normalization_is_case_and_punctuation_insensitive() -> None:
    assert normalize_identifier("CIK", "000-0001") == ("cik", "0000001")
    assert normalize_surface_form("Synthetic, Acme Corporation") == "synthetic acme corporation"


@pytest.mark.unit
def test_identifier_match_wins_over_alias_path() -> None:
    mention = EntityMention(
        surface_form="Acme",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        locator=locator(),
        confidence=0.9,
    )
    result = EntityResolver(EntityIndex([entity()])).resolve(
        document_id="doc-1",
        run_id="run-1",
        mentions=[mention],
        now=datetime(2025, 3, 1, tzinfo=UTC),
    )

    decision = result.batch.decisions[0]
    assert decision.selected_entity_id == "company-acme"
    assert decision.reason_code == "identifier_match"
    assert result.quarantined == []


@pytest.mark.unit
def test_unknown_mention_is_deferred_and_quarantined() -> None:
    mention = EntityMention(
        surface_form="Unknown Supplier",
        entity_type="Supplier",
        locator=locator(),
        confidence=0.7,
    )
    result = EntityResolver(EntityIndex([entity()])).resolve(
        document_id="doc-1",
        run_id="run-1",
        mentions=[mention],
        now=datetime(2025, 3, 1, tzinfo=UTC),
    )

    assert result.batch.unresolved_surface_forms == ["Unknown Supplier"]
    assert result.quarantined[0].code == "no_identifier_or_alias_match"


@pytest.mark.unit
def test_conflicting_identifier_is_not_auto_merged() -> None:
    other = entity().model_copy(update={"entity_id": "company-other"})
    mention = EntityMention(
        surface_form="Acme",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        locator=locator(),
        confidence=0.9,
    )
    result = EntityResolver(EntityIndex([entity(), other])).resolve(
        document_id="doc-1",
        run_id="run-1",
        mentions=[mention],
    )

    assert result.batch.decisions[0].decision == "defer"
    assert result.batch.decisions[0].reason_code == "identifier_conflict"
