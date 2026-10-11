import json
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from fingraph_rag.domain.contracts import EntityMention, EvidenceLocator
from fingraph_rag.ingestion.contracts import (
    CanonicalDocument,
    DocumentChunk,
    EntityResolutionBatch,
    EntityResolutionDecision,
    ExtractionEnvelope,
    ExtractionRecord,
    ExtractionRun,
    RawArtifact,
    SourceReference,
)

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "filings"


@pytest.mark.unit
def test_raw_artifact_identity_is_replayable() -> None:
    content = (FIXTURE_DIR / "synthetic_acme_10k.txt").read_bytes()
    acquired_at = datetime(2025, 2, 28, 12, 0, tzinfo=UTC)

    first = RawArtifact.from_bytes(
        content,
        source_uri="fixture://synthetic-acme/2024-10k",
        media_type="text/plain",
        acquired_at=acquired_at,
    )
    replay = RawArtifact.from_bytes(
        content,
        source_uri="fixture://synthetic-acme/2024-10k",
        media_type="text/plain",
        acquired_at=acquired_at,
    )

    assert first.artifact_id == replay.artifact_id
    assert first.content_sha256 == replay.content_sha256
    assert first.byte_size == len(content)


@pytest.mark.unit
def test_canonical_fixture_preserves_atomic_table_and_footnote_provenance() -> None:
    payload = json.loads((FIXTURE_DIR / "synthetic_acme_10k.json").read_text())
    document = CanonicalDocument.model_validate(payload)

    table = document.tables[0]
    margin_cell = next(cell for cell in table.cells if cell.cell_id == "cell_margin_2024")
    footnote = document.footnotes[0]
    table_chunk = next(chunk for chunk in document.chunks if chunk.family == "table_summary")

    assert margin_cell.footnote_ids == [footnote.footnote_id]
    assert footnote.linked_cell_ids == [margin_cell.cell_id]
    assert any(
        reference.source_id == margin_cell.cell_id for reference in table_chunk.source_references
    )


@pytest.mark.unit
def test_invalid_table_cell_locator_is_rejected() -> None:
    payload = json.loads((FIXTURE_DIR / "synthetic_acme_10k.json").read_text())
    payload["tables"][0]["cells"][0]["locator"]["cell_id"] = "wrong-cell"

    with pytest.raises(ValidationError, match="cell locator"):
        CanonicalDocument.model_validate(payload)


@pytest.mark.unit
def test_chunk_identity_includes_provenance_and_is_stable() -> None:
    references = [SourceReference(kind="table_cell", source_id="cell_margin_2024")]
    first = DocumentChunk.build(
        document_id="doc-1",
        ordinal=0,
        family="table_summary",
        text="FY2024 operating margin: 18.4%.",
        chunking_policy="table-summary-v1",
        source_references=references,
    )
    replay = DocumentChunk.build(
        document_id="doc-1",
        ordinal=0,
        family="table_summary",
        text="FY2024 operating margin: 18.4%.",
        chunking_policy="table-summary-v1",
        source_references=references,
    )

    assert first.chunk_id == replay.chunk_id
    assert first.content_sha256 == replay.content_sha256


@pytest.mark.unit
def test_extraction_and_resolution_records_match_their_run() -> None:
    locator = EvidenceLocator(
        doc_id="doc-1",
        filing_date=date(2025, 2, 28),
        section="Item 7",
        page=1,
        quote_hash="d" * 16,
    )
    entity = EntityMention(
        surface_form="Synthetic Acme",
        entity_type="Company",
        locator=locator,
        confidence=0.99,
    )
    run = ExtractionRun(
        run_id="run-1",
        document_id="doc-1",
        parser_version="fixture-parser-1",
        extractor_version="extractor-1",
        ontology_version="ontology-1",
        model_id="fixture-model",
        input_fingerprint="e" * 64,
        created_at=datetime(2025, 2, 28, tzinfo=UTC),
    )
    envelope = ExtractionEnvelope(
        document_id="doc-1",
        parser_version="fixture-parser-1",
        ontology_version="ontology-1",
        extraction_run_id="run-1",
        entities=[entity],
    )
    record = ExtractionRecord(run=run, envelope=envelope)
    decision = EntityResolutionDecision(
        mention=entity,
        decision="new",
        reason_code="no_identifier_match",
        confidence=0.8,
    )
    batch = EntityResolutionBatch(
        run_id=run.run_id,
        document_id=run.document_id,
        decisions=[decision],
        policy_version="resolution-1",
    )

    assert record.run.run_id == batch.run_id
    assert batch.decisions[0].mention.surface_form == "Synthetic Acme"

    with pytest.raises(ValidationError, match="run IDs"):
        ExtractionRecord(
            run=run,
            envelope=envelope.model_copy(update={"extraction_run_id": "other-run"}),
        )
