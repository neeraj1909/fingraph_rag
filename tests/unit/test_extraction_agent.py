from datetime import UTC, datetime
from pathlib import Path

import pytest

from fingraph_rag.domain.contracts import ExtractionEnvelope
from fingraph_rag.ingestion.contracts import ExtractionRun, RawArtifact
from fingraph_rag.ingestion.extraction_agent import FixtureExtractionAgent
from fingraph_rag.ingestion.parsers import FixtureDocumentParser, ParseOptions

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "filings"


def parsed_fixture():
    raw_path = FIXTURE_DIR / "synthetic_acme_10k.txt"
    artifact = RawArtifact.from_bytes(
        raw_path.read_bytes(),
        source_uri="fixture://synthetic-acme/2024-10k",
        media_type="text/plain",
        acquired_at=datetime(2025, 2, 28, 12, tzinfo=UTC),
    )
    return FixtureDocumentParser().parse(
        (FIXTURE_DIR / "synthetic_acme_10k.json").read_bytes(),
        ParseOptions(
            parser_version="fixture-parser-1",
            expected_artifact_id=artifact.artifact_id,
            expected_source_content_sha256=artifact.content_sha256,
        ),
    )


@pytest.mark.unit
def test_fixture_extraction_agent_enforces_run_identity() -> None:
    document = parsed_fixture()
    run = ExtractionRun(
        run_id="run-fixture-1",
        document_id=document.document_id,
        parser_version="fixture-parser-1",
        extractor_version="fixture-extractor-1",
        ontology_version="ontology-1",
        model_id="fixture-model",
        input_fingerprint="a" * 64,
        created_at=datetime(2025, 3, 1, tzinfo=UTC),
    )
    envelope = ExtractionEnvelope(
        document_id=document.document_id,
        parser_version=run.parser_version,
        ontology_version=run.ontology_version,
        extraction_run_id=run.run_id,
        unresolved_mentions=["Unknown Supplier"],
    )

    record = FixtureExtractionAgent(envelope).extract(document, run)

    assert record.run.run_id == run.run_id
    assert record.envelope.unresolved_mentions == ["Unknown Supplier"]


@pytest.mark.unit
def test_fixture_extraction_agent_rejects_wrong_document() -> None:
    document = parsed_fixture()
    run = ExtractionRun(
        run_id="run-fixture-1",
        document_id=document.document_id,
        parser_version="fixture-parser-1",
        extractor_version="fixture-extractor-1",
        ontology_version="ontology-1",
        model_id="fixture-model",
        input_fingerprint="a" * 64,
        created_at=datetime(2025, 3, 1, tzinfo=UTC),
    )
    envelope = ExtractionEnvelope(
        document_id="other-document",
        parser_version=run.parser_version,
        ontology_version=run.ontology_version,
        extraction_run_id=run.run_id,
    )

    with pytest.raises(ValueError, match="document"):
        FixtureExtractionAgent(envelope).extract(document, run)
