import asyncio
import os
from datetime import UTC, date, datetime
from pathlib import Path

import pytest
from tests.fixtures.docling_pdf import synthetic_acme_10q_pdf

from fingraph_rag.application.ingest_service import IngestionService
from fingraph_rag.domain.contracts import CanonicalEntity, EntityMention, EvidenceLocator
from fingraph_rag.ingestion.audit import InMemoryIngestionAuditStore
from fingraph_rag.ingestion.entity_resolution import EntityIndex, EntityResolver
from fingraph_rag.ingestion.fetch import FixtureSourceFetcher
from fingraph_rag.ingestion.identity import content_sha256
from fingraph_rag.ingestion.parsers import DoclingLayoutParser, ParseOptions
from fingraph_rag.ingestion.sources import InMemorySourceRegistry, SourceDescriptor
from fingraph_rag.ingestion.temporal_normalization import FiscalCalendar, RawPeriod


@pytest.mark.integration
def test_real_docling_pdf_replays_through_ingestion_boundary(tmp_path: Path) -> None:
    if os.getenv("FINGRAPH_RUN_DOCLING_INTEGRATION") != "1":
        pytest.skip("set FINGRAPH_RUN_DOCLING_INTEGRATION=1 for the real Docling smoke test")
    pytest.importorskip("docling.document_converter")

    pdf_path = tmp_path / "synthetic_acme_2024_q3_10q.pdf"
    content = synthetic_acme_10q_pdf()
    pdf_path.write_bytes(content)
    descriptor = SourceDescriptor(
        source_id="synthetic-acme-2024-q3-10q-docling-real",
        source_uri=f"fixture://{pdf_path.name}",
        media_type="application/pdf",
        license_reference="synthetic-fixture",
        expected_content_sha256=content_sha256(content),
    )
    registry = InMemorySourceRegistry([descriptor])
    fetcher = FixtureSourceFetcher(tmp_path)
    document_id = "doc_synthetic_acme_2024_q3_10q_docling_real"
    locator = EvidenceLocator(
        doc_id=document_id,
        filing_date=date(2024, 11, 1),
        section="Item 2",
        page=1,
        quote_hash="5" * 16,
    )
    service = IngestionService(
        resolver=EntityResolver(
            EntityIndex(
                [
                    CanonicalEntity(
                        entity_id="company_synthetic_acme",
                        canonical_name="Synthetic Acme Corporation",
                        entity_type="Company",
                        identifiers={"CIK": "0000001"},
                        merged_surface_forms=["Acme"],
                        resolution_method="identifier",
                        confidence=1.0,
                    )
                ]
            )
        ),
        audit_store=InMemoryIngestionAuditStore(),
        parser=RecordingParser(DoclingLayoutParser()),
    )
    parse_options = ParseOptions(
        parser_version="docling-2.129.0-adapter-1",
        expected_artifact_id="bound-at-runtime",
        expected_source_content_sha256="0" * 64,
        source_uri=descriptor.source_uri,
        document_id=document_id,
        issuer_id="company_synthetic_acme",
        document_type="sec_10q",
        filing_date=date(2024, 11, 1),
        title="Synthetic Acme Corporation Form 10-Q",
    )
    kwargs = {
        "source_id": descriptor.source_id,
        "source_registry": registry,
        "source_fetcher": fetcher,
        "parse_options": parse_options,
        "mentions": [
            EntityMention(
                surface_form="Acme",
                entity_type="Company",
                identifiers={"CIK": "0000001"},
                locator=locator,
                confidence=0.9,
            )
        ],
        "periods": [RawPeriod(label="Q3 2024")],
        "calendar": FiscalCalendar(
            issuer_id="company_synthetic_acme",
            fiscal_year_end_month=12,
            fiscal_year_end_day=31,
        ),
        "parser_version": "docling-2.129.0-adapter-1",
        "extractor_version": "docling-smoke-extractor-1",
        "model_id": "docling-2.129.0",
        "now": datetime(2024, 11, 2, tzinfo=UTC),
    }

    first = asyncio.run(service.ingest_source(**kwargs))
    replay = asyncio.run(service.ingest_source(**kwargs))

    assert first.status == "processed"
    assert first.normalized_period_count == 1
    document = service._parser.document
    assert document is not None
    assert document.tables
    margin = next(cell for cell in document.tables[0].cells if cell.footnote_ids)
    assert margin.normalized_value == "18.4%"
    assert margin.period_label == "Q3 2024"
    assert document.footnotes[0].linked_cell_ids == [margin.cell_id]
    assert replay.status == "replayed"
    assert replay.audit_fingerprint == first.audit_fingerprint
    assert service.audit_store.count() == 1


class RecordingParser:
    """Capture the canonical output while preserving the real parser boundary."""

    def __init__(self, parser: DoclingLayoutParser) -> None:
        self._parser = parser
        self.document = None

    def parse(self, content: bytes, options: ParseOptions):
        self.document = self._parser.parse(content, options)
        return self.document
