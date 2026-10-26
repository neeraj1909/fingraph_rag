import asyncio
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from fingraph_rag.application.ingest_service import IngestionService
from fingraph_rag.domain.contracts import CanonicalEntity, EntityMention, EvidenceLocator
from fingraph_rag.ingestion.audit import InMemoryIngestionAuditStore
from fingraph_rag.ingestion.entity_resolution import EntityIndex, EntityResolver
from fingraph_rag.ingestion.fetch import FixtureSourceFetcher
from fingraph_rag.ingestion.identity import content_sha256
from fingraph_rag.ingestion.parsers import DoclingLayoutParser, ParseOptions, SyntheticLayoutParser
from fingraph_rag.ingestion.sources import InMemorySourceRegistry, SourceDescriptor
from fingraph_rag.ingestion.temporal_normalization import FiscalCalendar, RawPeriod

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "filings"


@pytest.mark.integration
def test_fixture_ingestion_is_idempotent_and_quarantines_unsafe_items() -> None:
    locator = EvidenceLocator(
        doc_id="doc_synthetic_acme_2024_10k",
        filing_date=date(2025, 2, 28),
        section="Item 7",
        page=1,
        quote_hash="1" * 16,
    )
    known_entity = CanonicalEntity(
        entity_id="company_synthetic_acme",
        canonical_name="Synthetic Acme Corporation",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        merged_surface_forms=["Acme"],
        resolution_method="identifier",
        confidence=1.0,
    )
    mentions = [
        EntityMention(
            surface_form="Acme",
            entity_type="Company",
            identifiers={"CIK": "0000001"},
            locator=locator,
            confidence=0.9,
        ),
        EntityMention(
            surface_form="Unknown Supplier",
            entity_type="Supplier",
            locator=locator,
            confidence=0.8,
        ),
    ]
    audit_store = InMemoryIngestionAuditStore()
    service = IngestionService(
        resolver=EntityResolver(EntityIndex([known_entity])),
        audit_store=audit_store,
    )
    options = {
        "raw_path": FIXTURE_DIR / "synthetic_acme_10k.txt",
        "canonical_path": FIXTURE_DIR / "synthetic_acme_10k.json",
        "mentions": mentions,
        "periods": [
            RawPeriod(label="FY2024"),
            RawPeriod(label="FY2024", restatement_version=1),
            RawPeriod(label="Q3 2024"),
            RawPeriod(label="Q5 2024"),
        ],
        "calendar": FiscalCalendar(
            issuer_id="company_synthetic_acme",
            fiscal_year_end_month=12,
            fiscal_year_end_day=31,
        ),
        "acquired_at": datetime(2025, 2, 28, 12, 0, tzinfo=UTC),
        "now": datetime(2025, 3, 1, tzinfo=UTC),
    }

    first = service.ingest_fixture(**options)
    replay = service.ingest_fixture(**options)

    assert first.status == "quarantined"
    assert first.replayed is False
    assert first.resolved_entity_count == 1
    assert first.normalized_period_count == 3
    assert first.quarantined_count == 2
    assert [period.fiscal_quarter for period in first.normalized_periods] == ["FY", "FY", "Q3"]
    assert [period.restatement_version for period in first.normalized_periods] == [0, 1, 0]
    assert first.normalized_periods[-1].period_end == date(2024, 9, 30)
    assert replay.status == "replayed"
    assert replay.replayed is True
    assert replay.audit_fingerprint == first.audit_fingerprint
    assert audit_store.count() == 1


@pytest.mark.integration
def test_multi_page_quarterly_fixture_replays_without_losing_period_or_locator() -> None:
    locator = EvidenceLocator(
        doc_id="doc_synthetic_acme_2024_q3_10q",
        filing_date=date(2024, 11, 1),
        section="Item 2",
        page=1,
        quote_hash="2" * 16,
    )
    known_entity = CanonicalEntity(
        entity_id="company_synthetic_acme",
        canonical_name="Synthetic Acme Corporation",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        merged_surface_forms=["Acme"],
        resolution_method="identifier",
        confidence=1.0,
    )
    service = IngestionService(
        resolver=EntityResolver(EntityIndex([known_entity])),
        audit_store=InMemoryIngestionAuditStore(),
    )
    options = {
        "raw_path": FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt",
        "canonical_path": FIXTURE_DIR / "synthetic_acme_2024_q3_10q.json",
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
        "acquired_at": datetime(2024, 11, 1, 12, 0, tzinfo=UTC),
        "now": datetime(2024, 11, 2, tzinfo=UTC),
    }

    first = service.ingest_fixture(**options)
    replay = service.ingest_fixture(**options)

    assert first.status == "processed"
    assert first.normalized_period_count == 1
    assert first.normalized_periods[0].period_end == date(2024, 9, 30)
    assert replay.status == "replayed"
    assert replay.audit_fingerprint == first.audit_fingerprint
    assert service.audit_store.count() == 1


@pytest.mark.integration
def test_source_registry_and_fetcher_feed_the_ingestion_boundary() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt"
    descriptor = SourceDescriptor(
        source_id="synthetic-acme-2024-q3-10q",
        source_uri=f"fixture://{raw_path.name}",
        media_type="text/plain",
        license_reference="synthetic-fixture",
        expected_content_sha256=content_sha256(raw_path.read_bytes()),
    )
    registry = InMemorySourceRegistry([descriptor])
    fetcher = FixtureSourceFetcher(FIXTURE_DIR)
    known_entity = CanonicalEntity(
        entity_id="company_synthetic_acme",
        canonical_name="Synthetic Acme Corporation",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        merged_surface_forms=["Acme"],
        resolution_method="identifier",
        confidence=1.0,
    )
    service = IngestionService(
        resolver=EntityResolver(EntityIndex([known_entity])),
        audit_store=InMemoryIngestionAuditStore(),
    )
    mention_locator = EvidenceLocator(
        doc_id="doc_synthetic_acme_2024_q3_10q",
        filing_date=date(2024, 11, 1),
        section="Item 2",
        page=1,
        quote_hash="3" * 16,
    )

    async def run() -> tuple[object, object]:
        kwargs = {
            "source_id": descriptor.source_id,
            "source_registry": registry,
            "source_fetcher": fetcher,
            "canonical_path": FIXTURE_DIR / "synthetic_acme_2024_q3_10q.json",
            "mentions": [
                EntityMention(
                    surface_form="Acme",
                    entity_type="Company",
                    identifiers={"CIK": "0000001"},
                    locator=mention_locator,
                    confidence=0.9,
                )
            ],
            "periods": [RawPeriod(label="Q3 2024")],
            "calendar": FiscalCalendar(
                issuer_id="company_synthetic_acme",
                fiscal_year_end_month=12,
                fiscal_year_end_day=31,
            ),
            "now": datetime(2024, 11, 2, tzinfo=UTC),
        }
        first = await service.ingest_source(**kwargs)
        replay = await service.ingest_source(**kwargs)
        return first, replay

    first, replay = asyncio.run(run())

    assert first.status == "processed"
    assert first.artifact_id == "artifact_9a3be594cb750099e7166eb13b548fe0"
    assert replay.status == "replayed"
    assert service.audit_store.count() == 1


@pytest.mark.integration
def test_source_boundary_can_parse_fetched_raw_bytes_without_canonical_sidecar() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt"
    descriptor = SourceDescriptor(
        source_id="synthetic-acme-2024-q3-10q-raw",
        source_uri=f"fixture://{raw_path.name}",
        media_type="text/plain",
        license_reference="synthetic-fixture",
        expected_content_sha256=content_sha256(raw_path.read_bytes()),
    )
    registry = InMemorySourceRegistry([descriptor])
    fetcher = FixtureSourceFetcher(FIXTURE_DIR)
    known_entity = CanonicalEntity(
        entity_id="company_synthetic_acme",
        canonical_name="Synthetic Acme Corporation",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        merged_surface_forms=["Acme"],
        resolution_method="identifier",
        confidence=1.0,
    )
    service = IngestionService(
        resolver=EntityResolver(EntityIndex([known_entity])),
        audit_store=InMemoryIngestionAuditStore(),
        parser=SyntheticLayoutParser(),
    )
    parse_options = ParseOptions(
        parser_version="synthetic-layout-1",
        expected_artifact_id="bound-at-runtime",
        expected_source_content_sha256="0" * 64,
        source_uri=descriptor.source_uri,
        document_id="doc_synthetic_acme_2024_q3_10q_raw_service",
        issuer_id="company_synthetic_acme",
        document_type="sec_10q",
        filing_date=date(2024, 11, 1),
        title="Synthetic Acme Corporation Form 10-Q Fixture",
    )
    mention_locator = EvidenceLocator(
        doc_id=parse_options.document_id,
        filing_date=date(2024, 11, 1),
        section="Item 2",
        page=1,
        quote_hash="4" * 16,
    )

    async def run() -> tuple[object, object]:
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
                    locator=mention_locator,
                    confidence=0.9,
                )
            ],
            "periods": [RawPeriod(label="Q3 2024")],
            "calendar": FiscalCalendar(
                issuer_id="company_synthetic_acme",
                fiscal_year_end_month=12,
                fiscal_year_end_day=31,
            ),
            "parser_version": "synthetic-layout-1",
            "now": datetime(2024, 11, 2, tzinfo=UTC),
        }
        first = await service.ingest_source(**kwargs)
        replay = await service.ingest_source(**kwargs)
        return first, replay

    first, replay = asyncio.run(run())

    assert first.status == "processed"
    assert first.document_id == parse_options.document_id
    assert replay.status == "replayed"
    assert service.audit_store.count() == 1


@pytest.mark.integration
def test_docling_adapter_replays_through_source_ingestion_boundary() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt"
    content = raw_path.read_bytes()
    descriptor = SourceDescriptor(
        source_id="synthetic-acme-2024-q3-10q-docling",
        source_uri=f"fixture://{raw_path.name}",
        media_type="text/plain",
        license_reference="synthetic-fixture",
        expected_content_sha256=content_sha256(content),
    )
    registry = InMemorySourceRegistry([descriptor])
    fetcher = FixtureSourceFetcher(FIXTURE_DIR)
    provenance = [SimpleNamespace(page_no=1)]
    fake_document = SimpleNamespace(
        name="synthetic_acme_2024_q3_10q.pdf",
        texts=[
            SimpleNamespace(
                text="Synthetic Acme Corporation Form 10-Q Fixture",
                label="title",
                prov=provenance,
            ),
            SimpleNamespace(
                text="Item 2. Management's Discussion and Analysis",
                label="section_header",
                prov=provenance,
            ),
            SimpleNamespace(
                text="Quarterly operations were affected by supplier concentration.",
                label="text",
                prov=provenance,
            ),
        ],
        tables=[
            SimpleNamespace(
                caption="Quarterly Operations",
                prov=provenance,
                data=SimpleNamespace(
                    grid=[
                        [
                            SimpleNamespace(text="Metric", prov=provenance),
                            SimpleNamespace(text="Q3 2024", prov=provenance),
                        ],
                        [
                            SimpleNamespace(text="Operating margin", prov=provenance),
                            SimpleNamespace(text="18.4%", prov=provenance),
                        ],
                    ]
                ),
            )
        ],
    )

    class FakeConverter:
        def convert(self, source: object) -> object:
            assert source == content
            return SimpleNamespace(document=fake_document)

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
        parser=DoclingLayoutParser(FakeConverter(), source_factory=lambda raw, _: raw),
    )
    parse_options = ParseOptions(
        parser_version="docling-v2-adapter-1",
        expected_artifact_id="bound-at-runtime",
        expected_source_content_sha256="0" * 64,
        source_uri=descriptor.source_uri,
        document_id="doc_synthetic_acme_2024_q3_10q_docling_service",
        issuer_id="company_synthetic_acme",
        document_type="sec_10q",
        filing_date=date(2024, 11, 1),
        title="Synthetic Acme Corporation Form 10-Q Fixture",
    )
    locator = EvidenceLocator(
        doc_id=parse_options.document_id,
        filing_date=date(2024, 11, 1),
        section="Item 2",
        page=1,
        quote_hash="5" * 16,
    )

    async def run() -> tuple[object, object]:
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
            "parser_version": "docling-v2-adapter-1",
            "now": datetime(2024, 11, 2, tzinfo=UTC),
        }
        first = await service.ingest_source(**kwargs)
        replay = await service.ingest_source(**kwargs)
        return first, replay

    first, replay = asyncio.run(run())

    assert first.status == "processed"
    assert first.document_id == parse_options.document_id
    assert replay.status == "replayed"
    assert service.audit_store.count() == 1
