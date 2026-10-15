"""Application boundary for deterministic fixture ingestion and replay audit."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

from fingraph_rag.domain.contracts import EntityMention
from fingraph_rag.ingestion.audit import (
    IngestionAuditStore,
    IngestionRunReport,
    InMemoryIngestionAuditStore,
)
from fingraph_rag.ingestion.contracts import QuarantineRecord, RawArtifact
from fingraph_rag.ingestion.entity_resolution import EntityResolver
from fingraph_rag.ingestion.fetch import SourceFetcher
from fingraph_rag.ingestion.identity import stable_id
from fingraph_rag.ingestion.parsers import DocumentParser, FixtureDocumentParser, ParseOptions
from fingraph_rag.ingestion.provenance import FootnoteLinker
from fingraph_rag.ingestion.sources import SourceRegistry
from fingraph_rag.ingestion.temporal_normalization import (
    FiscalCalendar,
    RawPeriod,
    TemporalNormalizer,
)


class IngestionService:
    """Orchestrate safe fixture ingestion without mutating graph/vector stores."""

    def __init__(
        self,
        *,
        resolver: EntityResolver,
        audit_store: IngestionAuditStore | None = None,
        temporal_normalizer: TemporalNormalizer | None = None,
        parser: DocumentParser | None = None,
        footnote_linker: FootnoteLinker | None = None,
    ) -> None:
        self._resolver = resolver
        self._audit_store = audit_store or InMemoryIngestionAuditStore()
        self._temporal_normalizer = temporal_normalizer or TemporalNormalizer()
        self._parser = parser or FixtureDocumentParser()
        self._footnote_linker = footnote_linker or FootnoteLinker()

    def ingest_fixture(
        self,
        *,
        raw_path: Path,
        canonical_path: Path,
        mentions: Iterable[EntityMention],
        periods: Iterable[RawPeriod],
        calendar: FiscalCalendar,
        parser_version: str = "fixture-parser-1",
        extractor_version: str = "fixture-extractor-1",
        ontology_version: str = "ontology-1",
        model_id: str = "fixture-model",
        acquired_at: datetime | None = None,
        now: datetime | None = None,
    ) -> IngestionRunReport:
        raw_content = raw_path.read_bytes()
        artifact = RawArtifact.from_bytes(
            raw_content,
            source_uri=f"fixture://{raw_path.name}",
            media_type="text/plain",
            acquired_at=acquired_at or datetime.now(UTC),
            license_reference="synthetic-fixture",
        )
        return self._ingest_artifact(
            artifact=artifact,
            parse_content=canonical_path.read_bytes(),
            mentions=mentions,
            periods=periods,
            calendar=calendar,
            parser_version=parser_version,
            extractor_version=extractor_version,
            ontology_version=ontology_version,
            model_id=model_id,
            now=now,
        )

    async def ingest_source(
        self,
        *,
        source_id: str,
        source_registry: SourceRegistry,
        source_fetcher: SourceFetcher,
        canonical_path: Path | None = None,
        parse_options: ParseOptions | None = None,
        mentions: Iterable[EntityMention],
        periods: Iterable[RawPeriod],
        calendar: FiscalCalendar,
        parser_version: str = "fixture-parser-1",
        extractor_version: str = "fixture-extractor-1",
        ontology_version: str = "ontology-1",
        model_id: str = "fixture-model",
        now: datetime | None = None,
    ) -> IngestionRunReport:
        descriptor = await source_registry.resolve(source_id)
        fetched = await source_fetcher.fetch(descriptor)
        if parse_options is None:
            if canonical_path is None:
                raise ValueError("canonical_path is required when parse_options is omitted")
            parse_content = canonical_path.read_bytes()
        else:
            parse_content = fetched.content
        return self._ingest_artifact(
            artifact=fetched.artifact,
            parse_content=parse_content,
            parse_options=parse_options,
            mentions=mentions,
            periods=periods,
            calendar=calendar,
            parser_version=parser_version,
            extractor_version=extractor_version,
            ontology_version=ontology_version,
            model_id=model_id,
            now=now,
        )

    def _ingest_artifact(
        self,
        *,
        artifact: RawArtifact,
        parse_content: bytes,
        parse_options: ParseOptions | None = None,
        mentions: Iterable[EntityMention],
        periods: Iterable[RawPeriod],
        calendar: FiscalCalendar,
        parser_version: str,
        extractor_version: str,
        ontology_version: str,
        model_id: str,
        now: datetime | None,
    ) -> IngestionRunReport:
        if parse_options is not None and parse_options.parser_version != parser_version:
            raise ValueError("parse options parser version does not match ingestion parser version")
        if parse_options is None:
            bound_options = ParseOptions(
                parser_version=parser_version,
                expected_artifact_id=artifact.artifact_id,
                expected_source_content_sha256=artifact.content_sha256,
            )
        else:
            bound_options = parse_options.model_copy(
                update={
                    "expected_artifact_id": artifact.artifact_id,
                    "expected_source_content_sha256": artifact.content_sha256,
                }
            )
        document = self._parser.parse(parse_content, bound_options)
        document = self._footnote_linker.link(document)

        mention_list = list(mentions)
        period_list = list(periods)
        audit_fingerprint = self._audit_fingerprint(
            artifact_hash=artifact.content_sha256,
            parser_version=parser_version,
            extractor_version=extractor_version,
            ontology_version=ontology_version,
            model_id=model_id,
            resolver_policy=self._resolver.policy_version,
            calendar=calendar,
        )
        existing = self._audit_store.get(audit_fingerprint)
        if existing is not None:
            return existing.model_copy(update={"status": "replayed", "replayed": True})

        timestamp = now or datetime.now(UTC)
        run_id = stable_id("run", (audit_fingerprint,))
        resolution = self._resolver.resolve(
            document_id=document.document_id,
            run_id=run_id,
            mentions=mention_list,
            now=timestamp,
        )
        normalized_periods = []
        quarantined = list(resolution.quarantined)
        for period in period_list:
            try:
                normalized_periods.append(
                    self._temporal_normalizer.normalize(
                        period,
                        calendar=calendar,
                        filing_date=document.filing_date,
                    )
                )
            except ValueError as exc:
                quarantined.append(
                    QuarantineRecord(
                        quarantine_id=stable_id(
                            "quarantine", (run_id, "temporal_normalization", period.label)
                        ),
                        document_id=document.document_id,
                        run_id=run_id,
                        stage="temporal_normalization",
                        code="invalid_period",
                        message=str(exc),
                        subject=period.label,
                        payload_hash=hashlib.sha256(period.model_dump_json().encode()).hexdigest(),
                        created_at=timestamp,
                    )
                )

        report = IngestionRunReport(
            ingestion_id=stable_id("ingestion", (audit_fingerprint,)),
            audit_fingerprint=audit_fingerprint,
            artifact_id=artifact.artifact_id,
            document_id=document.document_id,
            run_id=run_id,
            status="quarantined" if quarantined else "processed",
            resolved_entity_count=sum(
                decision.selected_entity_id is not None for decision in resolution.batch.decisions
            ),
            normalized_period_count=len(normalized_periods),
            quarantined_count=len(quarantined),
            resolution=resolution.batch,
            normalized_periods=normalized_periods,
            quarantined=quarantined,
            created_at=timestamp,
        )
        self._audit_store.save(report)
        return report

    @property
    def audit_store(self) -> IngestionAuditStore:
        return self._audit_store

    def _audit_fingerprint(
        self,
        *,
        artifact_hash: str,
        parser_version: str,
        extractor_version: str,
        ontology_version: str,
        model_id: str,
        resolver_policy: str,
        calendar: FiscalCalendar,
    ) -> str:
        payload = "|".join(
            (
                artifact_hash,
                parser_version,
                extractor_version,
                ontology_version,
                model_id,
                resolver_policy,
                calendar.model_dump_json(),
            )
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()
