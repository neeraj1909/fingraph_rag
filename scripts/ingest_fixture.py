#!/usr/bin/env python3
"""Run the deterministic synthetic ingestion boundary."""

from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, date, datetime
from pathlib import Path

from fingraph_rag.application.ingest_service import IngestionService
from fingraph_rag.domain.contracts import CanonicalEntity, EntityMention, EvidenceLocator
from fingraph_rag.ingestion.audit import InMemoryIngestionAuditStore
from fingraph_rag.ingestion.entity_resolution import EntityIndex, EntityResolver
from fingraph_rag.ingestion.fetch import FixtureSourceFetcher
from fingraph_rag.ingestion.identity import content_sha256
from fingraph_rag.ingestion.sources import InMemorySourceRegistry, SourceDescriptor
from fingraph_rag.ingestion.temporal_normalization import FiscalCalendar, RawPeriod


def build_service() -> IngestionService:
    entity = CanonicalEntity(
        entity_id="company_synthetic_acme",
        canonical_name="Synthetic Acme Corporation",
        entity_type="Company",
        identifiers={"CIK": "0000001"},
        merged_surface_forms=["Acme"],
        resolution_method="identifier",
        confidence=1.0,
    )
    return IngestionService(
        resolver=EntityResolver(EntityIndex([entity])),
        audit_store=InMemoryIngestionAuditStore(),
    )


def parse_period(value: str) -> RawPeriod:
    label, separator, version = value.rpartition(":")
    if not separator:
        return RawPeriod(label=value)
    if not label or not version.isdigit():
        raise argparse.ArgumentTypeError("period must be LABEL or LABEL:RESTATEMENT_VERSION")
    return RawPeriod(label=label, restatement_version=int(version))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_path", type=Path)
    parser.add_argument("--canonical", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--replay-check", action="store_true")
    parser.add_argument(
        "--period",
        action="append",
        type=parse_period,
        help="period label, optionally with a restatement version (for example FY2024:1)",
    )
    args = parser.parse_args()

    service = build_service()
    canonical_metadata = json.loads(args.canonical.read_text())
    document_id = canonical_metadata["document_id"]
    filing_date = date.fromisoformat(canonical_metadata["filing_date"])
    source = SourceDescriptor(
        source_id=args.raw_path.name,
        source_uri=f"fixture://{args.raw_path.name}",
        media_type="text/plain",
        license_reference="synthetic-fixture",
        expected_content_sha256=content_sha256(args.raw_path.read_bytes()),
    )
    source_registry = InMemorySourceRegistry([source])
    source_fetcher = FixtureSourceFetcher(args.raw_path.parent)
    locator = EvidenceLocator(
        doc_id=document_id,
        filing_date=filing_date,
        section="Item 7",
        page=1,
        quote_hash="1" * 16,
    )
    options = {
        "source_id": source.source_id,
        "source_registry": source_registry,
        "source_fetcher": source_fetcher,
        "canonical_path": args.canonical,
        "mentions": [
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
        ],
        "periods": args.period
        if args.period is not None
        else [RawPeriod(label="FY2024"), RawPeriod(label="Q5 2024")],
        "calendar": FiscalCalendar(
            issuer_id="company_synthetic_acme",
            fiscal_year_end_month=12,
            fiscal_year_end_day=31,
        ),
        "now": datetime(2025, 3, 1, tzinfo=UTC),
    }
    first = asyncio.run(service.ingest_source(**options))
    replay = asyncio.run(service.ingest_source(**options)) if args.replay_check else None
    result = {"first": first.model_dump(mode="json"), "dry_run": args.dry_run}
    if replay is not None:
        result["replay"] = replay.model_dump(mode="json")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
