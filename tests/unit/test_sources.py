import asyncio
from datetime import UTC
from pathlib import Path

import pytest

from fingraph_rag.ingestion.fetch import FixtureSourceFetcher
from fingraph_rag.ingestion.identity import content_sha256
from fingraph_rag.ingestion.sources import InMemorySourceRegistry, SourceDescriptor

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "filings"


def descriptor_for(filename: str) -> SourceDescriptor:
    content = (FIXTURE_DIR / filename).read_bytes()
    return SourceDescriptor(
        source_id=filename,
        source_uri=f"fixture://{filename}",
        media_type="text/plain",
        license_reference="synthetic-fixture",
        expected_content_sha256=content_sha256(content),
        metadata={"fixture": "true"},
    )


@pytest.mark.unit
def test_fixture_source_fetch_round_trips_immutable_artifact_identity() -> None:
    descriptor = descriptor_for("synthetic_acme_2024_q3_10q.txt")
    registry = InMemorySourceRegistry([descriptor])
    fetcher = FixtureSourceFetcher(FIXTURE_DIR)

    resolved = asyncio.run(registry.resolve(descriptor.source_id))
    fetched = asyncio.run(fetcher.fetch(resolved))

    assert fetched.source == descriptor
    assert fetched.artifact.source_uri == descriptor.source_uri
    assert fetched.artifact.content_sha256 == descriptor.expected_content_sha256
    assert fetched.artifact.byte_size == len(fetched.content)
    assert fetched.artifact.acquired_at.tzinfo == UTC


@pytest.mark.unit
def test_source_registry_rejects_unknown_ids_and_duplicate_registration() -> None:
    descriptor = descriptor_for("synthetic_acme_10k.txt")
    with pytest.raises(ValueError, match="duplicate source ID"):
        InMemorySourceRegistry([descriptor, descriptor])

    registry = InMemorySourceRegistry([descriptor])
    with pytest.raises(ValueError, match="unknown source ID"):
        asyncio.run(registry.resolve("missing"))


@pytest.mark.unit
def test_fixture_fetcher_rejects_unsafe_or_unexpected_source() -> None:
    fetcher = FixtureSourceFetcher(FIXTURE_DIR)
    traversal = SourceDescriptor(
        source_id="escape",
        source_uri="fixture://../outside.txt",
        media_type="text/plain",
        license_reference="synthetic-fixture",
    )
    wrong_hash = descriptor_for("synthetic_acme_10k.txt").model_copy(
        update={"expected_content_sha256": "0" * 64}
    )

    with pytest.raises(ValueError, match="safe relative path"):
        asyncio.run(fetcher.fetch(traversal))
    with pytest.raises(ValueError, match="source expectation"):
        asyncio.run(fetcher.fetch(wrong_hash))
