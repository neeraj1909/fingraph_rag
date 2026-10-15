"""Source acquisition ports and a safe local fixture adapter."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import Field, model_validator

from fingraph_rag.domain.contracts import ContractModel

from .contracts import RawArtifact
from .identity import content_sha256
from .sources import SourceDescriptor


class FetchedArtifact(ContractModel):
    """Acquired bytes paired with the immutable artifact metadata."""

    source: SourceDescriptor
    artifact: RawArtifact
    content: bytes = Field(min_length=1)

    @model_validator(mode="after")
    def validate_artifact_binding(self) -> FetchedArtifact:
        if self.artifact.source_uri != self.source.source_uri:
            raise ValueError("artifact source URI does not match source descriptor")
        if self.artifact.media_type != self.source.media_type:
            raise ValueError("artifact media type does not match source descriptor")
        if self.artifact.license_reference != self.source.license_reference:
            raise ValueError("artifact license reference does not match source descriptor")
        if self.artifact.content_sha256 != content_sha256(self.content):
            raise ValueError("artifact content hash does not match acquired bytes")
        if self.source.expected_content_sha256 not in {
            None,
            self.artifact.content_sha256,
        }:
            raise ValueError("acquired bytes do not match the source expectation")
        return self


@runtime_checkable
class SourceFetcher(Protocol):
    """Acquire a source through a replaceable, auditable adapter."""

    async def fetch(self, source: SourceDescriptor) -> FetchedArtifact: ...


class FixtureSourceFetcher:
    """Read only registered fixture URIs from a bounded local directory."""

    def __init__(self, root: Path) -> None:
        self._root = root.resolve()

    async def fetch(self, source: SourceDescriptor) -> FetchedArtifact:
        relative_path = self._relative_fixture_path(source.source_uri)
        path = (self._root / relative_path).resolve()
        if not path.is_relative_to(self._root):
            raise ValueError("fixture source escapes the configured root")
        if not path.is_file():
            raise FileNotFoundError(f"fixture source does not exist: {relative_path}")

        content = path.read_bytes()
        artifact = RawArtifact.from_bytes(
            content,
            source_uri=source.source_uri,
            media_type=source.media_type,
            acquired_at=datetime.now(UTC),
            license_reference=source.license_reference,
            metadata=source.metadata,
        )
        return FetchedArtifact(source=source, artifact=artifact, content=content)

    @staticmethod
    def _relative_fixture_path(source_uri: str) -> Path:
        prefix = "fixture://"
        if not source_uri.startswith(prefix):
            raise ValueError("fixture fetcher only accepts fixture:// source URIs")
        relative = source_uri.removeprefix(prefix)
        path = Path(relative)
        if not relative or path.is_absolute() or ".." in path.parts:
            raise ValueError("fixture source URI must be a safe relative path")
        return path
