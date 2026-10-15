"""Provider-neutral source registry contracts for ingestion."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol, runtime_checkable

from pydantic import Field

from fingraph_rag.domain.contracts import ContractModel


class SourceDescriptor(ContractModel):
    """Safe source metadata used before any bytes are acquired."""

    source_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    license_reference: str = Field(min_length=1)
    expected_content_sha256: str | None = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    metadata: dict[str, str] = Field(default_factory=dict)


@runtime_checkable
class SourceRegistry(Protocol):
    """Resolve a stable source ID without acquiring or mutating source data."""

    async def resolve(self, source_id: str) -> SourceDescriptor: ...


class InMemorySourceRegistry:
    """Deterministic registry for fixtures and composition-root tests."""

    def __init__(self, descriptors: Iterable[SourceDescriptor]) -> None:
        self._descriptors: dict[str, SourceDescriptor] = {}
        for descriptor in descriptors:
            if descriptor.source_id in self._descriptors:
                raise ValueError(f"duplicate source ID: {descriptor.source_id}")
            self._descriptors[descriptor.source_id] = descriptor.model_copy(deep=True)

    async def resolve(self, source_id: str) -> SourceDescriptor:
        try:
            return self._descriptors[source_id].model_copy(deep=True)
        except KeyError as exc:
            raise ValueError(f"unknown source ID: {source_id}") from exc
