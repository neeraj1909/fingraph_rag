"""Vendor-neutral graph storage ports."""

from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class GraphStore(Protocol):
    """Read/write boundary for the temporal knowledge graph."""

    async def execute_read(
        self, template_id: str, parameters: Mapping[str, Any]
    ) -> Sequence[Mapping[str, Any]]: ...

    async def upsert_entities(self, entities: Sequence[Mapping[str, Any]]) -> int: ...

    async def upsert_relationships(self, relationships: Sequence[Mapping[str, Any]]) -> int: ...
