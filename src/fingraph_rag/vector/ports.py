"""Vendor-neutral vector and sparse retrieval ports."""

from collections.abc import Sequence
from typing import Protocol, runtime_checkable


@runtime_checkable
class VectorStore(Protocol):
    """Boundary for dense, sparse, and hybrid evidence retrieval."""

    async def upsert(self, records: Sequence[dict[str, object]]) -> int: ...

    async def search(self, query: str, limit: int) -> Sequence[dict[str, object]]: ...
