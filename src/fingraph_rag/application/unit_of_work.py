"""Unit-of-work boundary shared by application services."""

from types import TracebackType
from typing import Protocol, Self


class UnitOfWork(Protocol):
    """Transaction boundary without coupling the application to a driver."""

    async def __aenter__(self) -> Self: ...

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None: ...

    async def commit(self) -> None: ...

    async def rollback(self) -> None: ...


class NoOpUnitOfWork:
    """In-memory implementation useful for foundation and unit tests."""

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if exc_type is not None:
            await self.rollback()

    async def commit(self) -> None:
        return None

    async def rollback(self) -> None:
        return None
