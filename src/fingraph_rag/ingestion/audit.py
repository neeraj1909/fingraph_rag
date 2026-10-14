"""Idempotent ingestion audit records and in-memory test store."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Protocol

from pydantic import Field

from fingraph_rag.domain.contracts import ContractModel, TemporalContext

from .contracts import EntityResolutionBatch, QuarantineRecord


class IngestionRunReport(ContractModel):
    ingestion_id: str = Field(min_length=1)
    audit_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    artifact_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    status: Literal["processed", "quarantined", "replayed"]
    replayed: bool = False
    resolved_entity_count: int = Field(ge=0)
    normalized_period_count: int = Field(ge=0)
    quarantined_count: int = Field(ge=0)
    resolution: EntityResolutionBatch
    normalized_periods: list[TemporalContext] = Field(default_factory=list)
    quarantined: list[QuarantineRecord] = Field(default_factory=list)
    created_at: datetime

    @property
    def is_idempotent_replay(self) -> bool:
        return self.replayed


class IngestionAuditStore(Protocol):
    def get(self, audit_fingerprint: str) -> IngestionRunReport | None: ...

    def save(self, report: IngestionRunReport) -> None: ...


class InMemoryIngestionAuditStore:
    """Deterministic store used until PostgreSQL audit persistence is added."""

    def __init__(self) -> None:
        self._reports: dict[str, IngestionRunReport] = {}

    def get(self, audit_fingerprint: str) -> IngestionRunReport | None:
        return self._reports.get(audit_fingerprint)

    def save(self, report: IngestionRunReport) -> None:
        existing = self._reports.get(report.audit_fingerprint)
        if existing is not None:
            raise ValueError("an ingestion audit fingerprint cannot be written twice")
        self._reports[report.audit_fingerprint] = report

    def count(self) -> int:
        return len(self._reports)
