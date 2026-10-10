"""Strict contracts for replayable financial-document ingestion.

The models preserve source locators instead of treating normalized text as the
source of truth. Tables remain atomic at cell level, and chunks retain explicit
references to their originating sections, tables, cells, or footnotes.
"""

from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import Field, model_validator

from fingraph_rag.domain.contracts import (
    ContractModel,
    EntityResolutionDecision,
    ExtractionEnvelope,
)

from .identity import content_sha256, stable_id


class SourceLocator(ContractModel):
    """A truthful page, character, table, cell, or footnote locator."""

    kind: Literal["document", "section", "table", "table_cell", "footnote", "character_span"]
    page: int | None = Field(default=None, ge=1)
    char_start: int | None = Field(default=None, ge=0)
    char_end: int | None = Field(default=None, ge=0)
    table_id: str | None = None
    cell_id: str | None = None
    footnote_id: str | None = None

    @model_validator(mode="after")
    def validate_locator_shape(self) -> SourceLocator:
        if (
            self.char_start is not None
            and self.char_end is not None
            and self.char_end < self.char_start
        ):
            raise ValueError("char_end must be greater than or equal to char_start")
        if self.kind == "table_cell" and not (self.table_id and self.cell_id):
            raise ValueError("table_cell locators require table_id and cell_id")
        if self.kind == "table" and not self.table_id:
            raise ValueError("table locators require table_id")
        if self.kind == "footnote" and not self.footnote_id:
            raise ValueError("footnote locators require footnote_id")
        if (
            self.kind in {"section", "character_span"}
            and self.page is None
            and self.char_start is None
        ):
            raise ValueError("section/character_span locators require a page or character offset")
        return self


class RawArtifact(ContractModel):
    """Immutable acquired bytes and their replay identity."""

    artifact_id: str = Field(min_length=1)
    source_uri: str = Field(min_length=1)
    media_type: str = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    byte_size: int = Field(ge=1)
    acquired_at: datetime
    license_reference: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @classmethod
    def from_bytes(
        cls,
        content: bytes,
        *,
        source_uri: str,
        media_type: str,
        acquired_at: datetime,
        license_reference: str | None = None,
        metadata: dict[str, str] | None = None,
    ) -> RawArtifact:
        digest = content_sha256(content)
        return cls(
            artifact_id=stable_id("artifact", (digest, media_type)),
            source_uri=source_uri,
            media_type=media_type,
            content_sha256=digest,
            byte_size=len(content),
            acquired_at=acquired_at,
            license_reference=license_reference,
            metadata=metadata or {},
        )

    @model_validator(mode="after")
    def require_timezone(self) -> RawArtifact:
        if self.acquired_at.tzinfo is None:
            raise ValueError("acquired_at must include timezone information")
        return self


class DocumentSection(ContractModel):
    section_id: str = Field(min_length=1)
    heading: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    level: int = Field(ge=1)
    text: str = Field(min_length=1)
    locator: SourceLocator


class TableCell(ContractModel):
    cell_id: str = Field(min_length=1)
    table_id: str = Field(min_length=1)
    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    raw_value: str
    normalized_value: str | None = None
    cell_type: Literal["header", "label", "value", "blank", "footnote_marker"]
    period_label: str | None = None
    unit: str | None = None
    footnote_ids: list[str] = Field(default_factory=list)
    locator: SourceLocator

    @model_validator(mode="after")
    def require_cell_locator(self) -> TableCell:
        if self.locator.kind != "table_cell" or self.locator.table_id != self.table_id:
            raise ValueError("table cells require a matching table_cell locator")
        if self.locator.cell_id != self.cell_id:
            raise ValueError("cell locator must identify the cell itself")
        return self


class FinancialTable(ContractModel):
    table_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    cells: list[TableCell] = Field(min_length=1)
    footnote_ids: list[str] = Field(default_factory=list)
    locator: SourceLocator

    @model_validator(mode="after")
    def validate_cells(self) -> FinancialTable:
        if self.locator.kind != "table" or self.locator.table_id != self.table_id:
            raise ValueError("tables require a matching table locator")
        cell_ids = [cell.cell_id for cell in self.cells]
        if len(cell_ids) != len(set(cell_ids)):
            raise ValueError("table cell IDs must be unique")
        if any(cell.table_id != self.table_id for cell in self.cells):
            raise ValueError("every table cell must belong to its parent table")
        return self


class Footnote(ContractModel):
    footnote_id: str = Field(min_length=1)
    marker: str = Field(min_length=1)
    text: str = Field(min_length=1)
    linked_cell_ids: list[str] = Field(default_factory=list)
    locator: SourceLocator

    @model_validator(mode="after")
    def require_footnote_locator(self) -> Footnote:
        if self.locator.kind != "footnote" or self.locator.footnote_id != self.footnote_id:
            raise ValueError("footnotes require a matching footnote locator")
        return self


class SourceReference(ContractModel):
    kind: Literal["section", "table", "table_cell", "footnote"]
    source_id: str = Field(min_length=1)


class DocumentChunk(ContractModel):
    chunk_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    family: Literal["narrative", "table_summary"]
    text: str = Field(min_length=1)
    chunking_policy: str = Field(min_length=1)
    source_references: list[SourceReference] = Field(min_length=1)
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @classmethod
    def build(
        cls,
        *,
        document_id: str,
        ordinal: int,
        family: Literal["narrative", "table_summary"],
        text: str,
        chunking_policy: str,
        source_references: list[SourceReference],
    ) -> DocumentChunk:
        digest = content_sha256(text.encode("utf-8"))
        references = ",".join(f"{item.kind}:{item.source_id}" for item in source_references)
        chunk_id = stable_id(
            "chunk",
            (document_id, str(ordinal), family, chunking_policy, digest, references),
        )
        return cls(
            chunk_id=chunk_id,
            document_id=document_id,
            ordinal=ordinal,
            family=family,
            text=text,
            chunking_policy=chunking_policy,
            source_references=source_references,
            content_sha256=digest,
        )

    @model_validator(mode="after")
    def require_table_source_for_table_summary(self) -> DocumentChunk:
        if self.family == "table_summary" and not any(
            reference.kind in {"table", "table_cell"} for reference in self.source_references
        ):
            raise ValueError("table_summary chunks require a table or table_cell source")
        return self


class CanonicalDocument(ContractModel):
    document_id: str = Field(min_length=1)
    artifact_id: str = Field(min_length=1)
    document_type: Literal["sec_10k", "sec_10q", "earnings_transcript", "disclosure", "other"]
    issuer_id: str = Field(min_length=1)
    accession_number: str | None = None
    filing_date: date
    title: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    sections: list[DocumentSection] = Field(default_factory=list)
    tables: list[FinancialTable] = Field(default_factory=list)
    footnotes: list[Footnote] = Field(default_factory=list)
    chunks: list[DocumentChunk] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_provenance_graph(self) -> CanonicalDocument:
        section_ids = {section.section_id for section in self.sections}
        table_ids = {table.table_id for table in self.tables}
        cell_ids = {cell.cell_id for table in self.tables for cell in table.cells}
        footnote_ids = {footnote.footnote_id for footnote in self.footnotes}

        if len(section_ids) != len(self.sections):
            raise ValueError("section IDs must be unique")
        if len(table_ids) != len(self.tables):
            raise ValueError("table IDs must be unique")
        if len(footnote_ids) != len(self.footnotes):
            raise ValueError("footnote IDs must be unique")

        known_sources = {
            "section": section_ids,
            "table": table_ids,
            "table_cell": cell_ids,
            "footnote": footnote_ids,
        }
        for chunk in self.chunks:
            if chunk.document_id != self.document_id:
                raise ValueError("chunk document_id must match its canonical document")
            for reference in chunk.source_references:
                if reference.source_id not in known_sources[reference.kind]:
                    raise ValueError(
                        f"chunk references unknown {reference.kind}: {reference.source_id}"
                    )

        for footnote in self.footnotes:
            if any(cell_id not in cell_ids for cell_id in footnote.linked_cell_ids):
                raise ValueError("footnote links must resolve to known table cells")
        for table in self.tables:
            if any(footnote_id not in footnote_ids for footnote_id in table.footnote_ids):
                raise ValueError("table footnote links must resolve to known footnotes")
            for cell in table.cells:
                if any(footnote_id not in footnote_ids for footnote_id in cell.footnote_ids):
                    raise ValueError("cell footnote links must resolve to known footnotes")
        return self


class ExtractionRun(ContractModel):
    run_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    parser_version: str = Field(min_length=1)
    extractor_version: str = Field(min_length=1)
    ontology_version: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    input_fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime

    @model_validator(mode="after")
    def require_timezone(self) -> ExtractionRun:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return self


class ValidationNotice(ContractModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    source_reference: SourceReference | None = None


class QuarantineRecord(ContractModel):
    """A safe, replayable record for an item that cannot be trusted yet."""

    quarantine_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    run_id: str = Field(min_length=1)
    stage: Literal["entity_resolution", "temporal_normalization", "extraction"]
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_reference: SourceReference | None = None
    created_at: datetime

    @model_validator(mode="after")
    def require_timezone(self) -> QuarantineRecord:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must include timezone information")
        return self


class ExtractionRecord(ContractModel):
    run: ExtractionRun
    envelope: ExtractionEnvelope
    notices: list[ValidationNotice] = Field(default_factory=list)

    @model_validator(mode="after")
    def match_run_and_document(self) -> ExtractionRecord:
        if self.run.document_id != self.envelope.document_id:
            raise ValueError("extraction run and envelope document IDs must match")
        if self.run.run_id != self.envelope.extraction_run_id:
            raise ValueError("extraction run and envelope run IDs must match")
        return self


class EntityResolutionBatch(ContractModel):
    run_id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    decisions: list[EntityResolutionDecision] = Field(default_factory=list)
    unresolved_surface_forms: list[str] = Field(default_factory=list)
    policy_version: str = Field(min_length=1)


def utc_now() -> datetime:
    """Return an explicit UTC timestamp for composition roots and fixtures."""

    return datetime.now(UTC)
