"""Parser ports and the deterministic parser used by synthetic fixtures.

The fixture parser deliberately accepts canonical JSON rather than pretending
that a fixture is a PDF parser. A Docling/Unstructured adapter can implement
the same port later without changing the ingestion service or its contracts.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from datetime import date
from io import BytesIO
from typing import Any, Protocol, runtime_checkable

from pydantic import Field

from fingraph_rag.domain.contracts import ContractModel

from .contracts import (
    CanonicalDocument,
    DocumentChunk,
    DocumentSection,
    FinancialTable,
    Footnote,
    SourceLocator,
    SourceReference,
)
from .identity import content_sha256, stable_id
from .tables import TableItem, TableItemCell, TableNormalizer


class ParseOptions(ContractModel):
    """Input identity and parser policy required for a deterministic parse."""

    parser_version: str = Field(min_length=1)
    expected_artifact_id: str = Field(min_length=1)
    expected_source_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_uri: str | None = None
    document_id: str | None = None
    issuer_id: str | None = None
    document_type: str | None = None
    filing_date: date | None = None
    title: str | None = None
    accession_number: str | None = None


class DocumentParser(Protocol):
    """Provider-neutral parser boundary for raw document bytes."""

    def parse(self, content: bytes, options: ParseOptions) -> CanonicalDocument:
        """Parse bytes into a source-linked canonical document."""


@runtime_checkable
class DoclingConverter(Protocol):
    """Small seam around Docling's ``DocumentConverter``."""

    def convert(self, source: object) -> object:
        """Convert a path or in-memory document stream."""


class FixtureDocumentParser:
    """Validate canonical fixture JSON at the same boundary as real parsers."""

    def parse(self, content: bytes, options: ParseOptions) -> CanonicalDocument:
        document = CanonicalDocument.model_validate_json(content)
        if document.parser_version != options.parser_version:
            raise ValueError("canonical document parser version does not match parse options")
        if document.artifact_id != options.expected_artifact_id:
            raise ValueError("canonical document artifact identity does not match parse options")
        if document.source_content_sha256 != options.expected_source_content_sha256:
            raise ValueError("canonical document source hash does not match raw artifact")
        return document


class DoclingLayoutParser:
    """Adapt an optional Docling converter to the canonical parser contract.

    The adapter imports Docling only when no converter is injected. Tests and
    composition roots can therefore provide a fake or configured converter
    without adding a vendor SDK to the domain or ingestion contracts.
    """

    def __init__(
        self,
        converter: DoclingConverter | None = None,
        *,
        source_factory: Callable[[bytes, ParseOptions], object] | None = None,
    ) -> None:
        self._converter = converter or self._build_default_converter()
        self._source_factory = source_factory or self._build_default_source

    def parse(self, content: bytes, options: ParseOptions) -> CanonicalDocument:
        actual_hash = content_sha256(content)
        if actual_hash != options.expected_source_content_sha256:
            raise ValueError("Docling source content does not match expected artifact hash")
        converted = self._converter.convert(self._source_factory(content, options))
        document = getattr(converted, "document", None)
        if document is None:
            raise ValueError("Docling conversion result does not contain a document")
        return DoclingDocumentMapper().to_canonical(document, options)

    @staticmethod
    def _build_default_converter() -> DoclingConverter:
        try:
            from docling.document_converter import DocumentConverter
        except ImportError as exc:
            raise RuntimeError(
                "DoclingLayoutParser requires the optional 'docling' dependency "
                "or an injected converter"
            ) from exc
        return DocumentConverter()

    @staticmethod
    def _build_default_source(content: bytes, options: ParseOptions) -> object:
        try:
            from docling.datamodel.base_models import DocumentStream
        except ImportError:
            return content
        source_uri = options.source_uri or "document.bin"
        source_name = source_uri.rsplit("/", maxsplit=1)[-1] or "document.bin"
        return DocumentStream(name=source_name, stream=BytesIO(content))


class DoclingDocumentMapper:
    """Map Docling v2's duck-typed document model into canonical contracts."""

    _section_labels = {"section_header", "heading", "header"}
    _period_pattern = re.compile(r"^(?:FY|Q[1-4])?\s*20\d{2}$|^Q[1-4]\s+20\d{2}$")
    _footnote_pattern = re.compile(r"(\[[^\]]+\]|\(\d+\))")

    def to_canonical(self, document: object, options: ParseOptions) -> CanonicalDocument:
        filing_date = options.filing_date
        if filing_date is None:
            raise ValueError("Docling parsing requires filing_date in ParseOptions")
        if not options.issuer_id:
            raise ValueError("Docling parsing requires issuer_id in ParseOptions")

        text_items = list(getattr(document, "texts", []) or [])
        document_id = options.document_id or stable_id(
            "document", (options.source_uri or "docling", options.expected_source_content_sha256)
        )
        title = options.title or self._document_title(document, text_items)
        sections = self._sections(text_items, document_id, title)
        footnotes_by_marker = self._footnotes(text_items, document_id)
        tables = self._tables(document, document_id, footnotes_by_marker)
        footnotes = [
            footnote.model_copy(
                update={
                    "linked_cell_ids": [
                        cell.cell_id
                        for table in tables
                        for cell in table.cells
                        if footnote.footnote_id in cell.footnote_ids
                    ]
                }
            )
            for footnote in footnotes_by_marker.values()
        ]
        chunks = self._chunks(sections, tables, footnotes, document_id)
        return CanonicalDocument(
            document_id=document_id,
            artifact_id=options.expected_artifact_id,
            document_type=options.document_type or "other",
            issuer_id=options.issuer_id,
            accession_number=options.accession_number,
            filing_date=filing_date,
            title=title,
            parser_version=options.parser_version,
            source_content_sha256=options.expected_source_content_sha256,
            sections=sections,
            tables=tables,
            footnotes=footnotes,
            chunks=chunks,
        )

    def _sections(
        self,
        text_items: list[object],
        document_id: str,
        title: str,
    ) -> list[DocumentSection]:
        sections: list[DocumentSection] = []
        current_heading: str | None = None
        current_level = 1
        current_page = 1
        current_text: list[str] = []

        def flush() -> None:
            if current_heading is None:
                return
            text = " ".join(current_text).strip() or current_heading
            section_id = stable_id(
                "section", (document_id, str(len(sections)), current_heading, text)
            )
            sections.append(
                DocumentSection(
                    section_id=section_id,
                    heading=current_heading,
                    ordinal=len(sections),
                    level=current_level,
                    text=text,
                    locator=SourceLocator(kind="section", page=current_page),
                )
            )

        for item in text_items:
            text = self._text(item)
            if not text:
                continue
            label = self._label(item)
            if label == "title":
                continue
            if label in self._section_labels:
                flush()
                current_heading = text
                current_level = int(getattr(item, "level", 1) or 1)
                current_page = self._page(item)
                current_text = []
            else:
                if current_heading is None:
                    current_heading = title
                    current_page = self._page(item)
                current_text.append(text)
        flush()
        if not sections:
            section_id = stable_id("section", (document_id, "0", title, title))
            sections.append(
                DocumentSection(
                    section_id=section_id,
                    heading=title,
                    ordinal=0,
                    level=1,
                    text=title,
                    locator=SourceLocator(kind="section", page=1),
                )
            )
        return sections

    def _footnotes(self, text_items: list[object], document_id: str) -> dict[str, Footnote]:
        footnotes: dict[str, Footnote] = {}
        for item in text_items:
            text = self._text(item)
            match = re.match(r"^(\[[^\]]+\]|\(\d+\))\s+(.+)$", text)
            if not match:
                continue
            marker, footnote_text = match.groups()
            footnote_id = stable_id("footnote", (document_id, marker, footnote_text))
            footnotes[marker] = Footnote(
                footnote_id=footnote_id,
                marker=marker,
                text=footnote_text,
                locator=SourceLocator(
                    kind="footnote", page=self._page(item), footnote_id=footnote_id
                ),
            )
        return footnotes

    def _tables(
        self,
        document: object,
        document_id: str,
        footnotes_by_marker: dict[str, Footnote],
    ) -> list[FinancialTable]:
        tables: list[FinancialTable] = []
        for table_index, item in enumerate(getattr(document, "tables", []) or []):
            title = self._table_title(item, table_index)
            table_id = stable_id("table", (document_id, str(table_index), title))
            grid = getattr(getattr(item, "data", None), "grid", None)
            if not grid:
                raise ValueError(f"Docling table {table_index} has no structured grid")
            period_by_column = {
                column_index: self._period_label(self._text(cell))
                for column_index, cell in enumerate(grid[0])
                if self._period_label(self._text(cell)) is not None
            }
            raw_cells: list[TableItemCell] = []
            for row_index, row in enumerate(grid):
                for column_index, cell in enumerate(row):
                    raw_value = self._text(cell)
                    markers = self._footnote_pattern.findall(raw_value)
                    cell_type = self._cell_type(cell, row_index, column_index, raw_value, markers)
                    raw_cells.append(
                        TableItemCell(
                            row_index=row_index,
                            column_index=column_index,
                            raw_value=raw_value,
                            cell_type=cell_type,
                            page=self._page(cell, self._page(item)),
                            period_label=(
                                self._period_label(raw_value)
                                if row_index == 0
                                else period_by_column.get(column_index)
                            ),
                            unit="%" if "%" in raw_value else None,
                            footnote_ids=[
                                footnotes_by_marker[marker].footnote_id
                                for marker in markers
                                if marker in footnotes_by_marker
                            ],
                        )
                    )
            cells = TableNormalizer().to_cells(
                TableItem(table_id=table_id, title=title, page=self._page(item), cells=raw_cells)
            )
            footnote_ids = sorted(
                {footnote_id for cell in cells for footnote_id in cell.footnote_ids}
            )
            tables.append(
                FinancialTable(
                    table_id=table_id,
                    title=title,
                    cells=cells,
                    footnote_ids=footnote_ids,
                    locator=SourceLocator(kind="table", page=self._page(item), table_id=table_id),
                )
            )
        return tables

    def _chunks(
        self,
        sections: list[DocumentSection],
        tables: list[FinancialTable],
        footnotes: list[Footnote],
        document_id: str,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []
        for section in sections:
            chunks.append(
                DocumentChunk.build(
                    document_id=document_id,
                    ordinal=len(chunks),
                    family="narrative",
                    text=section.text,
                    chunking_policy="docling-section-v1",
                    source_references=[
                        SourceReference(kind="section", source_id=section.section_id)
                    ],
                )
            )
        for table in tables:
            references = [SourceReference(kind="table", source_id=table.table_id)]
            references.extend(
                SourceReference(kind="table_cell", source_id=cell.cell_id) for cell in table.cells
            )
            references.extend(
                SourceReference(kind="footnote", source_id=footnote.footnote_id)
                for footnote in footnotes
                if footnote.footnote_id in table.footnote_ids
            )
            values = [
                f"{cell.period_label}: {cell.normalized_value}"
                for cell in table.cells
                if cell.cell_type in {"value", "footnote_marker"} and cell.period_label
            ]
            chunks.append(
                DocumentChunk.build(
                    document_id=document_id,
                    ordinal=len(chunks),
                    family="table_summary",
                    text=f"{table.title}; " + "; ".join(values),
                    chunking_policy="docling-table-v1",
                    source_references=references,
                )
            )
        return chunks

    @staticmethod
    def _document_title(document: object, text_items: list[object]) -> str:
        name = getattr(document, "name", None)
        if name:
            return str(name)
        for item in text_items:
            if DoclingDocumentMapper._label(item) == "title" and DoclingDocumentMapper._text(item):
                return DoclingDocumentMapper._text(item)
        return "Untitled financial document"

    @staticmethod
    def _table_title(item: object, index: int) -> str:
        caption = getattr(item, "caption", None)
        if caption is None:
            captions = getattr(item, "captions", None) or []
            caption = captions[0] if captions else None
        return DoclingDocumentMapper._text(caption) or f"Table {index + 1}"

    @staticmethod
    def _cell_type(
        cell: object,
        row_index: int,
        column_index: int,
        raw_value: str,
        markers: list[str],
    ) -> str:
        if not raw_value:
            return "blank"
        if (
            row_index == 0
            or bool(getattr(cell, "column_header", False))
            or bool(getattr(cell, "row_header", False))
        ):
            return "header"
        if column_index == 0:
            return "label"
        return "footnote_marker" if markers else "value"

    @classmethod
    def _period_label(cls, value: str) -> str | None:
        normalized = " ".join(value.split())
        if cls._period_pattern.fullmatch(normalized):
            return normalized
        return None

    @staticmethod
    def _label(item: object) -> str:
        label = getattr(item, "label", "text")
        value = getattr(label, "value", label)
        return str(value).lower().split(".")[-1]

    @staticmethod
    def _text(item: Any) -> str:
        if item is None:
            return ""
        if isinstance(item, str):
            return item.strip()
        value = getattr(item, "text", item)
        return str(value).strip() if value is not None else ""

    @staticmethod
    def _page(item: object, default: int = 1) -> int:
        for provenance in getattr(item, "prov", []) or []:
            page = getattr(provenance, "page_no", None)
            if page is None and isinstance(provenance, dict):
                page = provenance.get("page_no") or provenance.get("page")
            if page is not None:
                return int(page)
        return default


class SyntheticLayoutParser:
    """Parse the repository's fixed-width synthetic filings without an SDK.

    This adapter is intentionally limited to the fixture grammar. It proves
    that raw bytes can become canonical, page-linked evidence while the real
    Docling/Unstructured adapter remains a separate provider decision.
    """

    _page_marker = re.compile(r"^\[Page\s+(\d+)\]$")
    _footnote_marker = re.compile(r"(\[[^\]]+\])")

    def parse(self, content: bytes, options: ParseOptions) -> CanonicalDocument:
        actual_hash = content_sha256(content)
        if actual_hash != options.expected_source_content_sha256:
            raise ValueError("synthetic source content does not match expected artifact hash")
        raw_text = content.decode("utf-8")
        lines = raw_text.splitlines()
        source_uri = options.source_uri or "fixture://synthetic"
        document_id = options.document_id or stable_id(
            "document", (source_uri, options.expected_source_content_sha256)
        )
        issuer_id = options.issuer_id or "company_synthetic_acme"
        filing_date = options.filing_date
        if filing_date is None:
            raise ValueError("synthetic layout parsing requires filing_date in ParseOptions")
        document_type = options.document_type or (
            "sec_10q" if "FORM 10-Q" in raw_text else "sec_10k"
        )
        title = options.title or lines[0].strip()

        section_index = self._find_section(lines)
        table_index, table_title = self._find_table(lines)
        section_text = " ".join(
            line.strip()
            for line in lines[section_index + 1 : table_index]
            if line.strip() and not self._page_marker.match(line.strip())
        )
        if not section_text:
            section_text = lines[section_index].strip()
        section_id = stable_id("section", (document_id, str(section_index), section_text))
        section = DocumentSection(
            section_id=section_id,
            heading=lines[section_index].strip(),
            ordinal=0,
            level=1,
            text=section_text,
            locator=SourceLocator(kind="section", page=1),
        )

        footnotes_by_marker = self._parse_footnotes(lines, document_id)
        table = self._parse_table(
            lines,
            start_index=table_index,
            title=table_title,
            document_id=document_id,
            footnotes_by_marker=footnotes_by_marker,
        )
        footnotes = [
            footnote.model_copy(
                update={
                    "linked_cell_ids": [
                        cell.cell_id
                        for cell in table.cells
                        if footnote.footnote_id in cell.footnote_ids
                    ]
                }
            )
            for footnote in footnotes_by_marker.values()
        ]
        table_references = [SourceReference(kind="table", source_id=table.table_id)]
        table_references.extend(
            SourceReference(kind="table_cell", source_id=cell.cell_id) for cell in table.cells
        )
        table_references.extend(
            SourceReference(kind="footnote", source_id=footnote.footnote_id)
            for footnote in footnotes
        )
        table_chunk = DocumentChunk.build(
            document_id=document_id,
            ordinal=1,
            family="table_summary",
            text=self._table_summary(table),
            chunking_policy="synthetic-table-v1",
            source_references=table_references,
        )
        narrative_chunk = DocumentChunk.build(
            document_id=document_id,
            ordinal=0,
            family="narrative",
            text=section.text,
            chunking_policy="synthetic-section-v1",
            source_references=[SourceReference(kind="section", source_id=section.section_id)],
        )
        return CanonicalDocument(
            document_id=document_id,
            artifact_id=options.expected_artifact_id,
            document_type=document_type,
            issuer_id=issuer_id,
            accession_number=options.accession_number,
            filing_date=filing_date,
            title=title,
            parser_version=options.parser_version,
            source_content_sha256=options.expected_source_content_sha256,
            sections=[section],
            tables=[table],
            footnotes=footnotes,
            chunks=[narrative_chunk, table_chunk],
        )

    @staticmethod
    def _find_section(lines: list[str]) -> int:
        for index, line in enumerate(lines):
            if line.strip().startswith("Item "):
                return index
        raise ValueError("synthetic fixture has no Item section")

    @staticmethod
    def _find_table(lines: list[str]) -> tuple[int, str]:
        for index, line in enumerate(lines):
            title = line.strip()
            if title in {"Consolidated Results", "Quarterly Operations"}:
                return index, title
        raise ValueError("synthetic fixture has no supported table title")

    def _parse_footnotes(self, lines: list[str], document_id: str) -> dict[str, Footnote]:
        current_page = 1
        footnotes: dict[str, Footnote] = {}
        for line in lines:
            stripped = line.strip()
            page_match = self._page_marker.match(stripped)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            match = re.match(r"^(\[[^\]]+\])\s+(.+)$", stripped)
            if match:
                marker, text = match.groups()
                footnote_id = stable_id("footnote", (document_id, marker, text))
                footnotes[marker] = Footnote(
                    footnote_id=footnote_id,
                    marker=marker,
                    text=text,
                    locator=SourceLocator(
                        kind="footnote", page=current_page, footnote_id=footnote_id
                    ),
                )
        return footnotes

    def _parse_table(
        self,
        lines: list[str],
        *,
        start_index: int,
        title: str,
        document_id: str,
        footnotes_by_marker: dict[str, Footnote],
    ) -> FinancialTable:
        header_index = start_index + 1
        while header_index < len(lines) and not lines[header_index].strip():
            header_index += 1
        headers = self._split_columns(lines[header_index])
        if len(headers) < 2:
            raise ValueError("synthetic table header has no value columns")
        period_labels = self._period_labels(headers[1:])
        table_id = stable_id("table", (document_id, title))
        raw_cells: list[TableItemCell] = [
            TableItemCell(
                row_index=0,
                column_index=0,
                raw_value=headers[0],
                cell_type="header",
                page=1,
            )
        ]
        for column_index, header in enumerate(headers[1:], start=1):
            raw_cells.append(
                TableItemCell(
                    row_index=0,
                    column_index=column_index,
                    raw_value=header,
                    period_label=period_labels[column_index - 1],
                    cell_type="header",
                    page=1,
                )
            )

        current_page = 1
        row_index = 1
        for line in lines[header_index + 1 :]:
            stripped = line.strip()
            page_match = self._page_marker.match(stripped)
            if page_match:
                current_page = int(page_match.group(1))
                continue
            if not stripped or stripped.startswith(title + " (continued)"):
                continue
            if re.match(r"^\[[^\]]+\]\s+", stripped):
                continue
            columns = self._split_columns(stripped)
            if len(columns) < 2:
                continue
            label, *values = columns
            raw_cells.append(
                TableItemCell(
                    row_index=row_index,
                    column_index=0,
                    raw_value=label,
                    cell_type="label",
                    page=current_page,
                )
            )
            for column_index, value in enumerate(values[: len(period_labels)], start=1):
                markers = self._footnote_marker.findall(value)
                raw_cells.append(
                    TableItemCell(
                        row_index=row_index,
                        column_index=column_index,
                        raw_value=value,
                        period_label=period_labels[column_index - 1],
                        unit="%" if "%" in value else "USD millions",
                        cell_type="footnote_marker" if markers else "value",
                        page=current_page,
                        footnote_ids=[
                            footnotes_by_marker[marker].footnote_id
                            for marker in markers
                            if marker in footnotes_by_marker
                        ],
                    )
                )
            row_index += 1

        cells = TableNormalizer().to_cells(
            TableItem(table_id=table_id, title=title, page=1, cells=raw_cells)
        )
        footnote_ids = sorted({footnote_id for cell in cells for footnote_id in cell.footnote_ids})
        return FinancialTable(
            table_id=table_id,
            title=title,
            cells=cells,
            footnote_ids=footnote_ids,
            locator=SourceLocator(kind="table", page=1, table_id=table_id),
        )

    @staticmethod
    def _split_columns(line: str) -> list[str]:
        return [part.strip() for part in re.split(r"\s{2,}", line.strip()) if part.strip()]

    @staticmethod
    def _period_labels(headers: list[str]) -> list[str]:
        if all(re.fullmatch(r"20\d{2}", header) for header in headers):
            return [f"FY{header}" for header in headers]
        return headers

    @staticmethod
    def _table_summary(table: FinancialTable) -> str:
        values = [
            f"{cell.period_label}: {cell.normalized_value}"
            for cell in table.cells
            if cell.cell_type in {"value", "footnote_marker"} and cell.period_label
        ]
        return f"{table.title}; " + "; ".join(values)
