"""Parser ports and the deterministic parser used by synthetic fixtures.

The fixture parser deliberately accepts canonical JSON rather than pretending
that a fixture is a PDF parser. A Docling/Unstructured adapter can implement
the same port later without changing the ingestion service or its contracts.
"""

from __future__ import annotations

import re
from datetime import date
from typing import Protocol

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
