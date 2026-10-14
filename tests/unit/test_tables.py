from datetime import UTC, datetime
from pathlib import Path

import pytest

from fingraph_rag.ingestion.contracts import RawArtifact
from fingraph_rag.ingestion.parsers import (
    FixtureDocumentParser,
    ParseOptions,
    SyntheticLayoutParser,
)
from fingraph_rag.ingestion.provenance import FootnoteLinker, ProvenanceIndex
from fingraph_rag.ingestion.tables import TableItem, TableItemCell, TableNormalizer

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "filings"


@pytest.mark.unit
def test_table_normalizer_preserves_positions_and_strips_only_markers() -> None:
    table = TableItem(
        table_id="table-1",
        title="Synthetic results",
        page=7,
        cells=[
            TableItemCell(
                row_index=0,
                column_index=0,
                raw_value="Operating margin",
                cell_type="label",
            ),
            TableItemCell(
                row_index=0,
                column_index=1,
                raw_value="18.4%[1]",
                cell_type="footnote_marker",
                footnote_ids=["footnote-1"],
                period_label="FY2024",
                unit="%",
            ),
        ],
    )

    cells = TableNormalizer().to_cells(table)

    assert cells[0].locator.page == 7
    assert cells[1].normalized_value == "18.4%"
    assert cells[1].footnote_ids == ["footnote-1"]
    assert cells[1].locator.cell_id == cells[1].cell_id


@pytest.mark.unit
def test_table_normalizer_rejects_duplicate_positions() -> None:
    table = TableItem(
        table_id="table-1",
        title="Invalid table",
        cells=[
            TableItemCell(row_index=0, column_index=0, raw_value="A", cell_type="label"),
            TableItemCell(row_index=0, column_index=0, raw_value="B", cell_type="value"),
        ],
    )

    with pytest.raises(ValueError, match="duplicate table cell position"):
        TableNormalizer().to_cells(table)


@pytest.mark.unit
def test_fixture_parser_and_linker_preserve_cell_level_provenance() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_10k.txt"
    artifact = RawArtifact.from_bytes(
        raw_path.read_bytes(),
        source_uri="fixture://synthetic-acme/2024-10k",
        media_type="text/plain",
        acquired_at=datetime(2025, 2, 28, 12, tzinfo=UTC),
    )
    document = FixtureDocumentParser().parse(
        (FIXTURE_DIR / "synthetic_acme_10k.json").read_bytes(),
        ParseOptions(
            parser_version="fixture-parser-1",
            expected_artifact_id=artifact.artifact_id,
            expected_source_content_sha256=artifact.content_sha256,
        ),
    )
    table = document.tables[0]
    cells_without_reverse_link = [
        cell.model_copy(update={"footnote_ids": []}) if cell.cell_id == "cell_margin_2024" else cell
        for cell in table.cells
    ]
    asymmetric = document.model_copy(
        update={"tables": [table.model_copy(update={"cells": cells_without_reverse_link})]}
    )

    linked = FootnoteLinker().link(asymmetric)
    index = ProvenanceIndex.from_document(linked)

    assert index.require(document.chunks[1].source_references[0]).table_id == table.table_id
    margin = index.cells["cell_margin_2024"]
    assert margin.footnote_ids == ["footnote_margin_adjustment"]
    assert index.footnotes["footnote_margin_adjustment"].linked_cell_ids == ["cell_margin_2024"]


@pytest.mark.unit
def test_multi_page_fixture_keeps_continued_table_cell_pages() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt"
    artifact = RawArtifact.from_bytes(
        raw_path.read_bytes(),
        source_uri="fixture://synthetic-acme/2024-q3-10q",
        media_type="text/plain",
        acquired_at=datetime(2024, 11, 1, 12, tzinfo=UTC),
    )
    document = FixtureDocumentParser().parse(
        (FIXTURE_DIR / "synthetic_acme_2024_q3_10q.json").read_bytes(),
        ParseOptions(
            parser_version="fixture-parser-1",
            expected_artifact_id=artifact.artifact_id,
            expected_source_content_sha256=artifact.content_sha256,
        ),
    )
    linked = FootnoteLinker().link(document)
    table = linked.tables[0]

    assert {cell.locator.page for cell in table.cells} == {1, 2}
    assert table.cells[-1].locator.table_id == table.table_id
    assert linked.footnotes[0].linked_cell_ids == ["cell_q3_margin"]
    assert any(
        reference.source_id == "cell_q3_supplier_concentration"
        for reference in linked.chunks[1].source_references
    )


@pytest.mark.unit
def test_synthetic_layout_parser_builds_stable_canonical_provenance() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt"
    content = raw_path.read_bytes()
    artifact = RawArtifact.from_bytes(
        content,
        source_uri="fixture://synthetic_acme_2024_q3_10q.txt",
        media_type="text/plain",
        acquired_at=datetime(2024, 11, 1, 12, tzinfo=UTC),
    )
    options = ParseOptions(
        parser_version="synthetic-layout-1",
        expected_artifact_id=artifact.artifact_id,
        expected_source_content_sha256=artifact.content_sha256,
        source_uri=artifact.source_uri,
        document_id="doc_synthetic_acme_2024_q3_10q_raw",
        issuer_id="company_synthetic_acme",
        document_type="sec_10q",
        filing_date=datetime(2024, 11, 1, tzinfo=UTC).date(),
        title="Synthetic Acme Corporation Form 10-Q Fixture",
    )

    first = SyntheticLayoutParser().parse(content, options)
    replay = SyntheticLayoutParser().parse(content, options)
    table = first.tables[0]
    margin_cell = next(cell for cell in table.cells if cell.footnote_ids)

    assert first.document_id == replay.document_id
    assert first.tables[0].table_id == replay.tables[0].table_id
    assert {cell.locator.page for cell in table.cells} == {1, 2}
    assert table.cells[-1].locator.page == 2
    assert margin_cell.locator.page == 1
    assert first.footnotes[0].linked_cell_ids == [margin_cell.cell_id]
    assert first.chunks[1].source_references[0].source_id == table.table_id
