from dataclasses import dataclass
from pathlib import Path

import pytest

from fingraph_rag.ingestion.contracts import RawArtifact
from fingraph_rag.ingestion.parsers import DoclingLayoutParser, ParseOptions

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "filings"


@dataclass
class FakeProvenance:
    page_no: int


@dataclass
class FakeTextItem:
    text: str
    label: str
    prov: list[FakeProvenance]


@dataclass
class FakeTableCell:
    text: str
    prov: list[FakeProvenance]
    column_header: bool = False
    row_header: bool = False


@dataclass
class FakeTableData:
    grid: list[list[FakeTableCell]]


@dataclass
class FakeTableItem:
    data: FakeTableData
    prov: list[FakeProvenance]
    caption: str


@dataclass
class FakeDocument:
    name: str
    texts: list[FakeTextItem]
    tables: list[FakeTableItem]


@dataclass
class FakeConversion:
    document: FakeDocument


class FakeDoclingConverter:
    def __init__(self, document: FakeDocument) -> None:
        self.document = document
        self.received_source: object | None = None

    def convert(self, source: object) -> FakeConversion:
        self.received_source = source
        return FakeConversion(self.document)


@pytest.mark.unit
def test_docling_adapter_maps_pages_tables_and_footnotes_without_sdk_import() -> None:
    raw_path = FIXTURE_DIR / "synthetic_acme_2024_q3_10q.txt"
    content = raw_path.read_bytes()
    artifact = RawArtifact.from_bytes(
        content,
        source_uri="fixture://synthetic_acme_2024_q3_10q.txt",
        media_type="text/plain",
        acquired_at="2024-11-01T12:00:00Z",
        license_reference="synthetic-fixture",
    )
    document = FakeDocument(
        name="synthetic_acme_2024_q3_10q.pdf",
        texts=[
            FakeTextItem(
                text="Synthetic Acme Corporation Form 10-Q Fixture",
                label="title",
                prov=[FakeProvenance(1)],
            ),
            FakeTextItem(
                text="Item 2. Management's Discussion and Analysis",
                label="section_header",
                prov=[FakeProvenance(1)],
            ),
            FakeTextItem(
                text="Supplier concentration affected quarterly operations.",
                label="text",
                prov=[FakeProvenance(1)],
            ),
            FakeTextItem(
                text="[1] Margin excludes restructuring costs.",
                label="footnote",
                prov=[FakeProvenance(1)],
            ),
        ],
        tables=[
            FakeTableItem(
                caption="Quarterly Operations",
                prov=[FakeProvenance(1)],
                data=FakeTableData(
                    grid=[
                        [
                            FakeTableCell("Metric", [FakeProvenance(1)]),
                            FakeTableCell("Q3 2024", [FakeProvenance(1)]),
                            FakeTableCell("Q2 2024", [FakeProvenance(1)]),
                        ],
                        [
                            FakeTableCell("Operating margin", [FakeProvenance(1)]),
                            FakeTableCell("18.4%[1]", [FakeProvenance(1)]),
                            FakeTableCell("17.9%", [FakeProvenance(1)]),
                        ],
                        [
                            FakeTableCell("Supplier concentration", [FakeProvenance(2)]),
                            FakeTableCell("42%", [FakeProvenance(2)]),
                            FakeTableCell("39%", [FakeProvenance(2)]),
                        ],
                    ]
                ),
            )
        ],
    )
    converter = FakeDoclingConverter(document)
    parser = DoclingLayoutParser(converter, source_factory=lambda raw, _: raw)
    options = ParseOptions(
        parser_version="docling-v2-adapter-1",
        expected_artifact_id=artifact.artifact_id,
        expected_source_content_sha256=artifact.content_sha256,
        source_uri=artifact.source_uri,
        document_id="doc_synthetic_acme_2024_q3_10q_docling",
        issuer_id="company_synthetic_acme",
        document_type="sec_10q",
        filing_date="2024-11-01",
        title="Synthetic Acme Corporation Form 10-Q Fixture",
    )

    canonical = parser.parse(content, options)

    assert converter.received_source == content
    assert canonical.document_id == options.document_id
    assert canonical.tables[0].title == "Quarterly Operations"
    assert {cell.locator.page for cell in canonical.tables[0].cells} == {1, 2}
    margin = next(cell for cell in canonical.tables[0].cells if cell.footnote_ids)
    assert margin.normalized_value == "18.4%"
    assert canonical.footnotes[0].linked_cell_ids == [margin.cell_id]
    assert canonical.chunks[1].family == "table_summary"

    with pytest.raises(ValueError, match="expected artifact hash"):
        parser.parse(b"tampered", options)
