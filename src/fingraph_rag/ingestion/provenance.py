"""Source-reference indexes and bidirectional footnote provenance."""

from __future__ import annotations

from dataclasses import dataclass

from .contracts import CanonicalDocument, Footnote, SourceReference, TableCell


@dataclass(frozen=True)
class ProvenanceIndex:
    """Read-only lookup index for source references in one canonical document."""

    sections: dict[str, object]
    tables: dict[str, object]
    cells: dict[str, TableCell]
    footnotes: dict[str, Footnote]

    @classmethod
    def from_document(cls, document: CanonicalDocument) -> ProvenanceIndex:
        return cls(
            sections={section.section_id: section for section in document.sections},
            tables={table.table_id: table for table in document.tables},
            cells={cell.cell_id: cell for table in document.tables for cell in table.cells},
            footnotes={footnote.footnote_id: footnote for footnote in document.footnotes},
        )

    def require(self, reference: SourceReference) -> object:
        collection = {
            "section": self.sections,
            "table": self.tables,
            "table_cell": self.cells,
            "footnote": self.footnotes,
        }[reference.kind]
        try:
            return collection[reference.source_id]
        except KeyError as exc:
            raise ValueError(
                f"unknown {reference.kind} source reference: {reference.source_id}"
            ) from exc


class FootnoteLinker:
    """Make explicit cell↔footnote links symmetric before indexing."""

    def link(self, document: CanonicalDocument) -> CanonicalDocument:
        index = ProvenanceIndex.from_document(document)
        cell_to_footnotes = {
            cell_id: set(cell.footnote_ids) for cell_id, cell in index.cells.items()
        }
        footnote_to_cells = {
            footnote_id: set(footnote.linked_cell_ids)
            for footnote_id, footnote in index.footnotes.items()
        }

        for cell_id, footnote_ids in cell_to_footnotes.items():
            for footnote_id in footnote_ids:
                if footnote_id not in footnote_to_cells:
                    raise ValueError(f"cell references unknown footnote: {footnote_id}")
                footnote_to_cells[footnote_id].add(cell_id)
        for footnote_id, cell_ids in footnote_to_cells.items():
            for cell_id in cell_ids:
                if cell_id not in cell_to_footnotes:
                    raise ValueError(f"footnote references unknown table cell: {cell_id}")
                cell_to_footnotes[cell_id].add(footnote_id)

        tables = [
            table.model_copy(
                update={
                    "cells": [
                        cell.model_copy(
                            update={"footnote_ids": sorted(cell_to_footnotes[cell.cell_id])}
                        )
                        for cell in table.cells
                    ]
                }
            )
            for table in document.tables
        ]
        footnotes = [
            footnote.model_copy(
                update={"linked_cell_ids": sorted(footnote_to_cells[footnote.footnote_id])}
            )
            for footnote in document.footnotes
        ]
        return document.model_copy(update={"tables": tables, "footnotes": footnotes})
