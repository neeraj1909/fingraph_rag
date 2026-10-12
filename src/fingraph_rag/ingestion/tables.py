"""Layout-independent table normalization with cell-level provenance."""

from __future__ import annotations

import re
from typing import Literal

from pydantic import Field

from fingraph_rag.domain.contracts import ContractModel

from .contracts import SourceLocator, TableCell
from .identity import stable_id


class TableItemCell(ContractModel):
    """The small raw-cell shape emitted by a layout parser adapter."""

    row_index: int = Field(ge=0)
    column_index: int = Field(ge=0)
    raw_value: str
    cell_type: Literal["header", "label", "value", "blank", "footnote_marker"]
    page: int | None = Field(default=None, ge=1)
    cell_id: str | None = None
    period_label: str | None = None
    unit: str | None = None
    footnote_ids: list[str] = Field(default_factory=list)


class TableItem(ContractModel):
    """Parser-neutral table item before it becomes a canonical financial table."""

    table_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    page: int | None = Field(default=None, ge=1)
    cells: list[TableItemCell] = Field(min_length=1)


class TableNormalizer:
    """Convert parser cells without discarding layout or source identity."""

    _footnote_marker = re.compile(r"\s*(?:\[[^\]]+\]|\(\d+\))\s*$")

    def to_cells(self, table_item: TableItem) -> list[TableCell]:
        seen_positions: set[tuple[int, int]] = set()
        cells: list[TableCell] = []
        for item in table_item.cells:
            position = (item.row_index, item.column_index)
            if position in seen_positions:
                raise ValueError(f"duplicate table cell position: {position}")
            seen_positions.add(position)

            cell_id = item.cell_id or stable_id(
                "cell",
                (table_item.table_id, str(item.row_index), str(item.column_index)),
            )
            cells.append(
                TableCell(
                    cell_id=cell_id,
                    table_id=table_item.table_id,
                    row_index=item.row_index,
                    column_index=item.column_index,
                    raw_value=item.raw_value,
                    normalized_value=self._normalize_value(item.raw_value),
                    cell_type=item.cell_type,
                    period_label=item.period_label,
                    unit=item.unit,
                    footnote_ids=list(item.footnote_ids),
                    locator=SourceLocator(
                        kind="table_cell",
                        page=item.page or table_item.page,
                        table_id=table_item.table_id,
                        cell_id=cell_id,
                    ),
                )
            )
        return cells

    @classmethod
    def _normalize_value(cls, raw_value: str) -> str:
        value = " ".join(raw_value.split())
        return cls._footnote_marker.sub("", value).strip()
