"""Small, deterministic graph/vector ports backed by a repository fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class FixtureGraph:
    """Serve bounded graph rows without pretending to execute Cypher."""

    def __init__(self, rows: list[dict[str, Any]], *, filter_entity_ids: bool = False):
        self.rows = rows
        self.filter_entity_ids = filter_entity_ids

    async def execute_read(self, template_id: str, parameters: dict[str, Any]):
        rows = self.rows
        if self.filter_entity_ids:
            anchors = set(parameters.get("entity_ids", []))
            rows = [row for row in rows if not anchors or anchors & set(row["entity_ids"])]
        return rows[: parameters["limit"]]


class FixtureVector:
    """Return bounded fixture evidence in stable source order."""

    def __init__(self, rows: list[dict[str, Any]]):
        self.rows = rows

    async def search(self, query: str, limit: int):
        return self.rows[:limit]


@dataclass(frozen=True)
class FixtureData:
    graph_rows: list[dict[str, Any]]
    evidence_rows: list[dict[str, Any]]
    aliases: dict[str, str]

    def graph(self, *, filter_entity_ids: bool = False) -> FixtureGraph:
        return FixtureGraph(self.graph_rows, filter_entity_ids=filter_entity_ids)

    def vector(self) -> FixtureVector:
        return FixtureVector(self.evidence_rows)


def load_fixture(path: Path) -> FixtureData:
    data = json.loads(path.read_text())
    return FixtureData(
        graph_rows=data["graph_paths"],
        evidence_rows=data["evidence"],
        aliases=data.get("aliases", {}),
    )
