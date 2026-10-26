"""Checkpoint-safe workflow state; API clients never submit this structure."""

from typing import Any, TypedDict


class QueryState(TypedDict, total=False):
    request_id: str
    question: str
    query_spec: dict[str, Any]
    context: dict[str, Any]
    generation: dict[str, Any]
    candidates: list[dict[str, Any]]
    claims: list[dict[str, Any]]
    repairs: int
    repair_reasons: list[str]
    terminal: str
