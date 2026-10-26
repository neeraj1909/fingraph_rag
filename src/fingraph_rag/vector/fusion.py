"""Rank-based fusion without comparing incompatible dense and sparse scores."""

from collections.abc import Mapping, Sequence
from typing import Any


class RrfFusion:
    def __init__(self, rank_constant: int = 60):
        if rank_constant < 1:
            raise ValueError("rank_constant must be positive")
        self.rank_constant = rank_constant

    def fuse(
        self, rankings: Sequence[Sequence[dict[str, Any]]], limit: int
    ) -> list[dict[str, Any]]:
        if limit < 1:
            raise ValueError("limit must be positive")
        scores: dict[str, float] = {}
        records: dict[str, dict[str, Any]] = {}
        for ranking in rankings:
            seen: set[str] = set()
            for rank, record in enumerate(ranking, 1):
                if not isinstance(record, Mapping):
                    continue
                raw_identity = record.get("evidence_id")
                if not isinstance(raw_identity, str) or not raw_identity.strip():
                    continue
                identity = raw_identity
                if identity in seen:
                    continue
                seen.add(identity)
                records.setdefault(identity, record)
                scores[identity] = scores.get(identity, 0.0) + 1 / (self.rank_constant + rank)
        ordered = sorted(scores, key=lambda identity: (-scores[identity], identity))[:limit]
        return [{**records[identity], "score": scores[identity]} for identity in ordered]
