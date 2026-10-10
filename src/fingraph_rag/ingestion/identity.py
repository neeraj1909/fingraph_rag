"""Deterministic identities for replayable ingestion records."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable


def stable_id(prefix: str, parts: Iterable[str]) -> str:
    """Return a deterministic opaque ID from behaviorally relevant parts."""

    payload = "\x1f".join(parts).encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()[:32]
    return f"{prefix}_{digest}"


def content_sha256(content: bytes) -> str:
    """Return the canonical SHA-256 digest for acquired bytes."""

    return hashlib.sha256(content).hexdigest()
