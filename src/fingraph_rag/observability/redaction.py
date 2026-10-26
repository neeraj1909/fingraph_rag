"""Telemetry accepts structured metadata only; raw content is never redacted heuristically."""

import hashlib
import hmac
import math
import re
from collections.abc import Mapping
from typing import Any

SAFE_SPANS = frozenset(
    {
        "guardrail.input",
        "query.classify",
        "query.decompose",
        "retrieval.graph",
        "retrieval.vector",
        "retrieval.table",
        "graph.cypher",
        "rerank.lexical",
        "rerank.cross_encoder",
        "calculation.execute",
        "generation.llm",
        "verification.numerical",
        "verification.faithfulness",
        "verification.compliance",
        "sse.stream",
        "query.execute",
        "evaluation.run",
    }
)
NUMERIC = frozenset(
    {
        "source.count",
        "graph.nodes",
        "graph.edges",
        "graph.hops",
        "retrieval.top_k",
        "cypher.retry_count",
        "token.input",
        "token.output",
        "cost.usd",
        "duration_ms",
    }
)
BOOLEAN = frozenset({"retrieval.cache_hit"})
IDENTIFIERS = frozenset(
    {
        "request.id",
        "tenant.id",
        "route.class",
        "model.provider",
        "model.name",
        "prompt.version",
        "schema.version",
        "ontology.version",
        "cypher.template_id",
        "cypher.policy_decision",
        "verification.status",
        "guardrail.reason_codes",
    }
)


class TraceRedactor:
    def redact(self, attributes: Mapping[str, Any]) -> dict[str, Any]:
        safe: dict[str, Any] = {}
        for key, value in attributes.items():
            if (
                key in NUMERIC
                and type(value) in {int, float}
                and math.isfinite(value)
                and value >= 0
            ):
                safe[key] = value
            elif key in BOOLEAN and type(value) is bool:
                safe[key] = value
            elif (
                key in IDENTIFIERS
                and isinstance(value, str)
                and re.fullmatch(r"[A-Za-z0-9_.:/-]{1,128}", value)
            ):
                safe[key] = value
        return safe

    @staticmethod
    def correlate(value: str, key: bytes) -> str:
        if len(key) < 32:
            raise ValueError("correlation keys need at least 32 bytes")
        return hmac.new(key, value.encode(), hashlib.sha256).hexdigest()
