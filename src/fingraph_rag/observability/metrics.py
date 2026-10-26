"""Bounded metric vocabulary; no arbitrary user labels."""

import math

from opentelemetry.sdk.metrics import MeterProvider


class Metrics:
    def __init__(self, provider: MeterProvider):
        meter = provider.get_meter("fingraph_rag", "0.1.0")
        self.requests = meter.create_counter("fingraph.requests", unit="{request}")
        self.duration = meter.create_histogram("fingraph.query.duration", unit="ms")
        self.rejections = meter.create_counter("fingraph.guardrail.rejections", unit="{decision}")

    def record_request(self, route: str, outcome: str, duration_ms: float):
        if route not in {"fact_table", "multi_hop", "cross_period", "thematic", "unknown"}:
            route = "unknown"
        if outcome not in {"answer", "abstained", "error"}:
            outcome = "error"
        if type(duration_ms) not in {int, float} or not math.isfinite(duration_ms):
            duration_ms = 0.0
        labels = {"route": route, "outcome": outcome}
        self.requests.add(1, labels)
        self.duration.record(max(0.0, duration_ms), labels)
