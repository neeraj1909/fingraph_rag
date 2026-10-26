"""Safe application telemetry."""

from fingraph_rag.observability.tracing import configure_otel, get_telemetry, set_telemetry

__all__ = ["configure_otel", "get_telemetry", "set_telemetry"]
