"""Application-owned OpenTelemetry instrumentation with explicit safe export."""

from contextlib import contextmanager
from pathlib import Path

from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import InMemoryMetricReader
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.trace import Status, StatusCode

from fingraph_rag.observability.exporters import RedactingExporter, SafeJsonExporter
from fingraph_rag.observability.metrics import Metrics
from fingraph_rag.observability.redaction import SAFE_SPANS, TraceRedactor


class Telemetry:
    def __init__(
        self,
        provider: TracerProvider,
        meter_provider: MeterProvider,
        metric_reader: InMemoryMetricReader | None = None,
    ):
        self.provider = provider
        self.meter_provider = meter_provider
        self.metric_reader = metric_reader
        self.tracer = provider.get_tracer("fingraph_rag", "0.1.0")
        self.metrics = Metrics(meter_provider)

    @contextmanager
    def span(self, name: str, attributes=None):
        if name not in SAFE_SPANS:
            raise ValueError("span name is outside the application allowlist")
        with self.tracer.start_as_current_span(
            name,
            attributes=TraceRedactor().redact(attributes or {}),
            record_exception=False,
            set_status_on_exception=False,
        ) as span:
            try:
                yield span
            except BaseException:
                span.set_status(Status(StatusCode.ERROR))
                raise

    def flush(self):
        return self.provider.force_flush()

    def shutdown(self):
        self.provider.shutdown()
        self.meter_provider.shutdown()


def configure_otel(*, endpoint: str | None = None, output_path: Path | None = None) -> Telemetry:
    provider = TracerProvider(
        resource=Resource({"service.name": "fingraph-rag", "service.version": "0.1.0"})
    )
    if output_path is not None:
        provider.add_span_processor(SimpleSpanProcessor(SafeJsonExporter(output_path)))
    if endpoint:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        provider.add_span_processor(
            SimpleSpanProcessor(RedactingExporter(OTLPSpanExporter(endpoint=endpoint, timeout=5)))
        )
    metric_reader = InMemoryMetricReader()
    meter_provider = MeterProvider(
        resource=Resource({"service.name": "fingraph-rag"}),
        metric_readers=[metric_reader],
    )
    return Telemetry(provider, meter_provider, metric_reader)


_default = None


def get_telemetry() -> Telemetry:
    global _default
    if _default is None:
        _default = configure_otel()
    return _default


def set_telemetry(telemetry: Telemetry) -> None:
    global _default
    _default = telemetry
