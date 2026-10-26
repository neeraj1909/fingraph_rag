"""Append-only local trace evidence with no raw SDK serialization."""

import json
import threading
from pathlib import Path

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from fingraph_rag.observability.redaction import SAFE_SPANS, TraceRedactor


class SafeJsonExporter(SpanExporter):
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def export(self, spans):
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            for span in spans:
                if span.name not in SAFE_SPANS:
                    continue
                record = {
                    "name": span.name,
                    "trace_id": format(span.context.trace_id, "032x"),
                    "span_id": format(span.context.span_id, "016x"),
                    "parent_span_id": format(span.parent.span_id, "016x") if span.parent else None,
                    "start_ns": span.start_time,
                    "end_ns": span.end_time,
                    "status": span.status.status_code.name,
                    "attributes": TraceRedactor().redact(span.attributes or {}),
                }
                stream.write(json.dumps(record, sort_keys=True) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None


class RedactingExporter(SpanExporter):
    """Sanitize again at export, including mutations made through the raw SDK span."""

    def __init__(self, delegate: SpanExporter):
        self.delegate = delegate

    def export(self, spans):
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import ReadableSpan
        from opentelemetry.trace import Status

        safe = [
            ReadableSpan(
                name=span.name,
                context=span.context,
                parent=span.parent,
                resource=Resource({"service.name": "fingraph-rag", "service.version": "0.1.0"}),
                attributes=TraceRedactor().redact(span.attributes or {}),
                events=(),
                links=(),
                kind=span.kind,
                status=Status(span.status.status_code),
                start_time=span.start_time,
                end_time=span.end_time,
            )
            for span in spans
            if span.name in SAFE_SPANS
        ]
        return self.delegate.export(safe)

    def shutdown(self):
        self.delegate.shutdown()

    def force_flush(self, timeout_millis=30000):
        return self.delegate.force_flush(timeout_millis)
