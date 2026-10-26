#!/usr/bin/env python3
"""Verify the bounded application JSON trace artifact; never print raw spans."""

import argparse
import json
import re
from pathlib import Path

from fingraph_rag.observability.redaction import SAFE_SPANS, TraceRedactor


def verify_trace(path: Path) -> dict:
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not records:
        raise ValueError("trace is empty")
    allowed_fields = {
        "name",
        "trace_id",
        "span_id",
        "parent_span_id",
        "start_ns",
        "end_ns",
        "status",
        "attributes",
    }

    def matches(value, pattern):
        return isinstance(value, str) and re.fullmatch(pattern, value) is not None

    for record in records:
        if not isinstance(record, dict):
            raise ValueError("trace contains a non-object record")
        if (
            set(record) != allowed_fields
            or not isinstance(record["name"], str)
            or record["name"] not in SAFE_SPANS
        ):
            raise ValueError("trace contains unapproved fields or span names")
        if (
            not matches(record["trace_id"], r"[0-9a-f]{32}")
            or not matches(record["span_id"], r"[0-9a-f]{16}")
            or (
                record["parent_span_id"] is not None
                and not matches(record["parent_span_id"], r"[0-9a-f]{16}")
            )
            or type(record["start_ns"]) is not int
            or type(record["end_ns"]) is not int
            or record["start_ns"] < 0
            or record["end_ns"] < record["start_ns"]
            or not isinstance(record["status"], str)
            or record["status"] not in {"UNSET", "OK", "ERROR"}
            or not isinstance(record["attributes"], dict)
        ):
            raise ValueError("trace contains malformed span metadata")
        if record["attributes"] != TraceRedactor().redact(record["attributes"]):
            raise ValueError("trace contains unapproved attribute values")
    return {
        "verified": True,
        "span_count": len(records),
        "span_names": sorted({record["name"] for record in records}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path)
    parser.add_argument("--latest", action="store_true")
    parser.add_argument("--directory", type=Path, default=Path("reports/traces"))
    parser.add_argument("--verify-redaction", action="store_true")
    args = parser.parse_args()
    path = args.path
    if path is None and args.latest:
        candidates = list(args.directory.glob("*.jsonl"))
        if candidates:
            path = max(candidates, key=lambda value: value.stat().st_mtime_ns)
    if path is None:
        parser.error("supply --path or a nonempty --latest directory")
    print(json.dumps(verify_trace(path), sort_keys=True))


if __name__ == "__main__":
    main()
