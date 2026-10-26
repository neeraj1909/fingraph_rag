"""Fetch and validate pinned FinQA/DocFinQA dataset sources.

The script deliberately does not contain URLs or hashes.  A caller supplies a
manifest containing those pins.  Downloads are streamed to a temporary file,
validated, and atomically moved into place.  An existing file is never
silently replaced: it must already match the configured hash and shape.
"""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO
from urllib.request import Request, urlopen

CHUNK_SIZE = 1024 * 1024
NETWORK_TIMEOUT_SECONDS = 30
ALLOWED_DATASETS = frozenset({"finqa", "docfinqa"})
REQUIRED_QA_FIELDS = frozenset({"question", "answer", "program"})


class DatasetError(ValueError):
    """Raised when a source, manifest, cache, or dataset is unsafe."""


@dataclass(frozen=True)
class Source:
    name: str
    url: str
    sha256: str
    filename: str
    revision: str | None = None
    expected_rows: int | None = None
    split: str | None = None
    require_labels: bool = True
    profile: str = "finqa_nested"


def _source_from_mapping(name: str, value: Mapping[str, Any]) -> Source:
    url = value.get("url") or value.get("uri")
    digest = value.get("sha256")
    if not isinstance(url, str) or not url:
        raise DatasetError(f"{name}: manifest source requires a non-empty url")
    if not isinstance(digest, str) or len(digest) != 64:
        raise DatasetError(f"{name}: manifest source requires a SHA-256 hex digest")
    try:
        int(digest, 16)
    except ValueError as exc:
        raise DatasetError(f"{name}: invalid SHA-256 digest") from exc
    filename = value.get("filename") or value.get("path") or f"{name}.json"
    if (
        not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or filename == "manifest.json"
    ):
        raise DatasetError(f"{name}: filename must be a simple file name")
    rows = value.get("expected_rows")
    if rows is not None and (not isinstance(rows, int) or rows < 0):
        raise DatasetError(f"{name}: expected_rows must be a non-negative integer")
    revision = value.get("revision")
    if revision is not None and not isinstance(revision, str):
        raise DatasetError(f"{name}: revision must be a string")
    split = value.get("split")
    if split is not None and (not isinstance(split, str) or not split):
        raise DatasetError(f"{name}: split must be a non-empty string")
    if split is None and "_" in name:
        split = name.partition("_")[2]
    require_labels = value.get("require_labels", split != "private_test")
    if not isinstance(require_labels, bool):
        raise DatasetError(f"{name}: require_labels must be boolean")
    profile = value.get("profile")
    if profile is None:
        profile = "docfinqa_flat" if name.startswith("docfinqa") else "finqa_nested"
    if not isinstance(profile, str) or profile not in {"finqa_nested", "docfinqa_flat"}:
        raise DatasetError(f"{name}: unsupported profile {profile!r}")
    return Source(
        name, url, digest.lower(), filename, revision, rows, split, require_labels, profile
    )


def load_manifest(path: Path) -> tuple[Source, ...]:
    """Load a manifest with either ``sources`` or ``datasets`` mapping."""
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"cannot read manifest {path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise DatasetError("manifest must be a JSON object")
    entries = raw.get("sources", raw.get("datasets"))
    if not isinstance(entries, Mapping):
        raise DatasetError("manifest must contain a sources or datasets object")
    if any(not isinstance(spec, Mapping) for spec in entries.values()):
        raise DatasetError("each manifest source must be an object")
    sources = tuple(_source_from_mapping(str(name).lower(), spec) for name, spec in entries.items())
    if not sources or any(
        not (source.name in ALLOWED_DATASETS or source.name.startswith(("finqa_", "docfinqa_")))
        for source in sources
    ):
        raise DatasetError("manifest sources must be named finqa and/or docfinqa")
    if len({source.name for source in sources}) != len(sources):
        raise DatasetError("manifest contains duplicate dataset names")
    if len({source.filename for source in sources}) != len(sources):
        raise DatasetError("manifest contains duplicate output filenames")
    return sources


def _json_array_rows_streaming(stream: BinaryIO, chunk_size: int = CHUNK_SIZE) -> Iterator[Any]:
    """Incrementally decode a root array while retaining at most one row."""
    decoder = json.JSONDecoder()
    text = ""
    position = 0
    eof = False
    started = False
    need_value = True
    after_comma = False
    closed = False
    utf8 = codecs.getincrementaldecoder("utf-8")()
    while True:
        if not eof:
            chunk = stream.read(chunk_size)
            if chunk:
                try:
                    text += utf8.decode(chunk)
                except UnicodeDecodeError as exc:
                    raise DatasetError("dataset is not valid UTF-8") from exc
            else:
                eof = True
                try:
                    text += utf8.decode(b"", final=True)
                except UnicodeDecodeError as exc:
                    raise DatasetError("dataset is not valid UTF-8") from exc
        while True:
            while position < len(text) and text[position].isspace():
                position += 1
            if not started:
                if position >= len(text):
                    if eof:
                        raise DatasetError("dataset is empty; expected a JSON array")
                    break
                if text[position] != "[":
                    raise DatasetError("dataset root must be a JSON array")
                started = True
                position += 1
                need_value = True
            while position < len(text) and text[position].isspace():
                position += 1
            if need_value and position < len(text) and text[position] == "]":
                if after_comma:
                    raise DatasetError("trailing comma in JSON array")
                if closed:
                    return
                closed = True
                position += 1
                need_value = False
            elif need_value:
                if position >= len(text) and eof:
                    raise DatasetError("unterminated JSON array")
                try:
                    value, end = decoder.raw_decode(text, position)
                except json.JSONDecodeError:
                    if eof:
                        raise DatasetError("invalid JSON in dataset") from None
                    break
                yield value
                position = end
                need_value = False
                after_comma = False
            if closed:
                if text[position:].strip():
                    raise DatasetError("trailing data after JSON array")
                if eof:
                    return
                break
            while position < len(text) and text[position].isspace():
                position += 1
            if position >= len(text):
                if eof:
                    raise DatasetError("unterminated JSON array")
                break
            if text[position] == ",":
                position += 1
                need_value = True
                after_comma = True
                continue
            if text[position] == "]":
                position += 1
                closed = True
                if eof and text[position:].strip():
                    raise DatasetError("trailing data after JSON array")
                continue
            raise DatasetError("invalid separator in JSON array")
        # Drop consumed text, preserving a partial token for the next chunk.
        if position:
            text = text[position:]
            position = 0


_json_array_rows = _json_array_rows_streaming


def _validate_dataset(
    path: Path,
    dataset_name: str,
    expected_rows: int | None = None,
    require_labels: bool = True,
    profile: str = "finqa_nested",
) -> dict[str, int]:
    """Validate JSON root/row shape and return row count without full loading."""
    count = 0
    labeled = 0
    try:
        with path.open("rb") as stream:
            for row in _json_array_rows(stream):
                if not isinstance(row, dict):
                    raise DatasetError(f"{dataset_name}: each row must be an object")
                if profile == "docfinqa_flat":
                    required = {"Context", "Question"}
                    if require_labels:
                        required |= {"Program", "Answer"}
                    missing = required - row.keys()
                    if missing:
                        raise DatasetError(
                            f"{dataset_name}: row {count} missing fields {sorted(missing)}"
                        )
                    if not all(isinstance(row[key], str) for key in required):
                        raise DatasetError(f"{dataset_name}: row {count} fields must be strings")
                    if not row["Context"].strip() or not row["Question"].strip():
                        raise DatasetError(
                            f"{dataset_name}: row {count} context/question must be non-empty"
                        )
                    if any(
                        key in row and not isinstance(row[key], str)
                        for key in {"Program", "Answer"}
                    ):
                        raise DatasetError(f"{dataset_name}: row {count} fields must be strings")
                    is_labeled = {"Program", "Answer"}.issubset(row)
                else:
                    if not isinstance(row.get("qa"), dict):
                        raise DatasetError(f"{dataset_name}: row {count} lacks an object qa field")
                    if "question" not in row["qa"]:
                        raise DatasetError(
                            f"{dataset_name}: row {count} missing qa fields ['question']"
                        )
                    nested_fields = {"question", "answer", "program"}
                    present_fields = nested_fields & row["qa"].keys()
                    if any(not isinstance(row["qa"][key], str) for key in present_fields):
                        raise DatasetError(f"{dataset_name}: row {count} qa fields must be strings")
                    missing = REQUIRED_QA_FIELDS - row["qa"].keys() if require_labels else set()
                    if missing:
                        raise DatasetError(
                            f"{dataset_name}: row {count} missing qa fields {sorted(missing)}"
                        )
                    if not row["qa"]["question"].strip():
                        raise DatasetError(
                            f"{dataset_name}: row {count} question must be non-empty"
                        )
                    for field in ("pre_text", "post_text"):
                        value = row.get(field)
                        if not isinstance(value, list) or any(
                            not isinstance(item, str) for item in value
                        ):
                            raise DatasetError(
                                f"{dataset_name}: row {count} {field} must be a string array"
                            )
                    table = row.get("table")
                    if not isinstance(table, list) or any(
                        not isinstance(cells, list)
                        or any(not isinstance(cell, str) for cell in cells)
                        for cells in table
                    ):
                        raise DatasetError(
                            f"{dataset_name}: row {count} table must be a string matrix"
                        )
                    is_labeled = REQUIRED_QA_FIELDS.issubset(row["qa"])
                if is_labeled:
                    labeled += 1
                count += 1
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetError(f"{dataset_name}: cannot validate {path}: {exc}") from exc
    if expected_rows is not None and count != expected_rows:
        raise DatasetError(f"{dataset_name}: expected {expected_rows} rows, found {count}")
    # The unlabeled split is explicitly allowed to omit answer/program; count
    # complete gold contracts for reporting without a second corpus pass.
    if require_labels:
        labeled = count
    return {"rows": count, "labeled_rows": labeled, "unlabeled_rows": count - labeled}


def validate_dataset(
    path: Path,
    dataset_name: str,
    expected_rows: int | None = None,
    profile: str = "finqa_nested",
    require_labels: bool = True,
) -> int:
    """Validate a labeled dataset and return its row count (compatibility API)."""
    return _validate_dataset(path, dataset_name, expected_rows, require_labels, profile)["rows"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(CHUNK_SIZE), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetError(f"cannot hash cached file {path}: {exc}") from exc
    return digest.hexdigest()


def _open_source(source: Source) -> BinaryIO:
    try:
        return urlopen(
            Request(source.url, headers={"User-Agent": "fingraph-rag-dataset-fetcher/1"}),
            timeout=NETWORK_TIMEOUT_SECONDS,
        )
    except OSError as exc:
        raise DatasetError(f"{source.name}: download failed: {exc}") from exc


def fetch_source(
    source: Source,
    destination: Path,
    opener: Callable[[Source], BinaryIO] = _open_source,
) -> dict[str, Any]:
    """Fetch one source, preserving a valid cache and refusing bad overwrites."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        actual = _sha256(destination)
        if actual != source.sha256:
            raise DatasetError(
                f"{source.name}: refusing to overwrite hash-mismatched cache {destination}"
            )
        stats = _validate_dataset(
            destination, source.name, source.expected_rows, source.require_labels, source.profile
        )
        return {
            "name": source.name,
            "url": source.url,
            "revision": source.revision,
            "path": str(destination),
            "sha256": actual,
            **stats,
            "cached": True,
        }

    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".part", dir=destination.parent
    )
    os.close(fd)
    temp_path = Path(temporary)
    digest = hashlib.sha256()
    try:
        with opener(source) as incoming, temp_path.open("wb") as output:
            for chunk in iter(lambda: incoming.read(CHUNK_SIZE), b""):
                if not isinstance(chunk, bytes):
                    raise DatasetError(f"{source.name}: source returned non-bytes data")
                output.write(chunk)
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != source.sha256:
            raise DatasetError(
                f"{source.name}: SHA-256 mismatch (expected {source.sha256}, got {actual})"
            )
        stats = _validate_dataset(
            temp_path, source.name, source.expected_rows, source.require_labels, source.profile
        )
        try:
            os.link(temp_path, destination)
        except FileExistsError as exc:
            raise DatasetError(
                f"{source.name}: destination appeared during download: {destination}"
            ) from exc
        temp_path.unlink(missing_ok=True)
        return {
            "name": source.name,
            "url": source.url,
            "revision": source.revision,
            "path": str(destination),
            "sha256": actual,
            **stats,
            "cached": False,
        }
    except OSError as exc:
        raise DatasetError(f"{source.name}: download failed: {exc}") from exc
    finally:
        temp_path.unlink(missing_ok=True)


def fetch_manifest(
    manifest: Path,
    output: Path,
    opener: Callable[[Source], BinaryIO] = _open_source,
    check_only: bool = False,
) -> dict[str, Any]:
    sources = load_manifest(manifest)
    if check_only:
        records = []
        for source in sources:
            destination = output / source.filename
            if not destination.is_file():
                raise DatasetError(f"{source.name}: cached file is missing: {destination}")
            actual = _sha256(destination)
            if actual != source.sha256:
                raise DatasetError(
                    f"{source.name}: cached SHA-256 mismatch (expected {source.sha256}, got {actual})"
                )
            stats = _validate_dataset(
                destination,
                source.name,
                source.expected_rows,
                source.require_labels,
                source.profile,
            )
            records.append(
                {
                    "name": source.name,
                    "url": source.url,
                    "revision": source.revision,
                    "path": str(destination),
                    "sha256": actual,
                    **stats,
                    "cached": True,
                }
            )
    else:
        records = [fetch_source(source, output / source.filename, opener) for source in sources]
    metadata = {"manifest": str(manifest), "sources": records}
    target = output / "manifest.json"
    fd, temporary = tempfile.mkstemp(prefix=".manifest.", suffix=".part", dir=output)
    os.close(fd)
    try:
        Path(temporary).write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--check-only", action="store_true", help="verify cached files without network access"
    )
    args = parser.parse_args()
    try:
        result = fetch_manifest(args.manifest, args.output, check_only=args.check_only)
    except DatasetError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
