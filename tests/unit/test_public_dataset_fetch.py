"""Safety and shape tests for the public dataset fetcher."""

import hashlib
import importlib.util
import io
import json
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

SPEC = importlib.util.spec_from_file_location(
    "fetch_public_datasets", Path(__file__).parents[2] / "scripts/fetch_public_datasets.py"
)
fetch = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = fetch
SPEC.loader.exec_module(fetch)


def finqa_row(qa):
    return {"pre_text": [], "post_text": [], "table": [], "qa": qa}


def payload(rows=None):
    rows = rows or [finqa_row({"question": "q", "answer": "a", "program": "1"})]
    return json.dumps(rows, ensure_ascii=False).encode()


def source_for(data, name="finqa", filename="train.json"):
    return fetch.Source(
        name, "https://example.invalid/data.json", hashlib.sha256(data).hexdigest(), filename
    )


class FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def test_fetch_streams_validates_and_is_idempotent(tmp_path):
    data = payload([finqa_row({"question": "q", "answer": "a", "program": "p"})] * 2)
    source = source_for(data)
    calls = []

    def opener(_source):
        calls.append(True)
        return FakeResponse(data)

    destination = tmp_path / source.filename
    first = fetch.fetch_source(source, destination, opener)
    second = fetch.fetch_source(source, destination, opener)
    assert first["rows"] == second["rows"] == 2
    assert second["cached"] is True
    assert len(calls) == 1
    assert not list(tmp_path.glob("*.part"))


def test_hash_mismatch_never_overwrites_existing_cache(tmp_path):
    destination = tmp_path / "train.json"
    destination.write_bytes(b"corrupt")
    data = json.dumps([{"Context": "c", "Question": "q", "Program": "p", "Answer": "a"}]).encode()
    source = source_for(data)
    with pytest.raises(fetch.DatasetError, match="refusing to overwrite"):
        fetch.fetch_source(source, destination, lambda _: FakeResponse(data))
    assert destination.read_bytes() == b"corrupt"


@pytest.mark.parametrize(
    "data,pattern",
    [
        (b"{}", "root must be"),
        (payload([finqa_row({"question": "q"})]), "missing qa fields"),
        (payload([1]), "each row must be"),
        (b"[", "unterminated"),
    ],
)
def test_invalid_shape_is_rejected_without_destination(tmp_path, data, pattern):
    source = source_for(data)
    destination = tmp_path / source.filename
    with pytest.raises(fetch.DatasetError, match=pattern):
        fetch.fetch_source(source, destination, lambda _: FakeResponse(data))
    assert not destination.exists()
    assert not list(tmp_path.glob("*.part"))


def test_manifest_metadata_and_manifest_variants(tmp_path):
    data = json.dumps([{"Context": "c", "Question": "q", "Program": "p", "Answer": "a"}]).encode()
    source_digest = hashlib.sha256(data).hexdigest()
    manifest = tmp_path / "pins.json"
    manifest.write_text(
        json.dumps(
            {
                "version": 1,
                "datasets": {
                    "docfinqa": {
                        "url": "file://x",
                        "sha256": source_digest,
                        "filename": "doc.json",
                        "revision": "abc",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    result = fetch.fetch_manifest(manifest, tmp_path / "out", lambda _: FakeResponse(data))
    assert result["sources"][0]["name"] == "docfinqa"
    metadata = json.loads((tmp_path / "out" / "manifest.json").read_text())
    assert metadata["sources"][0]["sha256"] == source_digest


def test_manifest_rejects_unpinned_or_unknown_source(tmp_path):
    manifest = tmp_path / "pins.json"
    manifest.write_text(json.dumps({"sources": {"other": {"url": "x", "sha256": "0" * 64}}}))
    with pytest.raises(fetch.DatasetError, match="finqa"):
        fetch.load_manifest(manifest)


def test_manifest_rejects_bad_profile_and_manifest_filename(tmp_path):
    digest = "0" * 64
    manifest = tmp_path / "pins.json"
    manifest.write_text(
        json.dumps({"sources": {"finqa": {"url": "x", "sha256": digest, "profile": []}}})
    )
    with pytest.raises(fetch.DatasetError, match="profile"):
        fetch.load_manifest(manifest)
    manifest.write_text(
        json.dumps(
            {"sources": {"finqa": {"url": "x", "sha256": digest, "filename": "manifest.json"}}}
        )
    )
    with pytest.raises(fetch.DatasetError, match="filename"):
        fetch.load_manifest(manifest)
    manifest.write_text(
        json.dumps(
            {
                "sources": {
                    "finqa_train": {"url": "x", "sha256": digest, "filename": "same.json"},
                    "finqa_dev": {"url": "y", "sha256": digest, "filename": "same.json"},
                }
            }
        )
    )
    with pytest.raises(fetch.DatasetError, match="duplicate output"):
        fetch.load_manifest(manifest)


def test_private_test_allows_unlabeled_rows_and_reports_them(tmp_path):
    data = payload([finqa_row({"question": "q"})])
    source = fetch.Source(
        "finqa_private_test",
        "https://example.invalid/private.json",
        hashlib.sha256(data).hexdigest(),
        "private.json",
        split="private_test",
        require_labels=False,
    )
    result = fetch.fetch_source(source, tmp_path / "private.json", lambda _: FakeResponse(data))
    assert result["rows"] == 1
    assert result["labeled_rows"] == 0
    assert result["unlabeled_rows"] == 1


def test_docfinqa_flat_profile(tmp_path):
    data = payload([{"Context": "c", "Question": "q", "Program": "p", "Answer": "a"}])
    source = fetch.Source(
        "docfinqa_train",
        "https://example.invalid/doc.json",
        hashlib.sha256(data).hexdigest(),
        "doc.json",
        profile="docfinqa_flat",
    )
    assert (
        fetch.fetch_source(source, tmp_path / "doc.json", lambda _: FakeResponse(data))["rows"] == 1
    )


def test_finqa_context_fields_are_validated_for_every_row(tmp_path):
    valid = finqa_row({"question": "q", "answer": "a", "program": "1"})
    valid["pre_text"] = ["prior context"]
    valid["post_text"] = ["later context"]
    valid["table"] = [["Metric", "Value"]]
    path = tmp_path / "valid.json"
    path.write_bytes(payload([valid]))
    assert fetch.validate_dataset(path, "finqa") == 1

    for field, malformed in (
        ("pre_text", "not a list"),
        ("post_text", [1]),
        ("table", [["Metric", 1]]),
    ):
        invalid = dict(valid, **{field: malformed})
        path.write_bytes(payload([invalid]))
        with pytest.raises(fetch.DatasetError, match=field if field != "table" else "table"):
            fetch.validate_dataset(path, "finqa")


def test_destination_created_during_download_is_not_overwritten(tmp_path):
    data = payload()
    source = source_for(data)
    destination = tmp_path / source.filename

    def opener(_source):
        destination.write_bytes(b"another process wrote this")
        return FakeResponse(data)

    with pytest.raises(fetch.DatasetError, match="destination appeared"):
        fetch.fetch_source(source, destination, opener)
    assert destination.read_bytes() == b"another process wrote this"
    assert not list(tmp_path.glob("*.part"))


@pytest.mark.parametrize(
    "data,pattern", [(b'[{"x": 1},]', "trailing comma"), (b'[{"x": 1}] trailing', "trailing data")]
)
def test_json_parser_rejects_trailing_syntax_across_tiny_chunks(data, pattern):
    with pytest.raises(fetch.DatasetError, match=pattern):
        list(fetch._json_array_rows(io.BytesIO(data), chunk_size=1))


def test_json_parser_handles_separator_at_chunk_boundary_and_rejects_missing_comma():
    row = b'{"qa":{"question":"q","answer":"a","program":"p"}}'
    assert list(
        fetch._json_array_rows(io.BytesIO(b"[" + row + b"," + row + b"]"), chunk_size=1)
    ) == [
        {"qa": {"question": "q", "answer": "a", "program": "p"}},
        {"qa": {"question": "q", "answer": "a", "program": "p"}},
    ]
    with pytest.raises(fetch.DatasetError, match="separator"):
        list(fetch._json_array_rows(io.BytesIO(b"[" + row + row + b"]"), chunk_size=1))


def test_json_parser_checks_invalid_utf8_after_closing_array():
    with pytest.raises(fetch.DatasetError, match="UTF-8"):
        list(fetch._json_array_rows(io.BytesIO(b"[]\xff"), chunk_size=1))
