"""Real HTTP over a local uvicorn subprocess, explicit repository synthetic mode."""

import copy
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

pytestmark = pytest.mark.e2e
ROOT = Path(__file__).parents[2]


def _write_growth_fixture(path: Path) -> None:
    fixture = json.loads((ROOT / "tests/fixtures/graph/sample.json").read_text())
    current_path = fixture["graph_paths"][0]
    current_evidence = fixture["evidence"][0]

    def metric(year, value, evidence):
        return {
            "metric_id": f"revenue-{year}",
            "metric_name": "revenue",
            "value_decimal": value,
            "unit": "USD",
            "currency": "USD",
            "scale": "millions",
            "statement": "income_statement",
            "temporal": evidence["temporal"],
            "locator": evidence["locator"],
        }

    current_path["relationships"][0]["properties"] = {
        "metric_observation": metric(2025, "120", current_evidence)
    }
    prior_path = copy.deepcopy(current_path)
    prior_path["path_id"] = "parent-subsidiary-supplier-2024"
    prior_evidence = copy.deepcopy(current_evidence)
    prior_evidence.update(
        {
            "evidence_id": "parent-revenue-2024",
            "text": "Parent FY2024 revenue was USD 100 million. Synthetic data only.",
        }
    )
    prior_evidence["locator"].update(
        {
            "doc_id": "synthetic-parent-fy2024",
            "filing_date": "2025-02-01",
            "quote_hash": "synthetic2024",
        }
    )
    prior_evidence["temporal"].update(
        {
            "fiscal_year": 2024,
            "filing_date": "2025-02-01",
            "period_start": "2024-01-01",
            "period_end": "2024-12-31",
        }
    )
    for index, relationship in enumerate(prior_path["relationships"]):
        relationship.update(
            {
                "assertion_id": f"assertion-2024-{index}",
                "source_doc_id": "synthetic-parent-fy2024",
            }
        )
        relationship["temporal"].update(
            {
                "fiscal_year": 2024,
                "filing_date": "2025-02-01",
                "period_start": "2024-01-01",
                "period_end": "2024-12-31",
            }
        )
        relationship["locators"][0].update(
            {
                "doc_id": "synthetic-parent-fy2024",
                "filing_date": "2025-02-01",
                "quote_hash": "synthetic2024",
            }
        )
    prior_path["relationships"][0]["properties"] = {
        "metric_observation": metric(2024, "100", prior_evidence)
    }
    fixture["graph_paths"].append(prior_path)
    fixture["evidence"].append(prior_evidence)
    path.write_text(json.dumps(fixture))


def test_real_http_sse_and_reconnect(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        "FINGRAPH_RUNTIME_MODE": "synthetic",
        "FINGRAPH_FIXTURE": str(ROOT / "tests/fixtures/graph/sample.json"),
        "FINGRAPH_STATE_DIRECTORY": str(tmp_path / "state"),
        "FINGRAPH_APP_ENV": "test",
    }
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fingraph_rag.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail((tmp_path / "server.log").read_text())
                try:
                    with urlopen(base + "/health/ready", timeout=1) as response:
                        assert response.status == 200
                    break
                except (URLError, HTTPError):
                    time.sleep(0.1)
            else:
                pytest.fail((tmp_path / "server.log").read_text())
            request = Request(
                base + "/v1/query",
                method="POST",
                data=json.dumps(
                    {"question": "How does Parent affect Supplier FY2025 as filed?"}
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=15) as response:
                request_id = response.headers["X-Request-ID"]
                assert response.headers["Content-Type"].startswith("text/event-stream")
                body = response.read().decode()
            events = [
                json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")
            ]
            kinds = [event["event_type"] for event in events]
            assert kinds[0] == "accepted" and kinds[-1] == "done", body
            assert kinds.index("verification") < kinds.index("answer"), body
            answer = events[kinds.index("answer")]["payload"]["claims"]
            assert all(claim["citations"] for claim in answer)
            assert "Synthetic data only" in answer[0]["text"]
            replay = Request(
                base + f"/v1/query/{request_id}/events",
                headers={"Last-Event-ID": str(events[-2]["sequence"])},
            )
            with urlopen(replay, timeout=5) as response:
                replay_body = response.read().decode()
            assert "event: done" in replay_body and "event: answer" not in replay_body
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)


def test_real_http_sse_calculation_claim_has_provenance(tmp_path):
    fixture = tmp_path / "growth.json"
    _write_growth_fixture(fixture)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    env = {
        **os.environ,
        "FINGRAPH_RUNTIME_MODE": "synthetic",
        "FINGRAPH_FIXTURE": str(fixture),
        "FINGRAPH_STATE_DIRECTORY": str(tmp_path / "state"),
        "FINGRAPH_APP_ENV": "test",
    }
    with (tmp_path / "server.log").open("w") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "fingraph_rag.api.app:app",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
        base = f"http://127.0.0.1:{port}"
        try:
            for _ in range(100):
                if process.poll() is not None:
                    pytest.fail((tmp_path / "server.log").read_text())
                try:
                    with urlopen(base + "/health/ready", timeout=1) as response:
                        assert response.status == 200
                    break
                except (URLError, HTTPError):
                    time.sleep(0.1)
            else:
                pytest.fail((tmp_path / "server.log").read_text())
            request = Request(
                base + "/v1/query",
                method="POST",
                data=json.dumps(
                    {
                        "question": "How does Parent affect Supplier revenue growth FY2025 vs FY2024 as filed?"
                    }
                ).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=15) as response:
                body = response.read().decode()
            events = [
                json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")
            ]
            kinds = [event["event_type"] for event in events]
            assert kinds[-1] == "done", body
            assert events[-1]["payload"]["outcome"] == "answer", body
            assert "calculation" in kinds and kinds.index("calculation") < kinds.index("answer")
            claims = events[kinds.index("answer")]["payload"]["claims"]
            assert any(citation["calculation_id"] for citation in claims[0]["citations"])
            assert "yoy_growth: 20.00 %." in claims[0]["text"]
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
