"""Shared manifests and stable evidence, while runtime timings remain measured."""

import asyncio
import json
from pathlib import Path

import pytest

from fingraph_rag.evaluation_service import (
    EvaluationManifest,
    extractive_generation,
    fixture_systems,
    load_cases,
    run_ablation,
)

pytestmark = pytest.mark.integration


def test_shared_manifest_stable_metrics_and_immutable_runs(tmp_path):
    root = Path(__file__).parents[2]
    dataset, corpus = root / "evals/datasets/smoke.jsonl", root / "tests/fixtures/graph/sample.json"
    manifest = EvaluationManifest.create(dataset, corpus)
    assert EvaluationManifest.create(dataset, corpus) == manifest
    paths = [
        asyncio.run(
            run_ablation(manifest, load_cases(dataset), fixture_systems(corpus), output=tmp_path)
        )
        for _ in range(2)
    ]
    assert paths[0] != paths[1]
    reports = [json.loads((path / "report.json").read_text()) for path in paths]
    assert all(report["manifest_id"] == manifest.manifest_id for report in reports)
    for report in reports:
        systems = report["systems"]
        assert systems["naive_hybrid"]["summary"]["subgraph_path_recall"]["value"] == 0
        assert systems["fingraph_rag"]["summary"]["subgraph_path_recall"]["value"] == 1
        for system in systems.values():
            assert system["summary"]["numerical_em"]["value"] is None
            assert system["summary"]["faithfulness"]["value"] is None
            assert system["summary"]["p95_latency_ms"]["value"] is None
            assert system["summary"]["context_recall"]["value"] == 1
    for name in reports[0]["systems"]:
        assert reports[0]["systems"][name]["summary"] == reports[1]["systems"][name]["summary"]


def test_generation_boundary_does_not_promote_invalid_metrics(tmp_path):
    root = Path(__file__).parents[2]
    dataset, corpus = root / "evals/datasets/smoke.jsonl", root / "tests/fixtures/graph/sample.json"
    manifest = EvaluationManifest.create(dataset, corpus)

    async def malformed_generation(query, context):
        return {
            "status": "provider_failed",
            "measured_metrics": {
                "faithfulness": 2,
                "numerical_em": float("nan"),
                "citation_precision": "1.0",
                "cost_per_query": -1,
                "unknown_metric": 1,
            },
        }

    path = asyncio.run(
        run_ablation(
            manifest,
            load_cases(dataset),
            {"fingraph_rag": fixture_systems(corpus)["fingraph_rag"]},
            output=tmp_path,
            generation=malformed_generation,
        )
    )
    report = json.loads((path / "report.json").read_text())
    system = report["systems"]["fingraph_rag"]
    assert all(row["generation_status"] == "unmeasured" for row in system["cases"])
    for metric in (
        "abstention_rate",
        "faithfulness",
        "numerical_em",
        "citation_precision",
        "cost_per_query",
    ):
        assert system["summary"][metric]["value"] is None


def test_extractive_runner_scores_labeled_gold_metrics(tmp_path):
    root = Path(__file__).parents[2]
    dataset, corpus = root / "evals/datasets/smoke.jsonl", root / "tests/fixtures/graph/sample.json"
    manifest = EvaluationManifest.create(dataset, corpus)
    path = asyncio.run(
        run_ablation(
            manifest,
            load_cases(dataset),
            fixture_systems(corpus),
            output=tmp_path,
            generation=extractive_generation,
        )
    )
    report = json.loads((path / "report.json").read_text())
    fingraph = report["systems"]["fingraph_rag"]["summary"]
    naive = report["systems"]["naive_hybrid"]["summary"]
    for metric in (
        "numerical_em",
        "citation_precision",
        "citation_recall",
        "temporal_consistency",
    ):
        assert fingraph[metric] == {"value": 0.5, "measured_cases": 2, "total_cases": 2}
        assert naive[metric] == {"value": 0.0, "measured_cases": 2, "total_cases": 2}
    assert fingraph["faithfulness"]["value"] == 1.0
    assert fingraph["cost_per_query"]["value"] is None
