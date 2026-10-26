"""Reproducible ablations through the production retrieval core.

Absent generation/provider/cost evidence is explicitly unmeasured, never scored as
success. Dataset questions and raw context stay in local evaluation artifacts.
"""

import hashlib
import json
import math
import platform
import re
import subprocess
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import Field

from fingraph_rag.domain.contracts import ContractModel, QuerySpec
from fingraph_rag.observability import get_telemetry
from fingraph_rag.retrieval import FusedContext, RetrievalBudget, RetrievalService
from fingraph_rag.retrieval.fixture_adapter import FixtureGraph, load_fixture

METRICS = (
    "context_precision",
    "context_recall",
    "multi_hop_entity_coverage",
    "subgraph_path_recall",
    "faithfulness",
    "numerical_em",
    "temporal_consistency",
    "citation_precision",
    "citation_recall",
    "cypher_success",
    "cost_per_query",
    "p50_latency_ms",
    "p95_latency_ms",
    "abstention_rate",
)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


class EvaluationManifest(ContractModel):
    manifest_id: str
    dataset_sha256: str
    corpus_sha256: str
    graph_schema: str = "temporal-assertion-v1"
    vector_schema: str = "source-chunk-v1"
    prompt_versions: dict[str, str] = Field(default_factory=dict)
    model_ids: list[str] = Field(default_factory=list)
    embedding_ids: list[str] = Field(default_factory=lambda: ["fixture-replay-no-embedding"])
    reranker_id: str = "lexical-baseline-v1"
    retrieval_parameters: dict[str, Any]
    seed: int = 0
    software_commit: str
    source_sha256: str
    environment: str

    @classmethod
    def create(
        cls,
        dataset: Path,
        corpus: Path,
        *,
        budget: RetrievalBudget | None = None,
        model_ids: list[str] | None = None,
        prompt_versions: dict[str, str] | None = None,
    ) -> "EvaluationManifest":
        root = Path(__file__).parents[2]
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        source = hashlib.sha256()
        for path in sorted((root / "src").rglob("*.py")):
            source.update(str(path.relative_to(root)).encode())
            source.update(path.read_bytes())
        for path in (root / "pyproject.toml", root / "uv.lock"):
            source.update(path.read_bytes())
        values = dict(
            dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),
            corpus_sha256=hashlib.sha256(corpus.read_bytes()).hexdigest(),
            retrieval_parameters=(budget or RetrievalBudget()).model_dump(),
            software_commit=commit,
            source_sha256=source.hexdigest(),
            environment=f"python-{platform.python_version()}-{platform.system()}",
            model_ids=model_ids or [],
            prompt_versions=prompt_versions or {},
        )
        return cls(manifest_id=canonical_hash(values), **values)


class NaiveHybridBaseline:
    def __init__(self, service: RetrievalService):
        self.service = service

    async def run(self, query: QuerySpec) -> FusedContext:
        return await self.service.retrieve(query)


class GraphRagSystem(NaiveHybridBaseline):
    """Same production service; caller injects graph-enabled storage ports."""


GenerationRunner = Callable[[QuerySpec, FusedContext], Awaitable[dict[str, Any]]]

GENERATION_METRICS = (
    "faithfulness",
    "numerical_em",
    "temporal_consistency",
    "citation_precision",
    "citation_recall",
    "cost_per_query",
)
BOUNDED_GENERATION_METRICS = frozenset(GENERATION_METRICS) - {"cost_per_query"}
NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.])")


def measured_generation_metrics(generated: Any) -> dict[str, float]:
    """Accept only finite, verifier-owned values from a generation boundary.

    The runner is an integration seam: its return value is deliberately not a
    Pydantic model because providers and the local verifier have different
    contracts. Treat malformed, unknown, or out-of-range values as unmeasured
    rather than allowing them to become benchmark claims.
    """
    if not isinstance(generated, dict):
        return {}
    values = generated.get("measured_metrics")
    if not isinstance(values, dict):
        return {}
    measured = {}
    for metric in GENERATION_METRICS:
        value = values.get(metric)
        if type(value) not in {int, float} or value < 0:
            continue
        try:
            finite = math.isfinite(value)
        except OverflowError:
            finite = False
        if not finite:
            continue
        if metric in BOUNDED_GENERATION_METRICS and value > 1:
            continue
        measured[metric] = float(value)
    return measured


def _gold_answers(case: dict[str, Any]) -> list[str] | None:
    answers = case.get("gold_answer_texts")
    if not isinstance(answers, list) or not all(isinstance(answer, str) for answer in answers):
        return None
    return answers


def _verified_claims(generated: Any) -> list[dict[str, Any]] | None:
    if not isinstance(generated, dict) or not isinstance(generated.get("verified_claims"), list):
        return None
    claims = generated["verified_claims"]
    if not all(isinstance(claim, dict) and isinstance(claim.get("text"), str) for claim in claims):
        return None
    return claims


def deterministic_generation_metrics(generated: Any, case: dict[str, Any]) -> dict[str, float]:
    """Score only explicit synthetic gold against verified extractive claims.

    A callback that does not provide the structured claim metadata used here is
    intentionally left unmeasured. This prevents arbitrary provider output from
    being mistaken for a citation, temporal, or numerical benchmark.
    """
    answers = _gold_answers(case)
    claims = _verified_claims(generated)
    if answers is None or claims is None:
        return {}

    gold_numbers = {tuple(NUMBER.findall(answer)) for answer in answers}
    predicted_numbers = {tuple(NUMBER.findall(claim["text"])) for claim in claims}
    numerical_em = float(bool(predicted_numbers & gold_numbers))

    raw_gold_docs = case.get("gold_citation_doc_ids")
    gold_docs = (
        set(raw_gold_docs)
        if isinstance(raw_gold_docs, list) and all(isinstance(item, str) for item in raw_gold_docs)
        else None
    )
    predicted_docs = {
        doc_id
        for claim in claims
        for doc_id in claim.get("citation_doc_ids", [])
        if isinstance(doc_id, str)
    }
    citation_precision = citation_recall = None
    if gold_docs is not None:
        cited_docs = predicted_docs & gold_docs
        citation_precision = (
            fraction(len(cited_docs), len(predicted_docs)) if predicted_docs else 0.0
        )
        citation_recall = fraction(len(cited_docs), len(gold_docs)) if gold_docs else None

    def temporal_key(value: Any):
        if not isinstance(value, dict):
            return None
        return tuple(
            value.get(key) for key in ("fiscal_year", "fiscal_quarter", "restatement_version")
        )

    raw_gold_temporal = case.get("gold_temporal_contexts")
    gold_temporal = (
        {key for item in raw_gold_temporal if (key := temporal_key(item)) is not None}
        if isinstance(raw_gold_temporal, list)
        else None
    )
    predicted_temporal = {
        key
        for claim in claims
        for item in claim.get("temporal_contexts", [])
        if (key := temporal_key(item)) is not None
    }
    temporal_consistency = None
    if gold_temporal is not None:
        temporal_consistency = (
            fraction(len(predicted_temporal & gold_temporal), len(predicted_temporal))
            if predicted_temporal
            else 0.0
        )
    metrics = {
        "numerical_em": numerical_em,
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "temporal_consistency": temporal_consistency,
    }
    return {metric: value for metric, value in metrics.items() if value is not None}


def fraction(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def context_metrics(context: FusedContext, case: dict[str, Any]) -> dict[str, float | None]:
    evidence = context.text_evidence + context.table_evidence
    retrieved = {e.evidence_id for e in evidence}
    gold = set(case.get("gold_evidence_ids", []))
    entities = {entity for path in context.graph_paths for entity in path.entity_ids} | {
        entity for item in evidence for entity in item.entity_ids
    }
    gold_entities = set(case.get("gold_entity_ids", []))
    paths = {path.path_id for path in context.graph_paths}
    gold_paths = set(case.get("gold_path_ids", []))
    return {
        "context_precision": fraction(len(retrieved & gold), len(retrieved)) if gold else None,
        "context_recall": fraction(len(retrieved & gold), len(gold)),
        "multi_hop_entity_coverage": fraction(len(entities & gold_entities), len(gold_entities)),
        "subgraph_path_recall": fraction(len(paths & gold_paths), len(gold_paths)),
    }


def percentile(values: list[float], percentile: float) -> float | None:
    if not values:
        return None
    values = sorted(values)
    index = (len(values) - 1) * percentile
    low = int(index)
    high = min(low + 1, len(values) - 1)
    return values[low] + (values[high] - values[low]) * (index - low)


async def run_ablation(
    manifest: EvaluationManifest,
    cases: list[dict[str, Any]],
    systems: dict[str, NaiveHybridBaseline],
    *,
    output: Path,
    generation: GenerationRunner | None = None,
) -> Path:
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S") + "-" + uuid4().hex[:12]
    destination = output / run_id
    destination.mkdir(parents=True, exist_ok=False)
    (destination / "manifest.json").write_text(manifest.model_dump_json(indent=2) + "\n")
    report: dict[str, Any] = {
        "run_id": run_id,
        "manifest_id": manifest.manifest_id,
        "boundary": "fixture-replay-production-retrieval",
        "systems": {},
    }
    with get_telemetry().span("evaluation.run", {"request.id": run_id}):
        for name, system in systems.items():
            rows = []
            for case in cases:
                query = QuerySpec.model_validate(case["query"])
                start = time.perf_counter()
                context = await system.run(query)
                metrics = {metric: None for metric in METRICS}
                metrics.update(context_metrics(context, case))
                generated = await generation(query, context) if generation else None
                duration = (time.perf_counter() - start) * 1000
                generation_status = "unmeasured"
                if isinstance(generated, dict):
                    status = generated.get("status")
                    if status in {"answer", "abstained"}:
                        generation_status = status
                        metrics["abstention_rate"] = float(status == "abstained")
                    metrics.update(deterministic_generation_metrics(generated, case))
                    # Only finite, verifier-owned measured metrics may cross this seam.
                    metrics.update(measured_generation_metrics(generated))
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "metrics": metrics,
                        "retrieval_latency_ms": duration,
                        "warnings": context.warnings,
                        "graph_path_ids": [p.path_id for p in context.graph_paths],
                        "evidence_ids": [
                            e.evidence_id for e in context.text_evidence + context.table_evidence
                        ],
                        "generation_status": generation_status,
                    }
                )
            summary = {}
            for metric in METRICS:
                measured = [
                    row["metrics"][metric] for row in rows if row["metrics"][metric] is not None
                ]
                summary[metric] = {
                    "value": sum(measured) / len(measured) if measured else None,
                    "measured_cases": len(measured),
                    "total_cases": len(rows),
                }
            # Verified-answer latency is intentionally unmeasured without a generation runner.
            if generation:
                latencies = [row["retrieval_latency_ms"] for row in rows]
                for metric, point in (("p50_latency_ms", 0.5), ("p95_latency_ms", 0.95)):
                    summary[metric] = {
                        "value": percentile(latencies, point),
                        "measured_cases": len(rows),
                        "total_cases": len(rows),
                    }
            report["systems"][name] = {
                "cases": rows,
                "summary": summary,
                "retrieval_p50_ms": percentile([row["retrieval_latency_ms"] for row in rows], 0.5),
            }
    (destination / "report.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return destination


def fixture_systems(corpus: Path, budget: RetrievalBudget | None = None):
    fixture = load_fixture(corpus)
    return {
        "naive_hybrid": NaiveHybridBaseline(
            RetrievalService(
                FixtureGraph([], filter_entity_ids=True), fixture.vector(), budget=budget
            )
        ),
        "fingraph_rag": GraphRagSystem(
            RetrievalService(fixture.graph(filter_entity_ids=True), fixture.vector(), budget=budget)
        ),
    }


def load_cases(dataset: Path):
    return [json.loads(line) for line in dataset.read_text().splitlines() if line.strip()]


async def extractive_generation(query: QuerySpec, context: FusedContext) -> dict[str, Any]:
    """Exercise actual calculator, extractive generation, and fail-closed verifier.

    This is deterministic application generation, not an LLM-as-judge score.
    Missing expected numerical answers and infrastructure cost stay unmeasured.
    """
    from fingraph_rag.generation.generator import ExtractiveGenerator, prepare_generation_context
    from fingraph_rag.guardrails.pipeline import GuardrailPipeline

    policies = GuardrailPipeline()
    if any(
        decision.action in {"reject", "abstain"}
        for decision in policies.evaluate_input(query.question, query)
    ):
        return {"status": "abstained", "measured_metrics": {}}
    with get_telemetry().span("calculation.execute"):
        generation = prepare_generation_context(context)
    with get_telemetry().span("generation.llm", {"model.name": "extractive-local-v1"}):
        candidates = ExtractiveGenerator().generate(generation)
    with get_telemetry().span("verification.faithfulness"):
        claims = policies.verify_claims(candidates, generation)
    verified = [claim for claim in claims if claim.verification == "verified"]
    candidates_by_id = {candidate.claim_id: candidate for candidate in candidates}
    evidence = generation.evidence()
    verified_claims = []
    for claim in verified:
        candidate = candidates_by_id[claim.claim_id]
        verified_claims.append(
            {
                "text": claim.text,
                "citation_doc_ids": [citation.document.doc_id for citation in claim.citations],
                "temporal_contexts": [
                    evidence[evidence_id].temporal.model_dump(mode="json")
                    for evidence_id in candidate.evidence_ids
                    if evidence_id in evidence
                ],
            }
        )
    return {
        "status": "answer" if claims and len(verified) == len(claims) else "abstained",
        "measured_metrics": {"faithfulness": fraction(len(verified), len(claims))},
        "verified_claim_count": len(verified),
        "verified_claims": verified_claims,
    }
