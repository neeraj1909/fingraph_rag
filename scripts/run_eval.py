#!/usr/bin/env python3
"""Run fixture-backed ablations; unmeasured provider metrics remain explicit."""

import argparse
import asyncio
from pathlib import Path

from fingraph_rag.evaluation_service import (
    EvaluationManifest,
    extractive_generation,
    fixture_systems,
    load_cases,
    run_ablation,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("evals/datasets/smoke.jsonl"))
    parser.add_argument("--corpus", type=Path, default=Path("tests/fixtures/graph/sample.json"))
    parser.add_argument("--systems", nargs="+", default=["naive_hybrid", "fingraph_rag"])
    parser.add_argument("--generation", choices=["none", "extractive"], default="extractive")
    parser.add_argument("--output", type=Path, default=Path("reports/evaluations"))
    args = parser.parse_args()
    systems = fixture_systems(args.corpus)
    unknown = set(args.systems) - systems.keys()
    if unknown:
        parser.error("unsupported systems: " + ",".join(sorted(unknown)))
    manifest = EvaluationManifest.create(args.dataset, args.corpus)
    path = asyncio.run(
        run_ablation(
            manifest,
            load_cases(args.dataset),
            {name: systems[name] for name in args.systems},
            output=args.output,
            generation=extractive_generation if args.generation == "extractive" else None,
        )
    )
    print(path)


if __name__ == "__main__":
    main()
