#!/usr/bin/env python3
"""Run one bounded, fixture-backed query through the real workflow and verifier."""

import argparse
import asyncio
import sys
from pathlib import Path

from fingraph_rag.application.query_service import QueryService
from fingraph_rag.guardrails.pipeline import GuardrailPipeline
from fingraph_rag.retrieval import RetrievalService
from fingraph_rag.retrieval.fixture_adapter import load_fixture
from fingraph_rag.workflow.generators import ExtractiveGenerationAdapter


async def run(args):
    fixture = load_fixture(args.fixture)
    retrieval = RetrievalService(fixture.graph(), fixture.vector(), aliases=fixture.aliases)
    service = QueryService(
        retrieval,
        ExtractiveGenerationAdapter(),
        GuardrailPipeline(),
        state_directory=str(args.state_directory),
    )
    await service.open()
    try:
        request_id = service.submit(args.question)
        events = [event async for event in service.stream(request_id)]
    finally:
        await service.close()

    for event in events:
        print(event.model_dump_json())
    terminal = events[-1].payload.get("outcome") if events else "error"
    answer = next((event for event in events if event.event_type == "answer"), None)
    claims = answer.payload.get("claims", []) if answer else []
    if args.require_citations and (
        not answer or not claims or any(not claim.get("citations") for claim in claims)
    ):
        print("query did not produce citations", file=sys.stderr)
        return 2
    if args.expect_abstention:
        return 0 if terminal == "abstain" else 2
    return 0 if terminal == "answer" else 2 if terminal == "abstain" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--state-directory",
        type=Path,
        default=Path(".fingraph-state/query-cli"),
        help="directory for bounded workflow checkpoints and replay events",
    )
    parser.add_argument(
        "--require-citations",
        action="store_true",
        help="return non-zero unless every answer claim has a citation",
    )
    parser.add_argument(
        "--expect-abstention",
        action="store_true",
        help="return zero only when the verifier/guardrails abstain",
    )
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
