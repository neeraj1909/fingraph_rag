"""Compiled LangGraph workflow; policies and retrieval are injected boundaries."""

from langgraph.graph import END, START, StateGraph

from fingraph_rag.domain.contracts import QuerySpec
from fingraph_rag.generation.contracts import CandidateClaim, GenerationContext
from fingraph_rag.retrieval.contracts import FusedContext
from fingraph_rag.workflow.routing import verification_route
from fingraph_rag.workflow.state import QueryState


def build_graph(retrieval, generator, policies, events, checkpointer, *, max_repairs=2):
    async def guard(state):
        decisions = policies.evaluate_input(state["question"])
        events.append(
            state["request_id"],
            "guardrail",
            {"decisions": [decision.model_dump(mode="json") for decision in decisions]},
        )
        rejected = any(decision.action in {"reject", "abstain"} for decision in decisions)
        return {"terminal": "abstain" if rejected else "", "repairs": 0}

    async def route(state):
        spec = retrieval.classifier.classify(state["question"])
        events.append(state["request_id"], "route", {"route": spec.route})
        return {"query_spec": spec.model_dump(mode="json")}

    async def retrieve(state):
        context = await retrieval.retrieve(QuerySpec.model_validate(state["query_spec"]))
        events.append(
            state["request_id"],
            "retrieval",
            {
                "evidence_ids": [
                    item.evidence_id for item in context.text_evidence + context.table_evidence
                ],
                "warnings": context.warnings,
            },
        )
        for path in context.graph_paths:
            events.append(
                state["request_id"],
                "graph_path",
                {
                    "path_id": path.path_id,
                    "assertion_ids": [edge.assertion_id for edge in path.relationships],
                },
            )
        return {"context": context.model_dump(mode="json")}

    async def calculate(state):
        from fingraph_rag.generation.generator import prepare_generation_context

        context = prepare_generation_context(FusedContext.model_validate(state["context"]))
        for key, result in context.calculations.items():
            events.append(
                state["request_id"],
                "calculation",
                {"calculation_id": key, "result": result.model_dump(mode="json")},
            )
        return {"generation": context.model_dump(mode="json")}

    async def generate(state):
        context = GenerationContext.model_validate(state["generation"])
        candidates = await generator.generate(context, state.get("repair_reasons", []))
        return {"candidates": [item.model_dump(mode="json") for item in candidates]}

    async def verify(state):
        context = GenerationContext.model_validate(state["generation"])
        candidates = [CandidateClaim.model_validate(item) for item in state["candidates"]]
        claims = policies.verify_claims(candidates, context)
        accepted = bool(claims) and all(item.verification == "verified" for item in claims)
        reasons = sorted({reason for claim in claims for reason in claim.reason_codes})
        events.append(
            state["request_id"],
            "verification",
            {"verified": accepted, "reason_codes": reasons, "repair_count": state["repairs"]},
        )
        return {
            "claims": [claim.model_dump(mode="json") for claim in claims],
            "terminal": "answer" if accepted else "abstain",
            "repair_reasons": reasons,
        }

    async def repair(state):
        return {"repairs": state["repairs"] + 1}

    async def finish(state):
        if state["terminal"] == "answer":
            events.append(state["request_id"], "answer", {"claims": state["claims"]})
        else:
            events.append(
                state["request_id"],
                "guardrail",
                {
                    "action": "abstain",
                    "reason_codes": state.get("repair_reasons")
                    or ["insufficient_verified_evidence"],
                },
            )
        return {}

    graph = StateGraph(QueryState)
    for name, node in {
        "input_guardrail": guard,
        "classify_query": route,
        "retrieve_parallel": retrieve,
        "calculate_if_needed": calculate,
        "generate_claims": generate,
        "verify_claims": verify,
        "repair": repair,
        "finish": finish,
    }.items():

        def instrument(node, span_name):
            async def traced(state):
                from fingraph_rag.observability.tracing import get_telemetry

                with get_telemetry().span(span_name, {"request.id": state["request_id"]}):
                    return await node(state)

            return traced

        span_names = {
            "input_guardrail": "guardrail.input",
            "classify_query": "query.classify",
            "retrieve_parallel": "retrieval.graph",
            "calculate_if_needed": "calculation.execute",
            "generate_claims": "generation.llm",
            "verify_claims": "verification.faithfulness",
            "repair": "query.execute",
            "finish": "sse.stream",
        }
        graph.add_node(name, instrument(node, span_names[name]))
    graph.add_edge(START, "input_guardrail")
    graph.add_conditional_edges(
        "input_guardrail", lambda state: "finish" if state["terminal"] else "classify_query"
    )
    graph.add_edge("classify_query", "retrieve_parallel")
    graph.add_edge("retrieve_parallel", "calculate_if_needed")
    graph.add_edge("calculate_if_needed", "generate_claims")
    graph.add_edge("generate_claims", "verify_claims")
    graph.add_conditional_edges(
        "verify_claims", lambda state: verification_route(state, max_repairs)
    )
    graph.add_edge("repair", "generate_claims")
    graph.add_edge("finish", END)
    return graph.compile(checkpointer=checkpointer)
