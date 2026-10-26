"""Conservative semantic boundary: exact evidence or recomputed formula output.

This deliberately refuses unrestricted paraphrases. Number overlap and a valid
path alone cannot prove that an arbitrary sentence is entailed by its sources.
"""

from fingraph_rag.domain.contracts import GuardrailDecision
from fingraph_rag.generation.contracts import CandidateClaim, GenerationContext
from fingraph_rag.guardrails.input import INJECTION, MNPI, PII, decision
from fingraph_rag.guardrails.numerical import calculation_text
from fingraph_rag.retrieval.temporal import compatible


class GraphFaithfulnessVerifier:
    def verify(self, claim: CandidateClaim, generation: GenerationContext) -> GuardrailDecision:
        evidence = generation.evidence()
        texts = [evidence[key].text for key in claim.evidence_ids if key in evidence]
        canonical = [
            calculation_text(generation.calculations[key])
            for key in claim.calculation_ids
            if key in generation.calculations
        ]
        reasons = []
        query = generation.context.query
        selected = [evidence[key] for key in claim.evidence_ids if key in evidence]
        paths = {path.path_id: path for path in generation.context.graph_paths}
        if any(not compatible(item.temporal, query) for item in selected):
            reasons.append("wrong_claim_period")
        if query.entities and any(
            not set(item.entity_ids) & set(query.entities) for item in selected
        ):
            reasons.append("wrong_claim_entity")
        if any(
            rel.status != "asserted" or not compatible(rel.temporal, query)
            for key in claim.graph_path_ids
            if key in paths
            for rel in paths[key].relationships
        ):
            reasons.append("invalid_or_out_of_period_graph_path")
        if claim.text not in texts + canonical:
            reasons.append("unsupported_claim_text")
        if claim.calculation_ids and claim.text not in canonical:
            reasons.append("calculation_claim_mismatch")
        if INJECTION.search(claim.text):
            reasons.append("untrusted_source_instruction")
        if PII.search(claim.text) or MNPI.search(claim.text):
            reasons.append("sensitive_output")
        return decision("output", reasons, reject="abstain")
