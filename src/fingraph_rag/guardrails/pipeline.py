"""One fail-closed boundary from unverified candidates to public claims."""

from collections.abc import Sequence

from fingraph_rag.domain.contracts import GuardrailDecision, QuerySpec, VerifiedClaim
from fingraph_rag.generation.citation_renderer import CitationError, CitationResolver
from fingraph_rag.generation.contracts import CandidateClaim, GenerationContext
from fingraph_rag.guardrails.compliance import CompliancePolicy
from fingraph_rag.guardrails.faithfulness import GraphFaithfulnessVerifier
from fingraph_rag.guardrails.input import InputPolicy
from fingraph_rag.guardrails.numerical import NumericGroundingVerifier, numeric_tokens
from fingraph_rag.retrieval.contracts import FusedContext


class GuardrailPipeline:
    def evaluate_input(
        self, question: str, query: QuerySpec | None = None
    ) -> list[GuardrailDecision]:
        return [InputPolicy().evaluate(question, query), CompliancePolicy().evaluate(question)]

    def verify_claims(
        self, candidates: Sequence[CandidateClaim], generation: GenerationContext
    ) -> list[VerifiedClaim]:
        output = []
        seen = set()
        for claim in candidates[:24]:
            reasons = []
            if claim.claim_id in seen:
                reasons.append("duplicate_claim_id")
            seen.add(claim.claim_id)
            for policy in (NumericGroundingVerifier(), GraphFaithfulnessVerifier()):
                try:
                    reasons.extend(policy.verify(claim, generation).reason_codes)
                except Exception:
                    # Verification is a public boundary. A malformed candidate or
                    # tampered context must abstain, never turn into an error/answer.
                    reasons.append("verification_error")
            try:
                reasons.extend(CompliancePolicy().evaluate(claim.text).reason_codes)
            except Exception:
                reasons.append("compliance_error")
            try:
                citations = CitationResolver().resolve(claim, generation)
            except CitationError as exc:
                reasons.append(str(exc))
            except Exception:
                reasons.append("citation_resolution_error")
                citations = []
            reasons = list(dict.fromkeys(reasons))
            output.append(
                VerifiedClaim(
                    claim_id=claim.claim_id,
                    text="Insufficient verified evidence." if reasons else claim.text,
                    numeric_tokens=[] if reasons else numeric_tokens(claim.text),
                    citations=[] if reasons else citations,
                    verification="abstained" if reasons else "verified",
                    reason_codes=reasons,
                )
            )
        return output

    def verify(
        self, context: FusedContext, candidates: Sequence[CandidateClaim]
    ) -> list[VerifiedClaim]:
        return self.verify_claims(candidates, GenerationContext.from_context(context))
