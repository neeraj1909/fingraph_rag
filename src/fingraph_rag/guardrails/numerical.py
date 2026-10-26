"""Inspect numbers in actual claim text; never trust model numeric-token lists."""

import re
from decimal import Decimal

from fingraph_rag.domain.contracts import GuardrailDecision
from fingraph_rag.generation.calculator import CalculationError, SafeCalculator
from fingraph_rag.generation.contracts import CandidateClaim, GenerationContext
from fingraph_rag.guardrails.input import decision

NUMBER = re.compile(r"(?<![\w.])[-+]?\d+(?:,\d{3})*(?:\.\d+)?%?(?![\w.])")


def numeric_tokens(text: str) -> list[str]:
    return NUMBER.findall(text)


def calculation_text(result) -> str:
    return f"{result.formula_id}: {result.output_value} {result.output_unit}."


class NumericGroundingVerifier:
    def verify(self, claim: CandidateClaim, generation: GenerationContext) -> GuardrailDecision:
        evidence = generation.evidence()
        source = " ".join(evidence[key].text for key in claim.evidence_ids if key in evidence)
        reasons = []
        for key in claim.calculation_ids:
            result = generation.calculations.get(key)
            request = generation.calculation_requests.get(key)
            if result is None or request is None:
                reasons.append("unverifiable_calculation")
                continue
            try:
                expected = SafeCalculator().calculate(request, generation.metrics)
            except CalculationError:
                reasons.append("invalid_calculation_inputs")
                continue
            if expected != result:
                reasons.append("calculation_result_mismatch")
                continue
            for evidence_id in request.inputs:
                item = evidence.get(evidence_id)
                metric = generation.metrics.get(evidence_id)
                # Numeric metadata must retain the same cited locator and the value
                # must be visible in the retained source, not invented by a caller.
                if (
                    item is None
                    or metric is None
                    or item.locator != metric.locator
                    or metric.value_decimal
                    not in {
                        Decimal(token.rstrip("%").replace(",", ""))
                        for token in numeric_tokens(item.text)
                    }
                ):
                    reasons.append("calculation_input_not_source_grounded")
            source += " " + calculation_text(result)
        allowed = set(numeric_tokens(source))
        if any(token not in allowed for token in numeric_tokens(claim.text)):
            reasons.append("unsupported_number")
        return decision("output", list(dict.fromkeys(reasons)), reject="abstain")
