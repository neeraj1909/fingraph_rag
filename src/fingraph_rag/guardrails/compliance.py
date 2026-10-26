"""Refuse personalized investment actions; factual research remains permitted."""

import re

from fingraph_rag.domain.contracts import GuardrailDecision
from fingraph_rag.guardrails.input import decision

ADVICE = re.compile(
    r"\b(?:should\s+(?:I|we|you)\s+(?:buy|sell|hold|invest)|"
    r"(?:should|can|could|would)\s+(?:I|we|you)\s+(?:invest|trade)|"
    r"(?:how\s+much|where|when)\s+should\s+(?:I|we)\s+invest|"
    r"(?:what|which)\s+(?:should|can|could)\s+(?:I|we)\s+(?:buy|sell|hold|invest)|"
    r"(?:buy|sell|hold)\s+(?:this|that|the)\s+(?:stock|share)|"
    r"(?:my|your)\s+(?:portfolio|risk\s+tolerance|retirement|savings)|"
    r"(?:recommend|recommendation|advi[cs]e)\s+(?:me\s+)?(?:to\s+)?(?:buy|sell|hold|invest|an?\s+(?:stock|share|security|investment))|"
    r"(?:which|what)\s+(?:stock|share|security|investment)s?\s+(?:should|can|could)\s+(?:I|we)\s+(?:buy|hold|invest)|"
    r"(?:good|safe|smart)\s+investment\s+for\s+(?:me|my)|"
    r"guaranteed\s+(?:return|profit)|(?:allocate|invest)\s+my)\b",
    re.I,
)


class CompliancePolicy:
    def evaluate(self, text: str) -> GuardrailDecision:
        if not isinstance(text, str):
            return decision("compliance", ["invalid_input_type"], reject="abstain")
        return decision(
            "compliance",
            ["personalized_investment_advice"] if ADVICE.search(text) else [],
            reject="abstain",
        )
