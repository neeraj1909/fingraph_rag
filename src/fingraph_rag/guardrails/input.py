"""Bounded input screening with explicit, inspectable refusal reasons."""

import re

from fingraph_rag.domain.contracts import GuardrailDecision, QuerySpec

POLICY_VERSION = "guardrails-1"
INJECTION = re.compile(
    r"ignore\s+(?:all\s+)?(?:previous|prior|system|above)\s+(?:instructions|prompts)|"
    r"(?:reveal|print|show|override)\s+(?:the\s+)?system\s+prompt|"
    r"\b(?:jailbreak|developer\s+mode)\b|<\s*/?\s*(?:system|assistant|tool)\s*>|"
    r"\b(?:execute|run)\s+(?:shell|python|bash|javascript)\b",
    re.I,
)
PII = re.compile(
    r"\b\d{3}-\d{2}-\d{4}\b|\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b|"
    r"\b(?:social security|passport|bank account|credit card)\s*(?:number|no\.?|:)\b",
    re.I,
)
MNPI = re.compile(
    r"\b(?:material\s+non[- ]public|mnpi|insider\s+information|"
    r"unpublished\s+(?:earnings|results)|confidential\s+(?:merger|acquisition|earnings))\b",
    re.I,
)
DOMAIN = re.compile(
    r"\b(?:revenue|profit|earnings|income|margin|ratio|growth|cagr|yoy|qoq|"
    r"stock|shares?|financial|filing|10-k|10-q|company|subsidiar\w*|supplier|"
    r"customer|debt|cash|balance|asset|liabilit\w*|fiscal|FY\d{4}|risk|"
    r"ownership|owns?|stake|supply|acquisition|merger|ebitda|dividend|"
    r"investment|invest|buy|sell|portfolio|coverage|leverage)\b",
    re.I,
)


def decision(stage: str, reasons: list[str], *, reject: str = "reject") -> GuardrailDecision:
    return GuardrailDecision(
        stage=stage,
        action=reject if reasons else "allow",
        reason_codes=reasons,
        policy_version=POLICY_VERSION,
    )


class InputPolicy:
    """Conservative local screening; production needs provenance/tenant policy too."""

    def evaluate(self, question: str, query: QuerySpec | None = None) -> GuardrailDecision:
        if not isinstance(question, str):
            return decision("input", ["invalid_input_type"])
        reasons = []
        if not question.strip() or len(question) > 8000:
            reasons.append("invalid_input_length")
        if INJECTION.search(question):
            reasons.append("prompt_injection")
        if PII.search(question):
            reasons.append("sensitive_personal_data")
        if MNPI.search(question):
            reasons.append("suspected_mnpi")
        if not DOMAIN.search(question):
            reasons.append("off_domain")
        if query and (
            len(query.entities) > 16 or len(query.periods) > 8 or query.required_hops > 4
        ):
            reasons.append("query_budget_exceeded")
        return decision("input", reasons)
