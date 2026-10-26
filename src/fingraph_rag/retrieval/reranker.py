"""Injectable reranking seam with an explicit deterministic lexical fallback."""

import re
from collections.abc import Sequence
from typing import Protocol

from fingraph_rag.retrieval.contracts import TextEvidence


class Reranker(Protocol):
    def rerank(self, question: str, evidence: Sequence[TextEvidence]) -> list[TextEvidence]: ...


class LexicalReranker:
    """Baseline only; this is not presented as a trained cross-encoder."""

    def rerank(self, question: str, evidence: Sequence[TextEvidence]) -> list[TextEvidence]:
        terms = set(re.findall(r"\w+", question.casefold()))

        def key(item: TextEvidence) -> tuple[float, str]:
            overlap = len(terms & set(re.findall(r"\w+", item.text.casefold())))
            return (-overlap, item.evidence_id)

        return sorted(evidence, key=key)
