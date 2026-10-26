"""Combine typed evidence using shared pruning and injected reranking."""

from collections.abc import Sequence

from fingraph_rag.domain.contracts import QuerySpec
from fingraph_rag.retrieval.contracts import FusedContext, GraphPath, RetrievalBudget, TextEvidence
from fingraph_rag.retrieval.reranker import LexicalReranker, Reranker
from fingraph_rag.retrieval.subgraph_pruner import SubgraphPruner


class ContextFusion:
    def __init__(self, reranker: Reranker | None = None) -> None:
        self.reranker = reranker or LexicalReranker()

    def fuse(
        self,
        query: QuerySpec,
        paths: Sequence[GraphPath],
        evidence: Sequence[TextEvidence],
        budget: RetrievalBudget,
    ) -> FusedContext:
        return SubgraphPruner().prune(
            query,
            sorted(paths, key=lambda p: p.path_id),
            self.reranker.rerank(query.question, evidence),
            budget,
        )
