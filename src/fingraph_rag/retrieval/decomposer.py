"""Bounded route plans without model-generated executable query text."""

from fingraph_rag.domain.contracts import CypherQueryPlan, QuerySpec
from fingraph_rag.retrieval.contracts import RetrievalBudget


class QueryDecomposer:
    def decompose(self, query: QuerySpec, budget: RetrievalBudget) -> list[CypherQueryPlan]:
        template = {
            "fact_table": "entity_facts",
            "multi_hop": "bounded_paths",
            "cross_period": "temporal_facts",
            "thematic": "thematic_paths",
        }[query.route or "fact_table"]
        if budget.max_hops == 0:
            return []
        plans = []
        for period in query.periods or [None]:
            parameters = {
                "entity_ids": query.entities,
                "max_hops": min(max(query.required_hops, 1), budget.max_hops),
                "limit": budget.candidate_limit,
            }
            if period:
                parameters.update(
                    {
                        key: value
                        for key, value in period.model_dump(mode="json", exclude_none=True).items()
                        if key
                        in {
                            "fiscal_year",
                            "fiscal_quarter",
                            "restatement_version",
                            "period_start",
                            "period_end",
                            "as_of_date",
                        }
                    }
                )
            plans.append(
                CypherQueryPlan(
                    template_id=template,
                    parameter_values=parameters,
                    max_hops=parameters["max_hops"],
                )
            )
        return plans
