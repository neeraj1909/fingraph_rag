"""Deterministic route signals; entity aliases are supplied by the caller."""

import re
from collections.abc import Mapping

from fingraph_rag.domain.contracts import PeriodConstraint, QuerySpec


class QueryClassifier:
    def __init__(self, aliases: Mapping[str, str] | None = None) -> None:
        self.aliases = dict(aliases or {})

    def classify(self, question: str) -> QuerySpec:
        text = question.casefold()
        years = list(
            dict.fromkeys(
                int(y) for y in re.findall(r"\b(?:FY)?((?:19|20|21)\d{2})\b", question, re.I)
            )
        )
        quarter = re.search(r"\b(Q[1-4])\b", question, re.I)
        periods = [
            PeriodConstraint(fiscal_year=y, fiscal_quarter=quarter[1].upper() if quarter else None)
            for y in years
        ]
        comparison = (
            bool(re.search(r"\b(yoy|qoq|cagr|growth|compare|versus|vs|change|difference)\b", text))
            or len(years) > 1
        )
        relation = bool(
            re.search(
                r"\b(through|supplier|subsidiary|impact|affect|ownership|owns|supplies)\b", text
            )
        )
        theme = bool(re.search(r"\b(theme|sector|landscape|overview|risks|risk|summary)\b", text))
        route = (
            "cross_period"
            if comparison
            else "multi_hop"
            if relation
            else "thematic"
            if theme
            else "fact_table"
        )
        entities = list(
            dict.fromkeys(
                entity
                for alias, entity in self.aliases.items()
                if re.search(rf"(?<!\w){re.escape(alias.casefold())}(?!\w)", text)
            )
        )
        metrics = [
            key
            for key in ("revenue", "income", "debt", "margin", "earnings", "cash")
            if re.search(rf"\b{key}\b", text)
        ]
        mode = (
            "both"
            if "both" in text and "restat" in text
            else "reported_as_filed"
            if "as filed" in text
            else "latest_restatement"
        )
        return QuerySpec(
            question=question,
            route=route,
            entities=entities,
            metrics=metrics,
            periods=periods,
            required_hops=4 if relation else 1,
            needs_calculation=comparison,
            answer_mode=mode,
        )
