"""Fail-closed query-time compatibility shared by graph and document evidence."""

from fingraph_rag.domain.contracts import PeriodConstraint, QuerySpec, TemporalContext


def matches_period(value: TemporalContext, period: PeriodConstraint) -> bool:
    for name in ("fiscal_year", "fiscal_quarter", "restatement_version"):
        expected = getattr(period, name)
        if expected is not None and getattr(value, name) != expected:
            return False
    if period.as_of_date:
        if value.filing_date > period.as_of_date:
            return False
        if value.valid_from and value.valid_from > period.as_of_date:
            return False
        if value.valid_to and value.valid_to < period.as_of_date:
            return False
    if period.period_start and (
        value.period_start is None or value.period_start < period.period_start
    ):
        return False
    if period.period_end and (value.period_end is None or value.period_end > period.period_end):
        return False
    return True


def compatible(value: TemporalContext, query: QuerySpec) -> bool:
    if query.answer_mode == "reported_as_filed" and value.restatement_version != 0:
        return False
    return not query.periods or any(matches_period(value, period) for period in query.periods)
