"""A deterministic baseline that emits only retained evidence candidates."""

import hashlib

from fingraph_rag.domain.contracts import CalculationRequest
from fingraph_rag.generation.calculator import CalculationError, SafeCalculator
from fingraph_rag.generation.citation_renderer import CitationError, CitationResolver
from fingraph_rag.generation.contracts import CandidateClaim, GenerationContext
from fingraph_rag.guardrails.numerical import calculation_text


class ExtractiveGenerator:
    def generate(self, generation: GenerationContext) -> list[CandidateClaim]:
        candidates = []
        paths = generation.context.graph_paths
        for item in (
            []
            if generation.context.query.needs_calculation
            else list(generation.evidence().values())[:24]
        ):
            matching = [
                p.path_id
                for p in paths
                if any(
                    item.locator == locator and rel.status == "asserted"
                    for rel in p.relationships
                    for locator in rel.locators
                )
            ]
            if not matching:
                continue
            claim = CandidateClaim(
                claim_id=hashlib.sha256(item.evidence_id.encode()).hexdigest(),
                text=item.text,
                evidence_ids=[item.evidence_id],
                graph_path_ids=[matching[0]],
            )
            candidates.append(claim)
        for key, result in list(generation.calculations.items())[:8]:
            claim = CandidateClaim(
                claim_id=f"calculation-{key}",
                text=calculation_text(result),
                evidence_ids=result.input_evidence_ids,
                graph_path_ids=[
                    p.path_id
                    for p in paths
                    if any(
                        rel.locators
                        and any(
                            generation.evidence().get(eid) is not None
                            and generation.evidence()[eid].locator in rel.locators
                            for eid in result.input_evidence_ids
                        )
                        for rel in p.relationships
                    )
                ],
                calculation_ids=[key],
            )
            try:
                CitationResolver().resolve(claim, generation)
            except CitationError:
                continue
            candidates.append(claim)
        return candidates[:24]


def prepare_generation_context(context) -> GenerationContext:
    """Resolve an unambiguous growth request; ambiguous formulas abstain.

    More complex formulas use explicit CalculationRequest objects from a trusted
    caller. This function does not guess numerator, denominator, or currencies.
    """
    generation = GenerationContext.from_context(context)
    if not context.query.needs_calculation:
        return generation
    question = context.query.question.casefold()
    formula = None
    if "cagr" in question:
        formula = "cagr"
    elif "qoq" in question or "quarter-over-quarter" in question:
        formula = "qoq_growth"
    elif any(term in question for term in ("yoy", "year-over-year", "growth")):
        formula = "yoy_growth"
    if formula is None:
        generation.calculation_error = "ambiguous_calculation_formula"
        return generation
    # Multiple chunks may cite one cell. Deduplicate the observation, then
    # require exactly two comparable observations and one metric vocabulary.
    unique = {}
    for evidence_id, metric in sorted(generation.metrics.items()):
        unique.setdefault(metric.model_dump_json(), (evidence_id, metric))
    items = list(unique.values())
    if len(items) != 2 or len({m.metric_name.casefold() for _, m in items}) != 1:
        generation.calculation_error = "ambiguous_calculation_inputs"
        return generation
    quarter_order = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4, "FY": 5, None: 0}
    items.sort(
        key=lambda pair: (
            pair[1].temporal.fiscal_year or 0,
            quarter_order[pair[1].temporal.fiscal_quarter],
        ),
        reverse=True,
    )
    request = CalculationRequest(
        formula_id=formula, inputs=[key for key, _ in items], output_unit="%"
    )
    try:
        result = SafeCalculator().calculate(request, generation.metrics)
    except CalculationError as exc:
        generation.calculation_error = str(exc)
        return generation
    key = result.reproducibility_hash
    generation.calculations[key] = result
    generation.calculation_requests[key] = request
    return generation
