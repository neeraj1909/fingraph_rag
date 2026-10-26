from datetime import date
from decimal import Decimal

import pytest

from fingraph_rag.domain.contracts import (
    CalculationRequest,
    EvidenceLocator,
    MetricObservation,
    TemporalContext,
)
from fingraph_rag.generation.calculator import CalculationError, SafeCalculator

pytestmark = pytest.mark.unit


def metric(value="120", year=2025, quarter="FY", **overrides):
    values = dict(
        metric_id=f"revenue-{year}",
        metric_name="revenue",
        value_decimal=Decimal(value),
        unit="USD",
        currency="USD",
        scale="millions",
        statement="income_statement",
        temporal=TemporalContext(
            fiscal_year=year, fiscal_quarter=quarter, filing_date=date(year + 1, 2, 1)
        ),
        locator=EvidenceLocator(
            doc_id=f"doc-{year}",
            filing_date=date(year + 1, 2, 1),
            page=1,
            table_id="table-1",
            cell_range="B2",
            quote_hash="a" * 64,
        ),
    )
    values.update(overrides)
    return MetricObservation(**values)


def request(formula="yoy_growth", **overrides):
    values = dict(formula_id=formula, inputs=["current", "prior"], output_unit="%")
    values.update(overrides)
    return CalculationRequest(**values)


def test_growth_is_scaled_cited_reproducible_decimal():
    current = metric("0.12", scale="billions")
    prior = metric("100", 2024)
    evidence = {"current": current, "prior": prior}
    calculator = SafeCalculator()
    result = calculator.calculate(request(), evidence)
    assert result.output_value == Decimal("20.00")
    assert result.input_evidence_ids == ["current", "prior"]
    assert calculator.calculate(request(), dict(reversed(list(evidence.items())))) == result
    changed = dict(evidence, current=metric("121"))
    assert (
        calculator.calculate(request(), changed).reproducibility_hash != result.reproducibility_hash
    )


@pytest.mark.parametrize(
    "formula,expected",
    [
        ("ratio", "1.20"),
        ("margin", "120.00"),
        ("leverage", "1.20"),
        ("coverage", "1.20"),
        ("sum", "220000000.00"),
        ("difference", "20000000.00"),
    ],
)
def test_allowlisted_same_period_formulas(formula, expected):
    unit = "%" if formula == "margin" else "USD" if formula in {"sum", "difference"} else "ratio"
    result = SafeCalculator().calculate(
        request(formula, output_unit=unit), {"current": metric(), "prior": metric("100")}
    )
    assert result.output_value == Decimal(expected)


def test_cagr_and_cross_year_qoq():
    result = SafeCalculator().calculate(
        request("cagr"), {"current": metric("121", 2025), "prior": metric("100", 2023)}
    )
    assert result.output_value == Decimal("10.00")
    result = SafeCalculator().calculate(
        request("qoq_growth"),
        {"current": metric("120", 2025, "Q1"), "prior": metric("100", 2024, "Q4")},
    )
    assert result.output_value == Decimal("20.00")


@pytest.mark.parametrize(
    "prior,reason",
    [
        (metric("0", 2024), "zero_denominator"),
        (metric("-1", 2024), "nonpositive_growth_base"),
        (metric("100", 2023), "invalid_year_distance"),
        (metric("100", 2024, "Q3"), "incompatible_fiscal_periods"),
        (metric("100", 2024, currency="EUR"), "incompatible_units_or_currency"),
        (
            metric("100", 2024).model_copy(update={"value_decimal": Decimal("NaN")}),
            "nonfinite_or_excessive_input",
        ),
        (metric("100", 2024, is_estimate=True), "estimated_input_not_approved"),
    ],
)
def test_growth_rejects_unsafe_inputs(prior, reason):
    with pytest.raises(CalculationError, match=reason):
        SafeCalculator().calculate(request(), {"current": metric(), "prior": prior})


def test_missing_duplicate_and_unit_manipulated_inputs_rejected():
    with pytest.raises(CalculationError, match="missing_input_evidence"):
        SafeCalculator().calculate(request(), {})
    with pytest.raises(CalculationError, match="invalid_input_count"):
        SafeCalculator().calculate(request(inputs=["current", "current"]), {"current": metric()})
    with pytest.raises(CalculationError, match="invalid_output_unit"):
        SafeCalculator().calculate(
            request(output_unit="USD"), {"current": metric(), "prior": metric("100", 2024)}
        )


def test_rounding_policy_is_part_of_result_identity():
    evidence = {"current": metric("1.005"), "prior": metric("1")}
    even = SafeCalculator().calculate(request("ratio", output_unit="ratio"), evidence)
    up = SafeCalculator().calculate(
        request("ratio", output_unit="ratio", rounding="half_up"), evidence
    )
    assert even.output_value == Decimal("1.00")
    assert up.output_value == Decimal("1.01")
    assert even.reproducibility_hash != up.reproducibility_hash


def test_runtime_dispatch_remains_allowlisted_for_corrupted_cached_contract():
    unsafe = request().model_copy(update={"formula_id": "__import__('os').system('id')"})
    with pytest.raises(CalculationError, match="unsupported_formula"):
        SafeCalculator().calculate(unsafe, {"current": metric(), "prior": metric("100", 2024)})


def test_corrupted_rounding_contract_fails_closed():
    unsafe = request().model_copy(update={"rounding": "truncate"})
    with pytest.raises(CalculationError, match="unsupported_rounding"):
        SafeCalculator().calculate(unsafe, {"current": metric(), "prior": metric("100", 2024)})
