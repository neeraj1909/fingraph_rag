"""Finite allowlisted Decimal arithmetic over identified source observations.

Binary formulas take [current/numerator, prior/denominator]. Growth outputs are
percentages. Currency conversion is deliberately unsupported: callers must
supply comparable source observations rather than assume an exchange rate.
"""

import hashlib
import json
from collections.abc import Mapping
from decimal import (
    ROUND_CEILING,
    ROUND_FLOOR,
    ROUND_HALF_EVEN,
    ROUND_HALF_UP,
    Decimal,
    InvalidOperation,
    localcontext,
)

from fingraph_rag.domain.contracts import CalculationRequest, CalculationResult, MetricObservation

SCALES = {
    "units": Decimal(1),
    "thousands": Decimal(1000),
    "millions": Decimal(1000000),
    "billions": Decimal(1000000000),
}
ROUNDING = {
    "half_even": ROUND_HALF_EVEN,
    "half_up": ROUND_HALF_UP,
    "floor": ROUND_FLOOR,
    "ceil": ROUND_CEILING,
}


class CalculationError(ValueError):
    """A reason-coded refusal, safe to expose without source values."""


class SafeCalculator:
    version = "decimal-1"

    def calculate(
        self, request: CalculationRequest, evidence: Mapping[str, MetricObservation]
    ) -> CalculationResult:
        if request.formula_id not in {
            "yoy_growth",
            "qoq_growth",
            "cagr",
            "margin",
            "leverage",
            "coverage",
            "sum",
            "difference",
            "ratio",
        }:
            raise CalculationError("unsupported_formula")
        if request.rounding not in ROUNDING:
            raise CalculationError("unsupported_rounding")
        if len(request.inputs) > 64 or len(set(request.inputs)) != len(request.inputs):
            raise CalculationError("invalid_input_count")
        if request.formula_id != "sum" and len(request.inputs) != 2:
            raise CalculationError("binary_formula_requires_two_inputs")
        try:
            metrics = [evidence[key] for key in request.inputs]
        except KeyError as exc:
            raise CalculationError("missing_input_evidence") from exc
        if any(m.is_estimate for m in metrics):
            raise CalculationError("estimated_input_not_approved")
        first = metrics[0]
        if any((m.unit, m.currency) != (first.unit, first.currency) for m in metrics):
            raise CalculationError("incompatible_units_or_currency")
        if len({m.temporal.restatement_version for m in metrics}) != 1:
            raise CalculationError("incompatible_restatement_versions")
        if any(not m.value_decimal.is_finite() for m in metrics):
            raise CalculationError("nonfinite_or_excessive_input")
        with localcontext() as scale_context:
            scale_context.prec = 50
            values = [m.value_decimal * SCALES[m.scale] for m in metrics]
        if any(abs(v) > Decimal("1e30") for v in values):
            raise CalculationError("nonfinite_or_excessive_input")
        formula = request.formula_id
        growth = formula in {"yoy_growth", "qoq_growth", "cagr"}
        if growth:
            if any(m.metric_name.casefold() != first.metric_name.casefold() for m in metrics):
                raise CalculationError("growth_metric_mismatch")
            self._validate_growth_period(formula, metrics)
        elif len({self._period(m) for m in metrics}) != 1 or self._period(first) == (
            None,
            None,
            None,
            None,
        ):
            raise CalculationError("incompatible_or_missing_period")
        expected = (
            "%"
            if growth or formula == "margin"
            else (first.unit if formula in {"sum", "difference"} else "ratio")
        )
        if request.output_unit != expected:
            raise CalculationError("invalid_output_unit")
        try:
            with localcontext() as context:
                context.prec = 50
                a = values[0]
                b = values[1] if len(values) > 1 else Decimal(0)
                if (
                    formula
                    in {
                        "yoy_growth",
                        "qoq_growth",
                        "cagr",
                        "ratio",
                        "margin",
                        "leverage",
                        "coverage",
                    }
                    and b == 0
                ):
                    raise CalculationError("zero_denominator")
                if formula == "sum":
                    result = sum(values, Decimal(0))
                elif formula == "difference":
                    result = a - b
                elif formula in {"yoy_growth", "qoq_growth"}:
                    if b <= 0:
                        raise CalculationError("nonpositive_growth_base")
                    result = (a / b - 1) * 100
                elif formula == "cagr":
                    if a <= 0 or b <= 0:
                        raise CalculationError("nonpositive_cagr_input")
                    years = metrics[0].temporal.fiscal_year - metrics[1].temporal.fiscal_year
                    result = ((a / b) ** (Decimal(1) / Decimal(years)) - 1) * 100
                else:
                    result = a / b * (100 if formula == "margin" else 1)
                result = result.quantize(
                    Decimal(1).scaleb(-request.precision), rounding=ROUNDING[request.rounding]
                )
                if not result.is_finite():
                    raise CalculationError("nonfinite_result")
        except (InvalidOperation, OverflowError, ZeroDivisionError) as exc:
            raise CalculationError("invalid_arithmetic") from exc
        material = {
            "version": self.version,
            "request": request.model_dump(mode="json"),
            "inputs": {key: evidence[key].model_dump(mode="json") for key in request.inputs},
            "result": str(result),
        }
        digest = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        return CalculationResult(
            formula_id=formula,
            output_value=result,
            output_unit=expected,
            input_evidence_ids=request.inputs,
            calculator_version=self.version,
            reproducibility_hash=digest,
        )

    @staticmethod
    def _period(metric: MetricObservation) -> tuple:
        t = metric.temporal
        return t.fiscal_year, t.fiscal_quarter, t.period_start, t.period_end

    @staticmethod
    def _validate_growth_period(formula: str, metrics: list[MetricObservation]) -> None:
        current, prior = (m.temporal for m in metrics)
        if current.fiscal_year is None or prior.fiscal_year is None:
            raise CalculationError("missing_fiscal_period")
        if formula == "qoq_growth":
            quarters = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}
            if current.fiscal_quarter not in quarters or prior.fiscal_quarter not in quarters:
                raise CalculationError("quarter_required")
            distance = (
                current.fiscal_year * 4
                + quarters[current.fiscal_quarter]
                - prior.fiscal_year * 4
                - quarters[prior.fiscal_quarter]
            )
            if distance != 1:
                raise CalculationError("nonadjacent_quarters")
        else:
            if current.fiscal_quarter != prior.fiscal_quarter or current.fiscal_quarter is None:
                raise CalculationError("incompatible_fiscal_periods")
            distance = current.fiscal_year - prior.fiscal_year
            if (formula == "yoy_growth" and distance != 1) or (
                formula == "cagr" and (distance <= 0 or distance > 100)
            ):
                raise CalculationError("invalid_year_distance")
            if formula == "cagr" and current.fiscal_quarter != "FY":
                raise CalculationError("cagr_requires_annual_periods")
        if current.period_end and prior.period_end and current.period_end <= prior.period_end:
            raise CalculationError("reversed_periods")
