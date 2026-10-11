from datetime import date

import pytest

from fingraph_rag.ingestion.temporal_normalization import (
    FiscalCalendar,
    RawPeriod,
    TemporalNormalizer,
)


@pytest.mark.unit
def test_calendar_normalizes_fiscal_year_and_quarter() -> None:
    calendar = FiscalCalendar(
        issuer_id="company-acme",
        fiscal_year_end_month=12,
        fiscal_year_end_day=31,
    )
    normalizer = TemporalNormalizer()

    fiscal_year = normalizer.normalize(
        RawPeriod(label="FY2024"),
        calendar=calendar,
        filing_date=date(2025, 2, 28),
    )
    quarter = normalizer.normalize(
        RawPeriod(label="Q3 2024"),
        calendar=calendar,
        filing_date=date(2024, 11, 1),
    )

    assert (fiscal_year.period_start, fiscal_year.period_end) == (
        date(2024, 1, 1),
        date(2024, 12, 31),
    )
    assert (quarter.period_start, quarter.period_end) == (
        date(2024, 7, 1),
        date(2024, 9, 30),
    )


@pytest.mark.unit
def test_non_calendar_fiscal_year_is_respected() -> None:
    calendar = FiscalCalendar(
        issuer_id="company-june",
        fiscal_year_end_month=6,
        fiscal_year_end_day=30,
    )

    quarter = TemporalNormalizer().normalize(
        RawPeriod(label="Q1 2024"),
        calendar=calendar,
        filing_date=date(2023, 11, 1),
    )

    assert (quarter.period_start, quarter.period_end) == (
        date(2023, 7, 1),
        date(2023, 9, 30),
    )


@pytest.mark.unit
def test_restatement_versions_are_additive_for_the_same_period() -> None:
    calendar = FiscalCalendar(
        issuer_id="company-acme",
        fiscal_year_end_month=12,
        fiscal_year_end_day=31,
    )
    normalizer = TemporalNormalizer()

    original = normalizer.normalize(
        RawPeriod(label="FY2024", restatement_version=0),
        calendar=calendar,
        filing_date=date(2025, 2, 28),
    )
    restated = normalizer.normalize(
        RawPeriod(label="FY2024", restatement_version=1),
        calendar=calendar,
        filing_date=date(2025, 2, 28),
    )

    assert (original.period_start, original.period_end) == (
        restated.period_start,
        restated.period_end,
    )
    assert (original.restatement_version, restated.restatement_version) == (0, 1)


@pytest.mark.unit
def test_unknown_period_and_early_filing_are_rejected() -> None:
    calendar = FiscalCalendar(
        issuer_id="company-acme",
        fiscal_year_end_month=12,
        fiscal_year_end_day=31,
    )
    normalizer = TemporalNormalizer()

    with pytest.raises(ValueError, match="unsupported fiscal period"):
        normalizer.normalize(
            RawPeriod(label="FY2024-Q5"),
            calendar=calendar,
            filing_date=date(2025, 2, 28),
        )
    with pytest.raises(ValueError, match="cannot precede"):
        normalizer.normalize(
            RawPeriod(label="FY2024"),
            calendar=calendar,
            filing_date=date(2024, 12, 1),
        )
