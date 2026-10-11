"""Fiscal-period normalization with explicit issuer calendars."""

from __future__ import annotations

import re
from calendar import monthrange
from datetime import date, timedelta
from typing import Literal

from pydantic import Field

from fingraph_rag.domain.contracts import ContractModel, TemporalContext


class FiscalCalendar(ContractModel):
    issuer_id: str = Field(min_length=1)
    fiscal_year_end_month: int = Field(ge=1, le=12)
    fiscal_year_end_day: int = Field(ge=1, le=31)


class RawPeriod(ContractModel):
    label: str = Field(min_length=1)
    restatement_version: int = Field(default=0, ge=0)


class TemporalNormalizer:
    """Normalize FY/Q labels using the issuer's declared fiscal year end."""

    def normalize(
        self,
        raw_period: RawPeriod,
        *,
        calendar: FiscalCalendar,
        filing_date: date,
    ) -> TemporalContext:
        fiscal_year, quarter = self._parse_label(raw_period.label)
        fiscal_year_end = self._fiscal_year_end(fiscal_year, calendar)

        if quarter == "FY":
            period_end = fiscal_year_end
            previous_end = self._shift_months(period_end, -12)
        else:
            quarter_number = int(quarter[1:])
            period_end = self._shift_months(fiscal_year_end, -(4 - quarter_number) * 3)
            previous_end = self._shift_months(period_end, -3)

        period_start = previous_end + timedelta(days=1)
        if filing_date < period_end:
            raise ValueError("filing_date cannot precede the normalized period end")

        return TemporalContext(
            fiscal_year=fiscal_year,
            fiscal_quarter=quarter,
            filing_date=filing_date,
            period_start=period_start,
            period_end=period_end,
            restatement_version=raw_period.restatement_version,
        )

    @staticmethod
    def _parse_label(label: str) -> tuple[int, Literal["Q1", "Q2", "Q3", "Q4", "FY"]]:
        compact = re.sub(r"\s+", "", label.upper())
        match = re.fullmatch(r"FY?(20\d{2})", compact)
        if match:
            return int(match.group(1)), "FY"
        match = re.fullmatch(r"Q([1-4])(20\d{2})", compact)
        if match:
            return int(match.group(2)), f"Q{match.group(1)}"  # type: ignore[return-value]
        match = re.fullmatch(r"(20\d{2})Q([1-4])", compact)
        if match:
            return int(match.group(1)), f"Q{match.group(2)}"  # type: ignore[return-value]
        raise ValueError(f"unsupported fiscal period label: {label}")

    @staticmethod
    def _fiscal_year_end(fiscal_year: int, calendar: FiscalCalendar) -> date:
        day = min(
            calendar.fiscal_year_end_day,
            monthrange(fiscal_year, calendar.fiscal_year_end_month)[1],
        )
        return date(fiscal_year, calendar.fiscal_year_end_month, day)

    @staticmethod
    def _shift_months(value: date, months: int) -> date:
        absolute_month = value.year * 12 + value.month - 1 + months
        year, month_index = divmod(absolute_month, 12)
        month = month_index + 1
        day = min(value.day, monthrange(year, month)[1])
        return date(year, month, day)
