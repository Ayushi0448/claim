"""Date arithmetic for waiting-period evaluation.

Waiting periods decide several claims outright, so the arithmetic is isolated
here, kept dependency-free and unit tested.
"""

from __future__ import annotations

from datetime import date, datetime

_FORMATS = ("%Y-%m-%d", "%d-%m-%Y", "%d/%m/%Y", "%Y/%m/%d", "%d %b %Y", "%d %B %Y")


def parse_date(value: str | date | datetime | None) -> date | None:
    """Parse the dataset's ISO dates, tolerating a few common alternatives."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        return None
    for fmt in _FORMATS:
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    return None


def days_between(start: date | None, end: date | None) -> int | None:
    if start is None or end is None:
        return None
    return (end - start).days


def months_between(start: date | None, end: date | None) -> float | None:
    """Elapsed months as a float, using completed months plus a day fraction.

    The policy speaks in *completed* months/years for waiting periods, so
    callers that need a completed count should floor this value.
    """
    if start is None or end is None:
        return None
    months = (end.year - start.year) * 12 + (end.month - start.month)
    if end.day < start.day:
        months -= 1
        # Approximate the partial month using a 30-day month.
        prev_month_days = 30
        frac = (end.day + prev_month_days - start.day) / prev_month_days
    else:
        frac = (end.day - start.day) / 30.0
    return round(months + max(0.0, min(frac, 0.999)), 3)


def completed_months(start: date | None, end: date | None) -> int | None:
    m = months_between(start, end)
    return None if m is None else int(m)


def completed_years(months: float | None) -> int | None:
    return None if months is None else int(months // 12)
