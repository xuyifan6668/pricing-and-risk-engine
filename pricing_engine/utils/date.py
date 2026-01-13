"""Date utilities."""

from __future__ import annotations

from datetime import date


def year_fraction(start: date, end: date, basis: float = 365.0) -> float:
    return (end - start).days / basis
