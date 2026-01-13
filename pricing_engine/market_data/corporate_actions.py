"""Corporate actions handling."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class CorporateAction:
    action_date: date


@dataclass(frozen=True)
class Split(CorporateAction):
    ratio: float


@dataclass(frozen=True)
class SpecialDividend(CorporateAction):
    amount: float
