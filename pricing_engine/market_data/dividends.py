"""Dividend structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Sequence

from pricing_engine.market_data.curves import Curve


@dataclass(frozen=True)
class Dividend:
    ex_date: date
    amount: float


@dataclass(frozen=True)
class DividendSchedule:
    discrete: Sequence[Dividend] = ()
    continuous_yield: float = 0.0

    def pv_discrete(self, discount_curve: Curve, t0: float, year_frac) -> float:
        pv = 0.0
        for div in self.discrete:
            t = year_frac(t0, div.ex_date)
            pv += div.amount * discount_curve.df(t)
        return pv

    def yield_rate(self, t: float) -> float:
        return self.continuous_yield
