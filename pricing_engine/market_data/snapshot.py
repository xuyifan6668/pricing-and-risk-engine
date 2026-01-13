"""Market data snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Sequence
from uuid import uuid4
import math

from pricing_engine.market_data.corporate_actions import CorporateAction
from pricing_engine.market_data.curves import Curve
from pricing_engine.market_data.dividends import DividendSchedule
from pricing_engine.market_data.surfaces import VolSurface
from pricing_engine.utils.date import year_fraction


@dataclass(frozen=True)
class MarketDataSnapshot:
    asof: date
    spot: float
    discount_curve: Curve
    funding_curve: Curve
    borrow_curve: Curve
    vol_surface: VolSurface
    dividends: DividendSchedule = DividendSchedule()
    corporate_actions: Sequence[CorporateAction] = ()
    snapshot_id: str = ""

    def __post_init__(self) -> None:
        if not self.snapshot_id:
            object.__setattr__(self, "snapshot_id", str(uuid4()))

    def time_to(self, maturity: date) -> float:
        return year_fraction(self.asof, maturity)

    def forward_price(self, maturity: date) -> float:
        t = self.time_to(maturity)
        r = self.funding_curve.zero_rate(t)
        q = self.dividends.yield_rate(t)
        b = self.borrow_curve.zero_rate(t)
        return self.spot * math.exp((r - q - b) * t)
