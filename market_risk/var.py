"""Historical VaR calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from market_risk.utils import quantile


@dataclass(frozen=True)
class VaRResult:
    level: float
    horizon_days: int
    value: float
    pnl_distribution: Sequence[float]


class HistoricalVaRCalculator:
    def compute(self, pnls: Sequence[float], level: float, horizon_days: int) -> VaRResult:
        if not pnls:
            return VaRResult(level=level, horizon_days=horizon_days, value=0.0, pnl_distribution=[])
        losses = [-p for p in pnls]
        var = quantile(losses, level)
        return VaRResult(level=level, horizon_days=horizon_days, value=var, pnl_distribution=pnls)
