"""Expected Shortfall calculation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from market_risk.utils import mean, quantile


@dataclass(frozen=True)
class ESResult:
    level: float
    horizon_days: int
    value: float
    pnl_distribution: Sequence[float]


class ExpectedShortfallCalculator:
    def compute(self, pnls: Sequence[float], level: float, horizon_days: int) -> ESResult:
        if not pnls:
            return ESResult(level=level, horizon_days=horizon_days, value=0.0, pnl_distribution=[])
        losses = [-p for p in pnls]
        var_threshold = quantile(losses, level)
        tail_losses = [loss for loss in losses if loss >= var_threshold]
        es = mean(tail_losses) if tail_losses else 0.0
        return ESResult(level=level, horizon_days=horizon_days, value=es, pnl_distribution=pnls)
