"""Ongoing performance monitoring for market risk desks."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from market_risk.backtesting import BacktestResult
from market_risk.es import ESResult
from market_risk.pla import PLATestResult
from market_risk.var import VaRResult


@dataclass(frozen=True)
class DailyMetrics:
    asof: date
    var_result: VaRResult
    es_result: ESResult
    backtest: Optional[BacktestResult]
    pla: Optional[PLATestResult]


@dataclass
class PerformanceMonitor:
    desk: str
    history: List[DailyMetrics] = field(default_factory=list)

    def add_metrics(self, metrics: DailyMetrics) -> None:
        self.history.append(metrics)

    def latest(self) -> Optional[DailyMetrics]:
        return self.history[-1] if self.history else None

    def summary(self) -> Dict[str, float | str]:
        latest = self.latest()
        if latest is None:
            return {"desk": self.desk, "status": "no data"}
        return {
            "desk": self.desk,
            "var": latest.var_result.value,
            "es": latest.es_result.value,
            "backtest": latest.backtest.traffic_light if latest.backtest else "n/a",
            "pla": latest.pla.status if latest.pla else "n/a",
        }
