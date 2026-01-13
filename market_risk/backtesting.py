"""Backtesting utilities for VaR exceptions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence


@dataclass(frozen=True)
class BacktestResult:
    window: int
    exceptions: int
    traffic_light: str
    exception_indices: List[int]


class Backtester:
    def __init__(self, green: int = 4, yellow: int = 9) -> None:
        self.green = green
        self.yellow = yellow

    def run(self, actual_pnls: Sequence[float], var_series: Sequence[float], window: int = 250) -> BacktestResult:
        if len(actual_pnls) != len(var_series):
            raise ValueError("actual_pnls and var_series must have the same length")
        start = max(0, len(actual_pnls) - window)
        exceptions = 0
        exception_indices: List[int] = []
        for idx in range(start, len(actual_pnls)):
            if actual_pnls[idx] < -var_series[idx]:
                exceptions += 1
                exception_indices.append(idx)
        if exceptions <= self.green:
            traffic = "green"
        elif exceptions <= self.yellow:
            traffic = "yellow"
        else:
            traffic = "red"
        return BacktestResult(window=window, exceptions=exceptions, traffic_light=traffic, exception_indices=exception_indices)
