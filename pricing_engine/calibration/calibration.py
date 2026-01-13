"""Calibration routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from pricing_engine.calibration.results import CalibrationResult
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model


@dataclass(frozen=True)
class CalibrationQuote:
    expiry: float
    strike: float
    market_iv: float
    weight: float = 1.0


class Calibrator:
    """Base calibrator skeleton."""

    def calibrate(
        self,
        model: Model,
        market: MarketDataSnapshot,
        quotes: Sequence[CalibrationQuote],
    ) -> CalibrationResult:
        params = model.initial_guess(market)
        return CalibrationResult(status="ok", params=params, diagnostics={"iterations": 0})
