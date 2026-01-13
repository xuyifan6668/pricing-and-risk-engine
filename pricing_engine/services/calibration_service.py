"""Calibration service skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Hashable, Sequence

from pricing_engine.calibration.calibration import CalibrationQuote, Calibrator
from pricing_engine.calibration.results import CalibrationResult
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model
from pricing_engine.services.cache import CacheEntry, SimpleCache


@dataclass
class CalibrationService:
    cache: SimpleCache
    calibrator: Calibrator

    def _cache_key(self, model: Model, market: MarketDataSnapshot) -> Hashable:
        return (model.name, market.snapshot_id)

    def calibrate(
        self,
        model: Model,
        market: MarketDataSnapshot,
        quotes: Sequence[CalibrationQuote],
    ) -> CalibrationResult:
        key = self._cache_key(model, market)
        cached = self.cache.get(key)
        if cached is not None:
            return cached.value
        result = self.calibrator.calibrate(model, market, quotes)
        self.cache.set(key, CacheEntry(value=result))
        return result
