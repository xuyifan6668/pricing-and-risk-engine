"""Pricing service with caching and concurrency hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Hashable

from pricing_engine.api import PricingResult, price
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine
from pricing_engine.products.base import Product
from pricing_engine.services.cache import CacheEntry, SimpleCache
from pricing_engine.api import PricingSettings


@dataclass
class PricingService:
    cache: SimpleCache

    def _cache_key(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        engine: Engine,
        settings: PricingSettings,
    ) -> Hashable:
        return (
            product.product_type,
            product.maturity,
            model.name,
            engine.name,
            market.snapshot_id,
            settings.engine_settings.deterministic,
        )

    def price(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        engine: Engine,
        settings: PricingSettings,
    ) -> PricingResult:
        key = self._cache_key(product, market, model, engine, settings)
        cached = self.cache.get(key)
        if cached is not None:
            return cached.value
        result = price(product, market, model, engine, settings)
        self.cache.set(key, CacheEntry(value=result))
        return result
