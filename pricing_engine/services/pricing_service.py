"""Pricing service with caching and concurrency hooks."""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from typing import Hashable, Iterable

from pricing_engine.api import PricingResult, price
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine
from pricing_engine.products.base import Product
from pricing_engine.services.cache import CacheEntry, SimpleCache
from pricing_engine.api import PricingSettings


def _freeze(value: object) -> Hashable:
    if isinstance(value, dict):
        return tuple(sorted((k, _freeze(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return tuple(sorted(_freeze(v) for v in value))
    return value


def _dataclass_payload(obj: object) -> Hashable:
    if is_dataclass(obj):
        return _freeze(asdict(obj))
    return _freeze(obj)


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
        product_payload = (
            _dataclass_payload(product)
            if is_dataclass(product)
            else (product.product_type, _freeze(product.metadata()))
        )
        greek_payload: Iterable[object] | None
        if settings.greek_request is None:
            greek_payload = None
        else:
            greek_payload = _dataclass_payload(settings.greek_request)
        engine_settings_payload = _dataclass_payload(settings.engine_settings)
        return (
            product.product_type,
            model.name,
            _freeze(model.params()),
            engine.name,
            market.snapshot_id,
            engine_settings_payload,
            settings.compute_greeks,
            greek_payload,
            product_payload,
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
