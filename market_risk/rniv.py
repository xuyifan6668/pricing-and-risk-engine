"""RNIV add-on for barrier proximity and residual risks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from pricing_engine.api import price, PricingSettings
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.products.exotics import BarrierOption

from market_risk.portfolio import Portfolio


@dataclass(frozen=True)
class RNIVConfig:
    proximity_threshold: float = 0.02
    add_on_multiplier: float = 0.1


def rniv_addon(
    portfolio: Portfolio,
    market: MarketDataSnapshot,
    config: RNIVConfig,
) -> float:
    if market.spot <= 0.0:
        return 0.0
    addon = 0.0
    for pos in portfolio.positions:
        product = pos.product
        if isinstance(product, BarrierOption):
            barrier = product.barrier
            if barrier <= 0.0:
                continue
            proximity = abs(market.spot - barrier) / market.spot
            if proximity <= config.proximity_threshold:
                result = price(product, market, pos.model, pos.engine, pos.pricing_settings)
                addon += config.add_on_multiplier * abs(result.price * pos.quantity)
    return addon
