"""Portfolio and positions for market risk."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Sequence

from pricing_engine.api import PricingSettings, price
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine
from pricing_engine.products.base import Product


@dataclass(frozen=True)
class Position:
    product: Product
    model: Model
    engine: Engine
    quantity: float = 1.0
    pricing_settings: PricingSettings = PricingSettings()
    name: str = ""


@dataclass
class Portfolio:
    positions: List[Position] = field(default_factory=list)

    def add(self, position: Position) -> None:
        self.positions.append(position)

    def value(self, market: MarketDataSnapshot) -> float:
        total = 0.0
        for pos in self.positions:
            result = price(pos.product, market, pos.model, pos.engine, pos.pricing_settings)
            total += pos.quantity * result.price
        return total

    def position_values(self, market: MarketDataSnapshot) -> List[float]:
        values = []
        for pos in self.positions:
            result = price(pos.product, market, pos.model, pos.engine, pos.pricing_settings)
            values.append(pos.quantity * result.price)
        return values

    def scenario_pnls(
        self,
        base_market: MarketDataSnapshot,
        scenarios: Iterable[MarketDataSnapshot],
    ) -> List[float]:
        base_vals = self.position_values(base_market)
        pnls = []
        for scenario_market in scenarios:
            scen_vals = self.position_values(scenario_market)
            pnl = sum(sv - bv for sv, bv in zip(scen_vals, base_vals))
            pnls.append(pnl)
        return pnls

    def filter_by_product_types(self, product_types: Sequence[str]) -> "Portfolio":
        filtered = [pos for pos in self.positions if pos.product.product_type in product_types]
        return Portfolio(positions=filtered)
