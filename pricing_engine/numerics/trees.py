"""Tree-based pricing engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine, EngineSettings
from pricing_engine.products.base import Product
from pricing_engine.products.vanilla import AmericanOption, EuropeanOption
from pricing_engine.utils.types import OptionType


@dataclass
class TreeEngine(Engine):
    steps: int = 400

    @property
    def name(self) -> str:
        return "tree"

    def price(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        settings: EngineSettings,
    ) -> Dict[str, float]:
        if not isinstance(model, BlackScholesModel):
            raise ValueError("TreeEngine supports BlackScholesModel only")
        if not isinstance(product, (EuropeanOption, AmericanOption)):
            raise ValueError("TreeEngine supports EuropeanOption and AmericanOption only")

        t = market.time_to(product.maturity)
        steps = max(self.steps, 1)
        dt = max(t, 0.0) / steps
        r = market.discount_curve.zero_rate(t)
        q = market.dividends.yield_rate(t) + market.borrow_curve.zero_rate(t)
        sigma = model.sigma

        if dt <= 0.0:
            return {"price": product.payoff(market.spot)}

        u = math.exp(sigma * math.sqrt(dt))
        d = 1.0 / u
        disc = math.exp(-r * dt)
        p = (math.exp((r - q) * dt) - d) / (u - d)
        p = min(max(p, 0.0), 1.0)

        # Terminal payoffs
        values = []
        for i in range(steps + 1):
            spot = market.spot * (u ** (steps - i)) * (d ** i)
            values.append(product.payoff(spot))

        # Backward induction
        for step in range(steps - 1, -1, -1):
            new_values = []
            for i in range(step + 1):
                continuation = disc * (p * values[i] + (1.0 - p) * values[i + 1])
                if isinstance(product, AmericanOption):
                    spot = market.spot * (u ** (step - i)) * (d ** i)
                    exercise = product.payoff(spot)
                    new_values.append(max(continuation, exercise))
                else:
                    new_values.append(continuation)
            values = new_values

        return {"price": values[0]}
