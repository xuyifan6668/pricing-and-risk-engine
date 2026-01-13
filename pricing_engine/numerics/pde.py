"""Finite-difference PDE pricing engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import math

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine, EngineSettings
from pricing_engine.products.base import Product
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.utils.types import OptionType


@dataclass
class PDEEngine(Engine):
    grid_points: int = 200
    time_steps: int = 200
    s_max_multiplier: float = 4.0
    use_richardson: bool = False

    @property
    def name(self) -> str:
        return "pde"

    def price(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        settings: EngineSettings,
    ) -> Dict[str, float]:
        if self.use_richardson:
            coarse = self._price_grid(product, market, model, self.grid_points, self.time_steps)
            fine = self._price_grid(product, market, model, self.grid_points * 2, self.time_steps * 2)
            price = fine + (fine - coarse) / 3.0
            return {"price": price, "coarse": coarse, "fine": fine}
        price = self._price_grid(product, market, model, self.grid_points, self.time_steps)
        return {"price": price}

    def _price_grid(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        grid_points: int,
        time_steps: int,
    ) -> float:
        if not isinstance(model, BlackScholesModel):
            raise ValueError("PDEEngine supports BlackScholesModel only")
        if not isinstance(product, EuropeanOption):
            raise ValueError("PDEEngine supports EuropeanOption only")

        t = market.time_to(product.maturity)
        if t <= 0.0:
            return {"price": product.payoff(market.spot)}

        sigma = model.sigma
        r = market.discount_curve.zero_rate(t)
        q = market.dividends.yield_rate(t) + market.borrow_curve.zero_rate(t)
        strike = product.strike
        spot = market.spot

        m = max(grid_points, 10)
        n = max(time_steps, 10)
        s_max = max(self.s_max_multiplier * spot, self.s_max_multiplier * strike, spot * 2.0)
        d_s = s_max / (m - 1)
        dt = t / n

        grid = [i * d_s for i in range(m)]
        values = [product.payoff(s) for s in grid]

        def _solve_tridiagonal(a, b, c, d):
            size = len(d)
            cp = [0.0] * size
            dp = [0.0] * size
            cp[0] = c[0] / b[0]
            dp[0] = d[0] / b[0]
            for i in range(1, size):
                denom = b[i] - a[i] * cp[i - 1]
                cp[i] = c[i] / denom if i < size - 1 else 0.0
                dp[i] = (d[i] - a[i] * dp[i - 1]) / denom
            x = [0.0] * size
            x[-1] = dp[-1]
            for i in range(size - 2, -1, -1):
                x[i] = dp[i] - cp[i] * x[i + 1]
            return x

        for step in range(n):
            tau = t - step * dt
            if product.option_type == OptionType.CALL:
                values[0] = 0.0
                values[-1] = s_max * math.exp(-q * tau) - strike * math.exp(-r * tau)
            else:
                values[0] = strike * math.exp(-r * tau)
                values[-1] = 0.0

            a = []
            b = []
            c = []
            d_vec = []
            for i in range(1, m - 1):
                s = grid[i]
                a_i = -0.5 * dt * (sigma * sigma * s * s / (d_s * d_s) - (r - q) * s / d_s)
                b_i = 1.0 + dt * (sigma * sigma * s * s / (d_s * d_s) + r)
                c_i = -0.5 * dt * (sigma * sigma * s * s / (d_s * d_s) + (r - q) * s / d_s)
                a.append(a_i)
                b.append(b_i)
                c.append(c_i)
                d_vec.append(values[i])

            d_vec[0] -= a[0] * values[0]
            d_vec[-1] -= c[-1] * values[-1]

            solved = _solve_tridiagonal(a, b, c, d_vec)
            for i in range(1, m - 1):
                values[i] = solved[i - 1]

        # Linear interpolation to spot
        idx = min(int(spot / d_s), m - 2)
        s0 = grid[idx]
        s1 = grid[idx + 1]
        v0 = values[idx]
        v1 = values[idx + 1]
        price = v0 + (v1 - v0) * (spot - s0) / max(s1 - s0, 1e-12)
        return price
