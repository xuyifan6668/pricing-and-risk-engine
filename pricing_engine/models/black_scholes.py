"""Black-Scholes model."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model, ModelConstraints


@dataclass
class BlackScholesModel(Model):
    sigma: float

    @property
    def name(self) -> str:
        return "black_scholes"

    def params(self) -> Dict[str, float]:
        return {"sigma": self.sigma}

    def set_params(self, params: Dict[str, float]) -> None:
        self.sigma = float(params["sigma"])

    def initial_guess(self, market: MarketDataSnapshot) -> Dict[str, float]:
        return {"sigma": market.vol_surface.implied_vol(1.0, market.spot)}

    def constraints(self, market: MarketDataSnapshot) -> ModelConstraints:
        return ModelConstraints(bounds={"sigma": (1e-6, 5.0)})

    def simulate_paths(
        self,
        market: MarketDataSnapshot,
        timesteps: int,
        num_paths: int,
        horizon: float,
        seed: int | None,
    ) -> Sequence[Sequence[float]]:
        if seed is not None:
            random.seed(seed)

        steps = max(timesteps, 1)
        dt = max(horizon, 0.0) / steps
        r = market.funding_curve.zero_rate(max(horizon, 0.0))
        q = market.dividends.yield_rate(max(horizon, 0.0))
        b = market.borrow_curve.zero_rate(max(horizon, 0.0))
        drift = (r - q - b - 0.5 * self.sigma * self.sigma) * dt
        paths = []
        for _ in range(num_paths):
            spot = market.spot
            path = [spot]
            for _ in range(steps):
                diffusion = self.sigma * math.sqrt(dt) * random.gauss(0.0, 1.0)
                spot *= math.exp(drift + diffusion)
                path.append(spot)
            paths.append(path)
        return paths
