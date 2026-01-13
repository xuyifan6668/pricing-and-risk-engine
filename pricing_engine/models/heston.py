"""Heston stochastic volatility model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model, ModelConstraints


@dataclass
class HestonModel(Model):
    kappa: float
    theta: float
    v0: float
    sigma: float
    rho: float

    @property
    def name(self) -> str:
        return "heston"

    def params(self) -> Dict[str, float]:
        return {
            "kappa": self.kappa,
            "theta": self.theta,
            "v0": self.v0,
            "sigma": self.sigma,
            "rho": self.rho,
        }

    def set_params(self, params: Dict[str, float]) -> None:
        self.kappa = float(params["kappa"])
        self.theta = float(params["theta"])
        self.v0 = float(params["v0"])
        self.sigma = float(params["sigma"])
        self.rho = float(params["rho"])

    def initial_guess(self, market: MarketDataSnapshot) -> Dict[str, float]:
        atm = market.vol_surface.implied_vol(1.0, market.spot)
        return {
            "kappa": 1.5,
            "theta": atm * atm,
            "v0": atm * atm,
            "sigma": 0.5,
            "rho": -0.5,
        }

    def constraints(self, market: MarketDataSnapshot) -> ModelConstraints:
        return ModelConstraints(
            bounds={
                "kappa": (1e-6, 20.0),
                "theta": (1e-8, 4.0),
                "v0": (1e-8, 4.0),
                "sigma": (1e-6, 5.0),
                "rho": (-0.999, 0.999),
            }
        )

    def simulate_paths(
        self,
        market: MarketDataSnapshot,
        timesteps: int,
        num_paths: int,
        horizon: float,
        seed: int | None,
    ):
        import math
        import random

        if seed is not None:
            random.seed(seed)

        steps = max(timesteps, 1)
        dt = max(horizon, 0.0) / steps
        r = market.funding_curve.zero_rate(max(horizon, 0.0))
        q = market.dividends.yield_rate(max(horizon, 0.0))
        b = market.borrow_curve.zero_rate(max(horizon, 0.0))
        paths = []
        for _ in range(num_paths):
            spot = market.spot
            var = max(self.v0, 0.0)
            path = [spot]
            for _ in range(steps):
                z1 = random.gauss(0.0, 1.0)
                z2 = self.rho * z1 + math.sqrt(1.0 - self.rho * self.rho) * random.gauss(0.0, 1.0)
                var = max(var + self.kappa * (self.theta - var) * dt + self.sigma * math.sqrt(var) * math.sqrt(dt) * z2, 0.0)
                spot *= math.exp((r - q - b - 0.5 * var) * dt + math.sqrt(var) * math.sqrt(dt) * z1)
                path.append(spot)
            paths.append(path)
        return paths
