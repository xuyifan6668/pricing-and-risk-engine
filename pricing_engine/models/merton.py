"""Merton jump diffusion model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model, ModelConstraints


@dataclass
class MertonJumpModel(Model):
    sigma: float
    jump_intensity: float
    jump_mean: float
    jump_vol: float

    @property
    def name(self) -> str:
        return "merton_jump"

    def params(self) -> Dict[str, float]:
        return {
            "sigma": self.sigma,
            "jump_intensity": self.jump_intensity,
            "jump_mean": self.jump_mean,
            "jump_vol": self.jump_vol,
        }

    def set_params(self, params: Dict[str, float]) -> None:
        self.sigma = float(params["sigma"])
        self.jump_intensity = float(params["jump_intensity"])
        self.jump_mean = float(params["jump_mean"])
        self.jump_vol = float(params["jump_vol"])

    def initial_guess(self, market: MarketDataSnapshot) -> Dict[str, float]:
        atm = market.vol_surface.implied_vol(1.0, market.spot)
        return {
            "sigma": atm,
            "jump_intensity": 0.1,
            "jump_mean": -0.05,
            "jump_vol": 0.2,
        }

    def constraints(self, market: MarketDataSnapshot) -> ModelConstraints:
        return ModelConstraints(
            bounds={
                "sigma": (1e-6, 5.0),
                "jump_intensity": (0.0, 10.0),
                "jump_mean": (-1.0, 1.0),
                "jump_vol": (1e-6, 5.0),
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

        def _poisson_sample(lam: float) -> int:
            if lam <= 0.0:
                return 0
            l_val = math.exp(-lam)
            k = 0
            p = 1.0
            while p > l_val:
                k += 1
                p *= random.random()
            return max(k - 1, 0)

        steps = max(timesteps, 1)
        dt = max(horizon, 0.0) / steps
        r = market.funding_curve.zero_rate(max(horizon, 0.0))
        q = market.dividends.yield_rate(max(horizon, 0.0))
        b = market.borrow_curve.zero_rate(max(horizon, 0.0))
        kappa = math.exp(self.jump_mean + 0.5 * self.jump_vol * self.jump_vol) - 1.0
        paths = []
        for _ in range(num_paths):
            spot = market.spot
            path = [spot]
            for _ in range(steps):
                z = random.gauss(0.0, 1.0)
                jumps = _poisson_sample(self.jump_intensity * dt)
                jump_sum = sum(random.gauss(self.jump_mean, self.jump_vol) for _ in range(jumps))
                drift = (r - q - b - 0.5 * self.sigma * self.sigma - self.jump_intensity * kappa) * dt
                diffusion = self.sigma * math.sqrt(dt) * z
                spot *= math.exp(drift + diffusion + jump_sum)
                path.append(spot)
            paths.append(path)
        return paths
