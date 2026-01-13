"""Hybrid local-stochastic volatility model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model, ModelConstraints


@dataclass
class HybridLocalStochModel(Model):
    # Placeholder parameters for hybrid models
    alpha: float
    beta: float

    @property
    def name(self) -> str:
        return "hybrid_local_stoch"

    def params(self) -> Dict[str, float]:
        return {"alpha": self.alpha, "beta": self.beta}

    def set_params(self, params: Dict[str, float]) -> None:
        self.alpha = float(params["alpha"])
        self.beta = float(params["beta"])

    def initial_guess(self, market: MarketDataSnapshot) -> Dict[str, float]:
        return {"alpha": 1.0, "beta": 0.5}

    def constraints(self, market: MarketDataSnapshot) -> ModelConstraints:
        return ModelConstraints(bounds={"alpha": (0.0, 5.0), "beta": (0.0, 5.0)})

    def simulate_paths(
        self,
        market: MarketDataSnapshot,
        timesteps: int,
        num_paths: int,
        horizon: float,
        seed: int | None,
    ):
        from pricing_engine.models.black_scholes import BlackScholesModel

        base = market.vol_surface.implied_vol(max(horizon, 0.0), market.spot)
        sigma = max(self.alpha * base + self.beta, 1e-6)
        return BlackScholesModel(sigma=sigma).simulate_paths(
            market=market,
            timesteps=timesteps,
            num_paths=num_paths,
            horizon=horizon,
            seed=seed,
        )
