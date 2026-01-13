"""Local volatility model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model, ModelConstraints


@dataclass
class LocalVolModel(Model):
    # Placeholder parameters for local vol surface representation
    params_grid: Dict[str, float]

    @property
    def name(self) -> str:
        return "local_vol"

    def params(self) -> Dict[str, float]:
        return dict(self.params_grid)

    def set_params(self, params: Dict[str, float]) -> None:
        self.params_grid = dict(params)

    def initial_guess(self, market: MarketDataSnapshot) -> Dict[str, float]:
        return {"sigma_atm": market.vol_surface.implied_vol(1.0, market.spot)}

    def constraints(self, market: MarketDataSnapshot) -> ModelConstraints:
        return ModelConstraints(bounds={"sigma_atm": (1e-6, 5.0)})

    def simulate_paths(
        self,
        market: MarketDataSnapshot,
        timesteps: int,
        num_paths: int,
        horizon: float,
        seed: int | None,
    ):
        from pricing_engine.models.black_scholes import BlackScholesModel

        sigma = float(self.params_grid.get("sigma_atm", 0.2))
        return BlackScholesModel(sigma=sigma).simulate_paths(
            market=market,
            timesteps=timesteps,
            num_paths=num_paths,
            horizon=horizon,
            seed=seed,
        )
