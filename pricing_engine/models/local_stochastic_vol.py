"""Stochastic local volatility (LSV) model."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model, ModelConstraints


@dataclass(frozen=True)
class LeverageSurface:
    times: Sequence[float]
    spots: Sequence[float]
    values: Sequence[Sequence[float]]

    def value(self, t: float, spot: float) -> float:
        if not self.times or not self.spots:
            return 1.0
        t = max(t, 0.0)
        s = max(spot, 1e-8)

        times = list(self.times)
        spots = list(self.spots)
        vals = [list(row) for row in self.values]

        if t <= times[0]:
            t0 = t1 = 0
        elif t >= times[-1]:
            t0 = t1 = len(times) - 1
        else:
            t0 = max(i for i in range(len(times)) if times[i] <= t)
            t1 = min(t0 + 1, len(times) - 1)

        if s <= spots[0]:
            s0 = s1 = 0
        elif s >= spots[-1]:
            s0 = s1 = len(spots) - 1
        else:
            s0 = max(i for i in range(len(spots)) if spots[i] <= s)
            s1 = min(s0 + 1, len(spots) - 1)

        t_low = times[t0]
        t_high = times[t1]
        s_low = spots[s0]
        s_high = spots[s1]

        if t1 == t0 and s1 == s0:
            return vals[t0][s0]

        wt = 0.0 if t_high == t_low else (t - t_low) / (t_high - t_low)
        ws = 0.0 if s_high == s_low else (s - s_low) / (s_high - s_low)

        v00 = vals[t0][s0]
        v01 = vals[t0][s1]
        v10 = vals[t1][s0]
        v11 = vals[t1][s1]

        v0 = v00 + ws * (v01 - v00)
        v1 = v10 + ws * (v11 - v10)
        return v0 + wt * (v1 - v0)


@dataclass
class LocalStochasticVolModel(Model):
    kappa: float
    theta: float
    v0: float
    sigma: float
    rho: float
    leverage: LeverageSurface | None = None

    @property
    def name(self) -> str:
        return "local_stochastic_vol"

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

    def expected_variance(self, t: float) -> float:
        return self.theta + (self.v0 - self.theta) * math.exp(-self.kappa * max(t, 0.0))

    def _leverage(self, t: float, spot: float) -> float:
        if self.leverage is None:
            return 1.0
        return max(self.leverage.value(t, spot), 1e-6)

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

        paths = []
        for _ in range(num_paths):
            spot = market.spot
            var = max(self.v0, 0.0)
            path = [spot]
            t = 0.0
            for _ in range(steps):
                z1 = random.gauss(0.0, 1.0)
                z2 = self.rho * z1 + math.sqrt(1.0 - self.rho * self.rho) * random.gauss(0.0, 1.0)
                var = max(var + self.kappa * (self.theta - var) * dt + self.sigma * math.sqrt(var) * math.sqrt(dt) * z2, 0.0)
                lever = self._leverage(t, spot)
                eff_var = max(var * lever * lever, 0.0)
                spot *= math.exp((r - q - b - 0.5 * eff_var) * dt + math.sqrt(eff_var) * math.sqrt(dt) * z1)
                t += dt
                path.append(spot)
            paths.append(path)
        return paths
