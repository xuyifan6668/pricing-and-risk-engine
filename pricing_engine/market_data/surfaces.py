"""Volatility surfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Sequence

from pricing_engine.utils.types import OptionType


class VolSurface(ABC):
    """Abstract implied volatility surface."""

    @abstractmethod
    def implied_vol(self, t: float, strike: float) -> float:
        raise NotImplementedError


@dataclass(frozen=True)
class FlatVolSurface(VolSurface):
    vol: float

    def implied_vol(self, t: float, strike: float) -> float:
        return self.vol


def _linear_interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty surface")
    if len(xs) != len(ys):
        raise ValueError("surface size mismatch")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0 = xs[i - 1]
            x1 = xs[i]
            y0 = ys[i - 1]
            y1 = ys[i]
            w = (x - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return ys[-1]


@dataclass(frozen=True)
class SmileVolSurface(VolSurface):
    expiries: Sequence[float]
    atm_vols: Sequence[float]
    skew: Sequence[float]
    curvature: Sequence[float]
    spot_ref: float
    min_vol: float = 1e-4
    max_vol: float = 5.0

    def implied_vol(self, t: float, strike: float) -> float:
        t = max(t, 0.0)
        atm = _linear_interp(t, list(self.expiries), list(self.atm_vols))
        skew = _linear_interp(t, list(self.expiries), list(self.skew))
        curv = _linear_interp(t, list(self.expiries), list(self.curvature))
        denom = max(self.spot_ref, 1e-8)
        k = max(strike, 1e-8)
        x = math.log(k / denom)
        vol = atm + skew * x + 0.5 * curv * x * x
        return min(max(vol, self.min_vol), self.max_vol)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _bs_price(
    spot: float,
    strike: float,
    t: float,
    r: float,
    q: float,
    sigma: float,
    opt: OptionType,
) -> float:
    if t <= 0.0:
        if opt == OptionType.CALL:
            return max(spot - strike, 0.0)
        return max(strike - spot, 0.0)
    vol_sqrt = sigma * math.sqrt(t)
    if vol_sqrt <= 0.0:
        if opt == OptionType.CALL:
            return max(spot - strike * math.exp(-r * t), 0.0)
        return max(strike * math.exp(-r * t) - spot, 0.0)
    d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt
    d2 = d1 - vol_sqrt
    if opt == OptionType.CALL:
        return spot * math.exp(-q * t) * _norm_cdf(d1) - strike * math.exp(-r * t) * _norm_cdf(d2)
    return strike * math.exp(-r * t) * _norm_cdf(-d2) - spot * math.exp(-q * t) * _norm_cdf(-d1)


@dataclass(frozen=True)
class LocalVolSurfaceFromIV:
    implied_surface: VolSurface
    spot_ref: float
    dt: float = 1e-3
    dk_rel: float = 0.02
    min_vol: float = 1e-4
    max_vol: float = 5.0

    def local_vol(self, t: float, strike: float, r: float, q: float) -> float:
        t = max(t, 1e-6)
        k = max(strike, 1e-8)
        dt = max(self.dt, 1e-6)
        d_k = max(self.dk_rel * k, 1e-4 * self.spot_ref)

        def call_price(tt: float, kk: float) -> float:
            vol = self.implied_surface.implied_vol(tt, kk)
            return _bs_price(self.spot_ref, kk, tt, r, q, vol, OptionType.CALL)

        c_t_plus = call_price(t + dt, k)
        c_t_minus = call_price(max(t - dt, 1e-6), k)
        dcdt = (c_t_plus - c_t_minus) / (2.0 * dt)

        c_k_plus = call_price(t, k + d_k)
        c_k_minus = call_price(t, max(k - d_k, 1e-8))
        dcdk = (c_k_plus - c_k_minus) / (2.0 * d_k)
        d2cdk2 = (c_k_plus - 2.0 * call_price(t, k) + c_k_minus) / (d_k * d_k)

        numerator = dcdt + (r - q) * k * dcdk + q * call_price(t, k)
        denom = 0.5 * k * k * max(d2cdk2, 1e-12)
        local_var = max(numerator / denom, 0.0)
        local_vol = math.sqrt(local_var)
        return min(max(local_vol, self.min_vol), self.max_vol)
