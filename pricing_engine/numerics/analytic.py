"""Analytic pricing engines."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine, EngineSettings
from pricing_engine.products.exotics import BasketOption, DigitalOption, VarianceSwap, VolSwap
from pricing_engine.products.vanilla import EquitySwap, EuropeanOption, Forward, Spot
from pricing_engine.products.base import Product
from pricing_engine.utils.types import OptionType


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


def _variance_swap_fair_strike(
    spot: float,
    forward: float,
    t: float,
    r: float,
    q: float,
    vol_surface,
    strike_min: float | None = None,
    strike_max: float | None = None,
    steps: int = 200,
) -> float:
    if t <= 0.0:
        return 0.0
    k_min = strike_min or max(0.2 * forward, 1e-6)
    k_max = strike_max or 3.0 * forward
    if k_max <= k_min:
        return 0.0
    log_k_min = math.log(k_min / forward)
    log_k_max = math.log(k_max / forward)
    dx = (log_k_max - log_k_min) / steps
    integral = 0.0
    for i in range(steps):
        x0 = log_k_min + i * dx
        x1 = x0 + dx
        k0 = forward * math.exp(x0)
        k1 = forward * math.exp(x1)
        v0 = max(vol_surface.implied_vol(t, k0), 1e-8)
        v1 = max(vol_surface.implied_vol(t, k1), 1e-8)
        opt0 = OptionType.PUT if k0 < forward else OptionType.CALL
        opt1 = OptionType.PUT if k1 < forward else OptionType.CALL
        p0 = _bs_price(spot, k0, t, r, q, v0, opt0) / k0
        p1 = _bs_price(spot, k1, t, r, q, v1, opt1) / k1
        integral += 0.5 * (p0 + p1) * dx
    return (2.0 * math.exp(r * t) / t) * integral


@dataclass
class AnalyticEngine(Engine):
    @property
    def name(self) -> str:
        return "analytic"

    def price(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        settings: EngineSettings,
    ) -> Dict[str, float]:
        if not isinstance(model, BlackScholesModel):
            raise ValueError("AnalyticEngine supports BlackScholesModel only")

        t = market.time_to(product.maturity)
        r = market.discount_curve.zero_rate(t)
        q = market.dividends.yield_rate(t) + market.borrow_curve.zero_rate(t)

        if isinstance(product, Spot):
            return {"price": market.spot}

        if isinstance(product, EuropeanOption):
            price = _bs_price(
                market.spot,
                product.strike,
                t,
                r,
                q,
                model.sigma,
                product.option_type,
            )
            return {"price": price}

        if isinstance(product, BasketOption):
            price = _bs_price(
                market.spot,
                product.strike,
                t,
                r,
                q,
                model.sigma,
                product.option_type,
            )
            return {"price": price}

        if isinstance(product, DigitalOption):
            if t <= 0.0:
                if product.option_type == OptionType.CALL:
                    return {"price": product.payout if market.spot > product.strike else 0.0}
                return {"price": product.payout if market.spot < product.strike else 0.0}
            vol_sqrt = model.sigma * math.sqrt(t)
            d2 = (
                math.log(market.spot / product.strike) + (r - q - 0.5 * model.sigma * model.sigma) * t
            ) / vol_sqrt
            if product.option_type == OptionType.CALL:
                price = product.payout * math.exp(-r * t) * _norm_cdf(d2)
            else:
                price = product.payout * math.exp(-r * t) * _norm_cdf(-d2)
            return {"price": price}

        if isinstance(product, Forward):
            fwd = market.forward_price(product.maturity)
            df = market.discount_curve.df(t)
            return {"price": df * (fwd - product.strike)}

        if isinstance(product, EquitySwap):
            fwd = market.forward_price(product.maturity)
            df = market.discount_curve.df(t)
            equity_leg = df * product.notional * (fwd / max(market.spot, 1e-12) - 1.0)
            fixed_leg = df * product.notional * product.fixed_rate * t
            return {"price": equity_leg - fixed_leg}

        if isinstance(product, VarianceSwap):
            forward = market.forward_price(product.maturity)
            fair_var = _variance_swap_fair_strike(
                spot=market.spot,
                forward=forward,
                t=t,
                r=r,
                q=q,
                vol_surface=market.vol_surface,
            )
            df = market.discount_curve.df(t)
            return {"price": df * product.notional * (fair_var - product.strike), "fair_var": fair_var}

        if isinstance(product, VolSwap):
            forward = market.forward_price(product.maturity)
            fair_var = _variance_swap_fair_strike(
                spot=market.spot,
                forward=forward,
                t=t,
                r=r,
                q=q,
                vol_surface=market.vol_surface,
            )
            fair_vol = math.sqrt(max(fair_var, 0.0))
            df = market.discount_curve.df(t)
            return {"price": df * product.notional * (fair_vol - product.strike), "fair_vol": fair_vol}

        raise ValueError(f"AnalyticEngine unsupported product: {product.product_type}")
