"""Greek calculations."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, Protocol

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.surfaces import FlatVolSurface, SmileVolSurface, VolSurface
from pricing_engine.market_data.curves import Curve
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.utils.types import OptionType


@dataclass(frozen=True)
class GreekRequest:
    greeks: Iterable[str]
    method: str = "auto"
    bump_size: float = 1e-4
    vol_bump: float = 1e-4
    rate_bump: float = 1e-4

    @staticmethod
    def default() -> "GreekRequest":
        return GreekRequest(greeks=("delta", "gamma", "vega", "theta", "rho"), method="auto")


class _PricingContext(Protocol):
    product: object
    model: object
    market: object
    engine: object
    settings: object


class GreeksCalculator:
    def calculate(self, context: _PricingContext, request: GreekRequest) -> Dict[str, float]:
        if request.method == "auto":
            analytic = self._analytic_greeks(context, request)
            if analytic:
                return analytic
            return self._bump_greeks(context, request)
        if request.method == "bump":
            return self._bump_greeks(context, request)
        if request.method != "analytic":
            return {}
        return self._analytic_greeks(context, request)

    def _analytic_greeks(self, context: _PricingContext, request: GreekRequest) -> Dict[str, float]:
        product = context.product
        model = context.model
        market = context.market

        if not isinstance(product, EuropeanOption):
            return {}
        if not isinstance(model, BlackScholesModel):
            return {}

        t = market.time_to(product.maturity)
        if t <= 0.0:
            return {k: 0.0 for k in request.greeks}

        r = market.discount_curve.zero_rate(t)
        q = market.dividends.yield_rate(t) + market.borrow_curve.zero_rate(t)
        sigma = model.sigma
        spot = market.spot
        strike = product.strike

        vol_sqrt = sigma * math.sqrt(t)
        d1 = (math.log(spot / strike) + (r - q + 0.5 * sigma * sigma) * t) / vol_sqrt
        d2 = d1 - vol_sqrt
        pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
        nd1 = 0.5 * (1.0 + math.erf(d1 / math.sqrt(2.0)))
        nd2 = 0.5 * (1.0 + math.erf(d2 / math.sqrt(2.0)))

        sign = 1.0 if product.option_type == OptionType.CALL else -1.0

        delta = math.exp(-q * t) * (nd1 if sign > 0 else nd1 - 1.0)
        gamma = math.exp(-q * t) * pdf / (spot * vol_sqrt)
        vega = spot * math.exp(-q * t) * pdf * math.sqrt(t)
        theta = (
            -spot * pdf * sigma * math.exp(-q * t) / (2.0 * math.sqrt(t))
            - sign * r * strike * math.exp(-r * t) * (nd2 if sign > 0 else 1.0 - nd2)
            + sign * q * spot * math.exp(-q * t) * (nd1 if sign > 0 else 1.0 - nd1)
        )
        rho = sign * strike * t * math.exp(-r * t) * (nd2 if sign > 0 else 1.0 - nd2)

        values = {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "rho": rho,
        }
        return {k: values[k] for k in request.greeks if k in values}

    def _bump_greeks(self, context: _PricingContext, request: GreekRequest) -> Dict[str, float]:
        product = context.product
        model = context.model
        market = context.market
        engine = context.engine
        settings = context.settings.engine_settings if hasattr(context.settings, "engine_settings") else EngineSettings()

        spot = getattr(market, "spot", 0.0)
        if spot <= 0.0:
            return {}

        base_price = engine.price(product, market, model, settings).get("price")
        if base_price is None:
            return {}

        results: Dict[str, float] = {}

        if "delta" in request.greeks or "gamma" in request.greeks:
            bump = max(request.bump_size, 1e-8)
            up = _shift_spot(market, 1.0 + bump)
            dn = _shift_spot(market, 1.0 - bump)
            p_up = engine.price(product, up, model, settings).get("price", base_price)
            p_dn = engine.price(product, dn, model, settings).get("price", base_price)
            delta = (p_up - p_dn) / (2.0 * spot * bump)
            gamma = (p_up - 2.0 * base_price + p_dn) / (spot * spot * bump * bump)
            if "delta" in request.greeks:
                results["delta"] = delta
            if "gamma" in request.greeks:
                results["gamma"] = gamma

        if "vega" in request.greeks:
            v_bump = max(request.vol_bump, 1e-8)
            if isinstance(model, BlackScholesModel):
                model_up = BlackScholesModel(sigma=max(model.sigma + v_bump, 1e-8))
                model_dn = BlackScholesModel(sigma=max(model.sigma - v_bump, 1e-8))
                p_vu = engine.price(product, market, model_up, settings).get("price", base_price)
                p_vd = engine.price(product, market, model_dn, settings).get("price", base_price)
            else:
                vu = _shift_vol(market, v_bump)
                vd = _shift_vol(market, -v_bump)
                p_vu = engine.price(product, vu, model, settings).get("price", base_price)
                p_vd = engine.price(product, vd, model, settings).get("price", base_price)
            results["vega"] = (p_vu - p_vd) / (2.0 * v_bump)

        if "rho" in request.greeks:
            r_bump = max(request.rate_bump, 1e-8)
            ru = _shift_rate(market, r_bump)
            rd = _shift_rate(market, -r_bump)
            p_ru = engine.price(product, ru, model, settings).get("price", base_price)
            p_rd = engine.price(product, rd, model, settings).get("price", base_price)
            results["rho"] = (p_ru - p_rd) / (2.0 * r_bump)

        return results


def _shift_spot(market: MarketDataSnapshot, spot_multiplier: float) -> MarketDataSnapshot:
    return MarketDataSnapshot(
        asof=market.asof,
        spot=market.spot * spot_multiplier,
        discount_curve=market.discount_curve,
        funding_curve=market.funding_curve,
        borrow_curve=market.borrow_curve,
        vol_surface=market.vol_surface,
        dividends=market.dividends,
        corporate_actions=market.corporate_actions,
    )


def _shift_vol(market: MarketDataSnapshot, shift: float) -> MarketDataSnapshot:
    surface = market.vol_surface
    if isinstance(surface, FlatVolSurface):
        vol = max(surface.vol + shift, 1e-8)
        new_surface: VolSurface = FlatVolSurface(vol=vol)
    elif isinstance(surface, SmileVolSurface):
        atm = [max(v + shift, surface.min_vol) for v in surface.atm_vols]
        new_surface = SmileVolSurface(
            expiries=surface.expiries,
            atm_vols=atm,
            skew=surface.skew,
            curvature=surface.curvature,
            spot_ref=surface.spot_ref,
            min_vol=surface.min_vol,
            max_vol=surface.max_vol,
        )
    else:
        new_surface = surface
    return MarketDataSnapshot(
        asof=market.asof,
        spot=market.spot,
        discount_curve=market.discount_curve,
        funding_curve=market.funding_curve,
        borrow_curve=market.borrow_curve,
        vol_surface=new_surface,
        dividends=market.dividends,
        corporate_actions=market.corporate_actions,
    )


class _ShiftedCurve(Curve):
    def __init__(self, base: Curve, shift: float) -> None:
        self._base = base
        self._shift = shift

    def df(self, t: float) -> float:
        rate = self._base.zero_rate(t) + self._shift
        return math.exp(-rate * max(t, 0.0))


def _shift_rate(market: MarketDataSnapshot, shift: float) -> MarketDataSnapshot:
    discount_curve = _ShiftedCurve(market.discount_curve, shift)
    return MarketDataSnapshot(
        asof=market.asof,
        spot=market.spot,
        discount_curve=discount_curve,
        funding_curve=market.funding_curve,
        borrow_curve=market.borrow_curve,
        vol_surface=market.vol_surface,
        dividends=market.dividends,
        corporate_actions=market.corporate_actions,
    )
