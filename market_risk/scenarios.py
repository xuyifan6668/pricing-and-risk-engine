"""Historical scenario generation and shock application."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, List, Sequence

from pricing_engine.market_data.curves import Curve, FlatCurve, ZeroCurve
from pricing_engine.market_data.dividends import DividendSchedule
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.surfaces import FlatVolSurface, SmileVolSurface, VolSurface

from market_risk.utils import linear_interp


@dataclass(frozen=True)
class VolShift:
    atm: Sequence[float] | None = None
    skew: Sequence[float] | None = None
    curvature: Sequence[float] | None = None
    flat: float = 0.0


class ShiftedCurve(Curve):
    def __init__(self, base: Curve, shifts: Dict[float, float]):
        self._base = base
        self._shifts = dict(sorted(shifts.items()))

    def df(self, t: float) -> float:
        shift = 0.0
        if self._shifts:
            tenors = list(self._shifts.keys())
            values = list(self._shifts.values())
            shift = linear_interp(t, tenors, values)
        rate = self._base.zero_rate(t) + shift
        return ZeroCurve([t], [rate]).df(t)


@dataclass(frozen=True)
class ScenarioShock:
    spot_multiplier: float
    rate_shifts: Dict[float, float]
    funding_shifts: Dict[float, float]
    borrow_shifts: Dict[float, float]
    vol_shift: VolShift
    dividend_shift: float = 0.0

    def apply(self, base: MarketDataSnapshot) -> MarketDataSnapshot:
        spot = base.spot * self.spot_multiplier
        discount_curve = ShiftedCurve(base.discount_curve, self.rate_shifts)
        funding_curve = ShiftedCurve(base.funding_curve, self.funding_shifts)
        borrow_curve = ShiftedCurve(base.borrow_curve, self.borrow_shifts)

        dividends = DividendSchedule(
            discrete=base.dividends.discrete,
            continuous_yield=base.dividends.continuous_yield + self.dividend_shift,
        )

        vol_surface = shift_surface(base.vol_surface, self.vol_shift, spot)

        return MarketDataSnapshot(
            asof=base.asof,
            spot=spot,
            discount_curve=discount_curve,
            funding_curve=funding_curve,
            borrow_curve=borrow_curve,
            vol_surface=vol_surface,
            dividends=dividends,
            corporate_actions=base.corporate_actions,
        )


@dataclass(frozen=True)
class Scenario:
    label: str
    asof: date
    shock: ScenarioShock


@dataclass(frozen=True)
class ScenarioSet:
    scenarios: Sequence[Scenario]
    horizon_days: int = 1

    def markets(self, base: MarketDataSnapshot) -> List[MarketDataSnapshot]:
        return [scenario.shock.apply(base) for scenario in self.scenarios]


def shift_surface(surface: VolSurface, shift: VolShift, spot_ref: float) -> VolSurface:
    if isinstance(surface, SmileVolSurface):
        expiries = list(surface.expiries)
        atm = list(surface.atm_vols)
        skew = list(surface.skew)
        curv = list(surface.curvature)
        if shift.atm:
            atm = [max(v + s, 1e-4) for v, s in zip(atm, shift.atm)]
        if shift.skew:
            skew = [v + s for v, s in zip(skew, shift.skew)]
        if shift.curvature:
            curv = [max(v + s, -1.0) for v, s in zip(curv, shift.curvature)]
        return SmileVolSurface(
            expiries=expiries,
            atm_vols=atm,
            skew=skew,
            curvature=curv,
            spot_ref=spot_ref,
        )
    if isinstance(surface, FlatVolSurface):
        return FlatVolSurface(max(surface.vol + shift.flat, 1e-4))
    return surface


def _extract_curve_nodes(curve: Curve, tenors: Sequence[float]) -> Dict[float, float]:
    return {t: curve.zero_rate(t) for t in tenors}


def _extract_surface_shift(prev: VolSurface, curr: VolSurface) -> VolShift:
    if isinstance(prev, SmileVolSurface) and isinstance(curr, SmileVolSurface):
        atm = [c - p for p, c in zip(prev.atm_vols, curr.atm_vols)]
        skew = [c - p for p, c in zip(prev.skew, curr.skew)]
        curv = [c - p for p, c in zip(prev.curvature, curr.curvature)]
        return VolShift(atm=atm, skew=skew, curvature=curv)
    if isinstance(prev, FlatVolSurface) and isinstance(curr, FlatVolSurface):
        return VolShift(flat=curr.vol - prev.vol)
    return VolShift()


@dataclass
class HistoricalScenarioBuilder:
    tenor_grid: Sequence[float]
    horizon_days: int = 1

    def build(self, history: Sequence[MarketDataSnapshot]) -> ScenarioSet:
        if len(history) < 2:
            return ScenarioSet(scenarios=[], horizon_days=self.horizon_days)

        scenarios: List[Scenario] = []
        tenors = list(self.tenor_grid)
        for idx in range(1, len(history)):
            prev = history[idx - 1]
            curr = history[idx]

            spot_multiplier = curr.spot / prev.spot if prev.spot else 1.0

            rate_prev = _extract_curve_nodes(prev.discount_curve, tenors)
            rate_curr = _extract_curve_nodes(curr.discount_curve, tenors)
            rate_shifts = {t: rate_curr[t] - rate_prev[t] for t in tenors}

            fund_prev = _extract_curve_nodes(prev.funding_curve, tenors)
            fund_curr = _extract_curve_nodes(curr.funding_curve, tenors)
            funding_shifts = {t: fund_curr[t] - fund_prev[t] for t in tenors}

            borrow_prev = _extract_curve_nodes(prev.borrow_curve, tenors)
            borrow_curr = _extract_curve_nodes(curr.borrow_curve, tenors)
            borrow_shifts = {t: borrow_curr[t] - borrow_prev[t] for t in tenors}

            vol_shift = _extract_surface_shift(prev.vol_surface, curr.vol_surface)

            div_shift = curr.dividends.continuous_yield - prev.dividends.continuous_yield

            shock = ScenarioShock(
                spot_multiplier=spot_multiplier,
                rate_shifts=rate_shifts,
                funding_shifts=funding_shifts,
                borrow_shifts=borrow_shifts,
                vol_shift=vol_shift,
                dividend_shift=div_shift,
            )
            scenarios.append(Scenario(label=f"{prev.asof}->{curr.asof}", asof=curr.asof, shock=shock))

        return ScenarioSet(scenarios=scenarios, horizon_days=self.horizon_days)


def default_tenor_grid() -> Sequence[float]:
    return [0.25, 0.5, 1.0, 2.0, 5.0]
