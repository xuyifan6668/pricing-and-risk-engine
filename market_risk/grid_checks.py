"""Grid-based sensitivity checks and interpolation error maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
import json
import math
import os
import random
from typing import List, Optional, Sequence, Tuple

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.dividends import DividendSchedule

from market_risk.derivatives import (
    build_grid_map,
    cross_derivative,
    first_derivative,
    flatten,
    mean_ignore_nan,
    second_derivative,
    transpose,
)
from market_risk.grid_approx import (
    GridConfig,
    InterpolationSettings,
    PortfolioGridApproximator,
    apply_spot_vol_shift,
    barrier_cross_map,
    bilinear_interpolate_context,
    build_interpolation_context,
)


@dataclass(frozen=True)
class GridCheckConfig:
    spot_nodes: Sequence[float]
    vol_shifts: Sequence[float]
    dividend_shifts: Sequence[float]
    rate_shifts: Sequence[float] = (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)
    funding_shifts: Sequence[float] = (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)
    borrow_shifts: Sequence[float] = (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)
    spot_bump_bp: float = 1.0
    vol_bump_bp: float = 1.0
    dividend_bump_bp: float = 1.0
    rate_bump_bp: float = 1.0
    funding_bump_bp: float = 1.0
    borrow_bump_bp: float = 1.0
    sample_stride: int = 1
    interp_settings: InterpolationSettings = field(default_factory=InterpolationSettings)
    refine_delta_threshold: float = 0.05
    refine_gamma_threshold: float = 0.1
    refine_vega_threshold: float = 0.1
    refine_max_spot_nodes: int = 6
    refine_max_vol_nodes: int = 6
    adaptive_refine: bool = True
    adaptive_spot_nodes: int = 4
    adaptive_vol_nodes: int = 4
    min_spot_step: float = 0.01
    min_vol_step: float = 0.002
    rniv_addon_multiplier: float = 0.05
    output_dir: str = "market_risk/reports"


@dataclass(frozen=True)
class SensitivityErrorMap:
    name: str
    values: List[List[float]]


@dataclass(frozen=True)
class GridCheckSummary:
    asof: date
    base_value: float
    mean_rel_delta_error: float
    mean_rel_gamma_error: float
    mean_rel_vega_error: float
    mean_rel_cross_gamma_error: float
    mean_rel_dividend_error: float
    mean_rel_dividend_cross_error: float
    mean_rel_rate_error: float
    mean_rel_rate_cross_error: float
    mean_rel_funding_error: float
    mean_rel_funding_cross_error: float
    mean_rel_borrow_error: float
    mean_rel_borrow_cross_error: float
    interp_mean_rel_error: float
    interp_max_rel_error: float


@dataclass(frozen=True)
class GridRefinementPlan:
    spot_nodes: List[float]
    vol_shifts: List[float]
    added_spot_nodes: List[float]
    added_vol_nodes: List[float]
    flagged_cells: int
    unrefined_cells: int
    rniv_addon: float
    rniv_multiplier: float
    delta_threshold: float
    gamma_threshold: float
    vega_threshold: float
    adaptive_refine: bool
    min_spot_step: float
    min_vol_step: float


@dataclass(frozen=True)
class GridCheckResult:
    summary: GridCheckSummary
    delta_error_map: SensitivityErrorMap
    gamma_error_map: SensitivityErrorMap
    vega_error_map: SensitivityErrorMap
    cross_gamma_error_map: SensitivityErrorMap
    dividend_error_map: SensitivityErrorMap
    dividend_cross_error_map: SensitivityErrorMap
    rate_error_map: SensitivityErrorMap
    rate_cross_error_map: SensitivityErrorMap
    funding_error_map: SensitivityErrorMap
    funding_cross_error_map: SensitivityErrorMap
    borrow_error_map: SensitivityErrorMap
    borrow_cross_error_map: SensitivityErrorMap
    regime_map: SensitivityErrorMap
    interp_error_map: SensitivityErrorMap
    score_map: SensitivityErrorMap
    refine_map: SensitivityErrorMap
    refinement: GridRefinementPlan


class GridCheckRunner:
    def __init__(self, portfolio, base_market: MarketDataSnapshot) -> None:
        self.portfolio = portfolio
        self.base_market = base_market

    def run(self, config: GridCheckConfig) -> GridCheckResult:
        approx = PortfolioGridApproximator(
            self.portfolio,
            self.base_market,
            interp_settings=config.interp_settings,
        )
        grid_config = GridConfig(spot_nodes=config.spot_nodes, vol_shifts=config.vol_shifts)
        grid = approx.build_grid(grid_config)

        spot_bp = config.spot_bump_bp * 1e-4
        vol_bp = config.vol_bump_bp * 1e-4
        div_bp = config.dividend_bump_bp * 1e-4
        rate_bp = config.rate_bump_bp * 1e-4
        funding_bp = config.funding_bump_bp * 1e-4
        borrow_bp = config.borrow_bump_bp * 1e-4

        delta_errs = build_grid_map(grid.spot_grid, grid.vol_grid, 0.0)
        gamma_errs = build_grid_map(grid.spot_grid, grid.vol_grid, 0.0)
        vega_errs = build_grid_map(grid.spot_grid, grid.vol_grid, 0.0)
        cross_errs = build_grid_map(grid.spot_grid, grid.vol_grid, 0.0)

        dividend_grid = build_dividend_grid(
            self.portfolio,
            self.base_market,
            grid.spot_grid,
            config.dividend_shifts,
        )
        dividend_errs = build_grid_map(grid.spot_grid, config.dividend_shifts, 0.0)
        dividend_cross_errs = build_grid_map(grid.spot_grid, config.dividend_shifts, 0.0)
        regime_mask = barrier_cross_map(self.portfolio, self.base_market, grid.spot_grid, grid.vol_grid)

        _, rate_errs, rate_cross_errs = curve_error_maps(
            self.portfolio,
            self.base_market,
            grid.spot_grid,
            config.rate_shifts,
            spot_bp,
            rate_bp,
            curve_name="discount",
            sample_stride=config.sample_stride,
        )
        _, funding_errs, funding_cross_errs = curve_error_maps(
            self.portfolio,
            self.base_market,
            grid.spot_grid,
            config.funding_shifts,
            spot_bp,
            funding_bp,
            curve_name="funding",
            sample_stride=config.sample_stride,
        )
        _, borrow_errs, borrow_cross_errs = curve_error_maps(
            self.portfolio,
            self.base_market,
            grid.spot_grid,
            config.borrow_shifts,
            spot_bp,
            borrow_bp,
            curve_name="borrow",
            sample_stride=config.sample_stride,
        )

        for i in range(1, len(grid.spot_grid) - 1, config.sample_stride):
            for j in range(1, len(grid.vol_grid) - 1, config.sample_stride):
                spot_mult = grid.spot_grid[i]
                vol_shift = grid.vol_grid[j]
                base_val = grid.grid_values[i][j]

                theo_delta = first_derivative(
                    grid.spot_grid,
                    grid.grid_values,
                    i,
                    j,
                    scale=self.base_market.spot,
                )
                theo_gamma = second_derivative(
                    grid.spot_grid,
                    grid.grid_values,
                    i,
                    j,
                    scale=self.base_market.spot,
                )
                theo_vega = first_derivative(
                    grid.vol_grid,
                    transpose(grid.grid_values),
                    j,
                    i,
                    scale=1.0,
                )

                shock_delta, shock_gamma, shock_vega = bump_sensitivities(
                    self.portfolio,
                    self.base_market,
                    spot_mult,
                    vol_shift,
                    spot_bp,
                    vol_bp,
                    base_val,
                )

                delta_errs[i][j] = rel_error(theo_delta, shock_delta)
                gamma_errs[i][j] = rel_error(theo_gamma, shock_gamma)
                vega_errs[i][j] = rel_error(theo_vega, shock_vega)

                theo_cross = cross_derivative(
                    grid.spot_grid,
                    grid.vol_grid,
                    grid.grid_values,
                    i,
                    j,
                    scale_x=self.base_market.spot,
                    scale_y=1.0,
                )
                shock_cross = bump_cross_gamma(
                    self.portfolio,
                    self.base_market,
                    spot_mult,
                    vol_shift,
                    spot_bp,
                    vol_bp,
                )
                cross_errs[i][j] = rel_error(theo_cross, shock_cross)

        for i in range(1, len(grid.spot_grid) - 1, config.sample_stride):
            for j in range(1, len(config.dividend_shifts) - 1, config.sample_stride):
                theo_div = first_derivative(
                    config.dividend_shifts,
                    dividend_grid,
                    j,
                    i,
                    scale=1.0,
                )
                shock_div = bump_dividend(
                    self.portfolio,
                    self.base_market,
                    grid.spot_grid[i],
                    config.dividend_shifts[j],
                    div_bp,
                )
                dividend_errs[i][j] = rel_error(theo_div, shock_div)
                theo_div_cross = cross_derivative(
                    grid.spot_grid,
                    config.dividend_shifts,
                    transpose(dividend_grid),
                    i,
                    j,
                    scale_x=self.base_market.spot,
                    scale_y=1.0,
                )
                shock_div_cross = bump_spot_dividend_cross(
                    self.portfolio,
                    self.base_market,
                    grid.spot_grid[i],
                    config.dividend_shifts[j],
                    spot_bp,
                    div_bp,
                )
                dividend_cross_errs[i][j] = rel_error(theo_div_cross, shock_div_cross)

        interp_context = build_interpolation_context(
            grid.spot_grid,
            grid.vol_grid,
            self.base_market,
            config.interp_settings,
        )
        interp_errors = interpolation_error_map(
            self.portfolio,
            self.base_market,
            grid.spot_grid,
            grid.vol_grid,
            grid.grid_values,
            stride=max(config.sample_stride, 1),
            regime_mask=regime_mask,
            interp_context=interp_context,
        )

        score_map, refine_map, flagged = error_score_maps(
            delta_errs,
            gamma_errs,
            vega_errs,
            regime_mask,
            config,
        )
        refinement = suggest_refinement(
            grid.spot_grid,
            grid.vol_grid,
            score_map,
            regime_mask,
            config,
            grid.base_value,
            flagged,
        )

        summary = GridCheckSummary(
            asof=self.base_market.asof,
            base_value=grid.base_value,
            mean_rel_delta_error=mean_ignore_nan(flatten(delta_errs)),
            mean_rel_gamma_error=mean_ignore_nan(flatten(gamma_errs)),
            mean_rel_vega_error=mean_ignore_nan(flatten(vega_errs)),
            mean_rel_cross_gamma_error=mean_ignore_nan(flatten(cross_errs)),
            mean_rel_dividend_error=mean_ignore_nan(flatten(dividend_errs)),
            mean_rel_dividend_cross_error=mean_ignore_nan(flatten(dividend_cross_errs)),
            mean_rel_rate_error=mean_ignore_nan(flatten(rate_errs)),
            mean_rel_rate_cross_error=mean_ignore_nan(flatten(rate_cross_errs)),
            mean_rel_funding_error=mean_ignore_nan(flatten(funding_errs)),
            mean_rel_funding_cross_error=mean_ignore_nan(flatten(funding_cross_errs)),
            mean_rel_borrow_error=mean_ignore_nan(flatten(borrow_errs)),
            mean_rel_borrow_cross_error=mean_ignore_nan(flatten(borrow_cross_errs)),
            interp_mean_rel_error=mean_ignore_nan(flatten(interp_errors)),
            interp_max_rel_error=max(flatten(interp_errors), default=0.0),
        )

        return GridCheckResult(
            summary=summary,
            delta_error_map=SensitivityErrorMap("delta", delta_errs),
            gamma_error_map=SensitivityErrorMap("gamma", gamma_errs),
            vega_error_map=SensitivityErrorMap("vega", vega_errs),
            cross_gamma_error_map=SensitivityErrorMap("cross_gamma", cross_errs),
            dividend_error_map=SensitivityErrorMap("dividend", dividend_errs),
            dividend_cross_error_map=SensitivityErrorMap("dividend_cross", dividend_cross_errs),
            rate_error_map=SensitivityErrorMap("rate", rate_errs),
            rate_cross_error_map=SensitivityErrorMap("rate_cross", rate_cross_errs),
            funding_error_map=SensitivityErrorMap("funding", funding_errs),
            funding_cross_error_map=SensitivityErrorMap("funding_cross", funding_cross_errs),
            borrow_error_map=SensitivityErrorMap("borrow", borrow_errs),
            borrow_cross_error_map=SensitivityErrorMap("borrow_cross", borrow_cross_errs),
            regime_map=SensitivityErrorMap("regime", regime_mask),
            interp_error_map=SensitivityErrorMap("interp", interp_errors),
            score_map=SensitivityErrorMap("error_score", score_map),
            refine_map=SensitivityErrorMap("refine_flags", refine_map),
            refinement=refinement,
        )


def bump_sensitivities(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_mult: float,
    vol_shift: float,
    spot_bp: float,
    vol_bp: float,
    base_val: float,
) -> Tuple[float, float, float]:
    spot = base_market.spot
    market_up = apply_spot_vol_shift(base_market, spot_mult * (1.0 + spot_bp), vol_shift)
    market_dn = apply_spot_vol_shift(base_market, spot_mult * (1.0 - spot_bp), vol_shift)

    p_up = portfolio.value(market_up)
    p_dn = portfolio.value(market_dn)

    delta = (p_up - p_dn) / (2.0 * spot * spot_bp)
    gamma = (p_up - 2.0 * base_val + p_dn) / (spot * spot * spot_bp * spot_bp)

    market_vu = apply_spot_vol_shift(base_market, spot_mult, vol_shift + vol_bp)
    market_vd = apply_spot_vol_shift(base_market, spot_mult, vol_shift - vol_bp)
    p_vu = portfolio.value(market_vu)
    p_vd = portfolio.value(market_vd)
    vega = (p_vu - p_vd) / (2.0 * vol_bp)

    return delta, gamma, vega


def bump_cross_gamma(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_mult: float,
    vol_shift: float,
    spot_bp: float,
    vol_bp: float,
) -> float:
    market_pp = apply_spot_vol_shift(base_market, spot_mult * (1.0 + spot_bp), vol_shift + vol_bp)
    market_pm = apply_spot_vol_shift(base_market, spot_mult * (1.0 + spot_bp), vol_shift - vol_bp)
    market_mp = apply_spot_vol_shift(base_market, spot_mult * (1.0 - spot_bp), vol_shift + vol_bp)
    market_mm = apply_spot_vol_shift(base_market, spot_mult * (1.0 - spot_bp), vol_shift - vol_bp)

    p_pp = portfolio.value(market_pp)
    p_pm = portfolio.value(market_pm)
    p_mp = portfolio.value(market_mp)
    p_mm = portfolio.value(market_mm)

    ds = base_market.spot * spot_bp * 2.0
    dv = vol_bp * 2.0
    if ds == 0.0 or dv == 0.0:
        return 0.0
    return (p_pp - p_pm - p_mp + p_mm) / (ds * dv)


def bump_dividend(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_mult: float,
    dividend_shift: float,
    div_bp: float,
) -> float:
    market_up = apply_dividend_shift(base_market, spot_mult, dividend_shift + div_bp)
    market_dn = apply_dividend_shift(base_market, spot_mult, dividend_shift - div_bp)
    p_up = portfolio.value(market_up)
    p_dn = portfolio.value(market_dn)
    return (p_up - p_dn) / (2.0 * div_bp)


def bump_spot_dividend_cross(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_mult: float,
    dividend_shift: float,
    spot_bp: float,
    div_bp: float,
) -> float:
    market_pp = apply_dividend_shift(base_market, spot_mult * (1.0 + spot_bp), dividend_shift + div_bp)
    market_pm = apply_dividend_shift(base_market, spot_mult * (1.0 + spot_bp), dividend_shift - div_bp)
    market_mp = apply_dividend_shift(base_market, spot_mult * (1.0 - spot_bp), dividend_shift + div_bp)
    market_mm = apply_dividend_shift(base_market, spot_mult * (1.0 - spot_bp), dividend_shift - div_bp)
    p_pp = portfolio.value(market_pp)
    p_pm = portfolio.value(market_pm)
    p_mp = portfolio.value(market_mp)
    p_mm = portfolio.value(market_mm)
    ds = base_market.spot * spot_bp * 2.0
    dv = div_bp * 2.0
    if ds == 0.0 or dv == 0.0:
        return 0.0
    return (p_pp - p_pm - p_mp + p_mm) / (ds * dv)


def interpolation_error_map(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    vol_grid: Sequence[float],
    values: Sequence[Sequence[float]],
    stride: int = 1,
    regime_mask: Sequence[Sequence[float]] | None = None,
    interp_context=None,
) -> List[List[float]]:
    rows = len(spot_grid) - 1
    cols = len(vol_grid) - 1
    errors = [[0.0 for _ in range(cols)] for _ in range(rows)]

    if interp_context is None:
        interp_context = build_interpolation_context(
            spot_grid,
            vol_grid,
            base_market,
            InterpolationSettings(),
        )

    for i in range(0, rows, stride):
        for j in range(0, cols, stride):
            if regime_mask is not None and regime_mask[i][j] > 0.5:
                errors[i][j] = float("nan")
                continue
            spot_mid = 0.5 * (spot_grid[i] + spot_grid[i + 1])
            vol_mid = 0.5 * (vol_grid[j] + vol_grid[j + 1])
            approx = bilinear_interpolate_context(interp_context, values, spot_mid, vol_mid)
            full = portfolio.value(apply_spot_vol_shift(base_market, spot_mid, vol_mid))
            errors[i][j] = rel_error(approx, full)
    return errors


def curve_error_maps(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    curve_shifts: Sequence[float],
    spot_bp: float,
    curve_bp: float,
    curve_name: str,
    sample_stride: int,
) -> Tuple[List[List[float]], List[List[float]], List[List[float]]]:
    values = build_curve_grid(portfolio, base_market, spot_grid, curve_shifts, curve_name)
    errs = build_grid_map(spot_grid, curve_shifts, 0.0)
    cross_errs = build_grid_map(spot_grid, curve_shifts, 0.0)
    if len(curve_shifts) < 3 or len(spot_grid) < 3:
        return values, errs, cross_errs

    for i in range(1, len(spot_grid) - 1, sample_stride):
        for j in range(1, len(curve_shifts) - 1, sample_stride):
            theo = first_derivative(
                curve_shifts,
                transpose(values),
                j,
                i,
                scale=1.0,
            )
            shock = bump_curve_sensitivity(
                portfolio,
                base_market,
                spot_grid[i],
                curve_shifts[j],
                curve_bp,
                curve_name,
            )
            errs[i][j] = rel_error(theo, shock)
            theo_cross = cross_derivative(
                spot_grid,
                curve_shifts,
                values,
                i,
                j,
                scale_x=base_market.spot,
                scale_y=1.0,
            )
            shock_cross = bump_spot_curve_cross(
                portfolio,
                base_market,
                spot_grid[i],
                curve_shifts[j],
                spot_bp,
                curve_bp,
                curve_name,
            )
            cross_errs[i][j] = rel_error(theo_cross, shock_cross)

    return values, errs, cross_errs


def build_curve_grid(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    curve_shifts: Sequence[float],
    curve_name: str,
) -> List[List[float]]:
    values: List[List[float]] = []
    for spot_mult in spot_grid:
        row = []
        for shift in curve_shifts:
            market = apply_curve_shift(base_market, spot_mult, shift, curve_name)
            row.append(portfolio.value(market))
        values.append(row)
    return values


class _ParallelShiftCurve:
    def __init__(self, base_curve, shift: float) -> None:
        self._base = base_curve
        self._shift = shift

    def df(self, t: float) -> float:
        import math

        rate = self._base.zero_rate(t) + self._shift
        return math.exp(-rate * max(t, 0.0))


def apply_curve_shift(
    base_market: MarketDataSnapshot,
    spot_mult: float,
    curve_shift: float,
    curve_name: str,
) -> MarketDataSnapshot:
    spot = base_market.spot * spot_mult
    discount_curve = base_market.discount_curve
    funding_curve = base_market.funding_curve
    borrow_curve = base_market.borrow_curve

    if curve_name == "discount":
        discount_curve = _ParallelShiftCurve(discount_curve, curve_shift)
    elif curve_name == "funding":
        funding_curve = _ParallelShiftCurve(funding_curve, curve_shift)
    elif curve_name == "borrow":
        borrow_curve = _ParallelShiftCurve(borrow_curve, curve_shift)
    else:
        raise ValueError(f"Unsupported curve_name: {curve_name}")

    return MarketDataSnapshot(
        asof=base_market.asof,
        spot=spot,
        discount_curve=discount_curve,
        funding_curve=funding_curve,
        borrow_curve=borrow_curve,
        vol_surface=base_market.vol_surface,
        dividends=base_market.dividends,
        corporate_actions=base_market.corporate_actions,
    )


def bump_curve_sensitivity(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_mult: float,
    curve_shift: float,
    curve_bp: float,
    curve_name: str,
) -> float:
    market_up = apply_curve_shift(base_market, spot_mult, curve_shift + curve_bp, curve_name)
    market_dn = apply_curve_shift(base_market, spot_mult, curve_shift - curve_bp, curve_name)
    p_up = portfolio.value(market_up)
    p_dn = portfolio.value(market_dn)
    return (p_up - p_dn) / (2.0 * curve_bp)


def bump_spot_curve_cross(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_mult: float,
    curve_shift: float,
    spot_bp: float,
    curve_bp: float,
    curve_name: str,
) -> float:
    market_pp = apply_curve_shift(base_market, spot_mult * (1.0 + spot_bp), curve_shift + curve_bp, curve_name)
    market_pm = apply_curve_shift(base_market, spot_mult * (1.0 + spot_bp), curve_shift - curve_bp, curve_name)
    market_mp = apply_curve_shift(base_market, spot_mult * (1.0 - spot_bp), curve_shift + curve_bp, curve_name)
    market_mm = apply_curve_shift(base_market, spot_mult * (1.0 - spot_bp), curve_shift - curve_bp, curve_name)
    p_pp = portfolio.value(market_pp)
    p_pm = portfolio.value(market_pm)
    p_mp = portfolio.value(market_mp)
    p_mm = portfolio.value(market_mm)
    ds = base_market.spot * spot_bp * 2.0
    dr = curve_bp * 2.0
    if ds == 0.0 or dr == 0.0:
        return 0.0
    return (p_pp - p_pm - p_mp + p_mm) / (ds * dr)


def build_dividend_grid(
    portfolio,
    base_market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    dividend_shifts: Sequence[float],
) -> List[List[float]]:
    values: List[List[float]] = []
    for shift in dividend_shifts:
        row = []
        for spot_mult in spot_grid:
            market = apply_dividend_shift(base_market, spot_mult, shift)
            row.append(portfolio.value(market))
        values.append(row)
    return values


def apply_dividend_shift(
    base_market: MarketDataSnapshot,
    spot_mult: float,
    dividend_shift: float,
) -> MarketDataSnapshot:
    spot = base_market.spot * spot_mult
    dividends = DividendSchedule(
        discrete=base_market.dividends.discrete,
        continuous_yield=base_market.dividends.continuous_yield + dividend_shift,
    )
    return MarketDataSnapshot(
        asof=base_market.asof,
        spot=spot,
        discount_curve=base_market.discount_curve,
        funding_curve=base_market.funding_curve,
        borrow_curve=base_market.borrow_curve,
        vol_surface=base_market.vol_surface,
        dividends=dividends,
        corporate_actions=base_market.corporate_actions,
    )


def rel_error(a: float, b: float) -> float:
    denom = max(1.0, abs(b))
    return abs(a - b) / denom


def error_score_maps(
    delta_errs: Sequence[Sequence[float]],
    gamma_errs: Sequence[Sequence[float]],
    vega_errs: Sequence[Sequence[float]],
    regime_mask: Sequence[Sequence[float]],
    config: GridCheckConfig,
) -> Tuple[List[List[float]], List[List[float]], int]:
    rows = len(delta_errs)
    cols = len(delta_errs[0]) if rows else 0
    score_map = [[float("nan") for _ in range(cols)] for _ in range(rows)]
    flag_map = [[float("nan") for _ in range(cols)] for _ in range(rows)]
    flagged = 0
    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if _node_in_regime(regime_mask, i, j):
                continue
            score = _error_score(delta_errs[i][j], gamma_errs[i][j], vega_errs[i][j], config)
            score_map[i][j] = score
            if score >= 1.0:
                flag_map[i][j] = 1.0
                flagged += 1
            else:
                flag_map[i][j] = 0.0

    return score_map, flag_map, flagged


def suggest_refinement(
    spot_grid: Sequence[float],
    vol_grid: Sequence[float],
    score_map: Sequence[Sequence[float]],
    regime_mask: Sequence[Sequence[float]],
    config: GridCheckConfig,
    base_value: float,
    flagged_cells: int,
) -> GridRefinementPlan:
    candidates: List[Tuple[float, int, int]] = []
    for i in range(1, len(score_map) - 1):
        for j in range(1, len(score_map[i]) - 1):
            score = score_map[i][j]
            if math.isnan(score) or score < 1.0:
                continue
            candidates.append((score, i, j))

    candidates.sort(reverse=True, key=lambda item: item[0])
    new_spot = set(spot_grid)
    new_vol = set(vol_grid)
    added_spot: List[float] = []
    added_vol: List[float] = []
    unrefined_scores: List[float] = []

    for score, i, j in candidates:
        spot_added = False
        vol_added = False
        spot_added |= _add_midpoint(spot_grid, new_spot, i - 1, i, config.refine_max_spot_nodes)
        spot_added |= _add_midpoint(spot_grid, new_spot, i, i + 1, config.refine_max_spot_nodes)
        vol_added |= _add_midpoint(vol_grid, new_vol, j - 1, j, config.refine_max_vol_nodes)
        vol_added |= _add_midpoint(vol_grid, new_vol, j, j + 1, config.refine_max_vol_nodes)

        if not (spot_added or vol_added):
            unrefined_scores.append(score)

    if config.adaptive_refine:
        added_spot.extend(
            refine_axis_by_score(
                axis=spot_grid,
                score_map=score_map,
                max_new=config.adaptive_spot_nodes,
                min_step=config.min_spot_step,
                axis_name="spot",
                new_nodes=new_spot,
                regime_mask=regime_mask,
            )
        )
        added_vol.extend(
            refine_axis_by_score(
                axis=vol_grid,
                score_map=score_map,
                max_new=config.adaptive_vol_nodes,
                min_step=config.min_vol_step,
                axis_name="vol",
                new_nodes=new_vol,
                regime_mask=regime_mask,
            )
        )

    residual_scores = [
        score
        for row in score_map
        for score in row
        if not math.isnan(score) and score < 1.0
    ]
    residual_scores.extend(unrefined_scores)
    rniv_base = sum(residual_scores) / len(residual_scores) if residual_scores else 0.0
    rniv_addon = base_value * config.rniv_addon_multiplier * rniv_base

    return GridRefinementPlan(
        spot_nodes=sorted(new_spot),
        vol_shifts=sorted(new_vol),
        added_spot_nodes=sorted(added_spot),
        added_vol_nodes=sorted(added_vol),
        flagged_cells=flagged_cells,
        unrefined_cells=len(unrefined_scores),
        rniv_addon=rniv_addon,
        rniv_multiplier=config.rniv_addon_multiplier,
        delta_threshold=config.refine_delta_threshold,
        gamma_threshold=config.refine_gamma_threshold,
        vega_threshold=config.refine_vega_threshold,
        adaptive_refine=config.adaptive_refine,
        min_spot_step=config.min_spot_step,
        min_vol_step=config.min_vol_step,
    )


def refine_axis_by_score(
    axis: Sequence[float],
    score_map: Sequence[Sequence[float]],
    max_new: int,
    min_step: float,
    axis_name: str,
    new_nodes: set[float],
    regime_mask: Sequence[Sequence[float]] | None = None,
) -> List[float]:
    if max_new <= 0 or len(axis) < 2:
        return []
    interval_scores = _interval_scores(score_map, axis_name)
    intervals = sorted(range(len(interval_scores)), key=lambda i: interval_scores[i], reverse=True)
    added: List[float] = []
    for idx in intervals:
        if len(added) >= max_new:
            break
        score = interval_scores[idx]
        if math.isnan(score) or score <= 0.0:
            continue
        if _interval_crosses_regime(regime_mask, axis_name, idx):
            continue
        left = axis[idx]
        right = axis[idx + 1]
        if right - left < min_step:
            continue
        mid = 0.5 * (left + right)
        if mid in new_nodes:
            continue
        new_nodes.add(mid)
        added.append(mid)
    return added


def _interval_scores(score_map: Sequence[Sequence[float]], axis_name: str) -> List[float]:
    rows = len(score_map)
    cols = len(score_map[0]) if rows else 0
    scores: List[float] = []
    if axis_name == "spot":
        for i in range(max(rows - 1, 0)):
            vals: List[float] = []
            for j in range(cols):
                vals.append(score_map[i][j])
                vals.append(score_map[i + 1][j])
            scores.append(mean_ignore_nan(vals))
        return scores
    for j in range(max(cols - 1, 0)):
        vals = []
        for i in range(rows):
            vals.append(score_map[i][j])
            vals.append(score_map[i][j + 1])
        scores.append(mean_ignore_nan(vals))
    return scores


def _interval_crosses_regime(
    regime_mask: Sequence[Sequence[float]] | None,
    axis_name: str,
    idx: int,
) -> bool:
    if not regime_mask:
        return False
    rows = len(regime_mask)
    cols = len(regime_mask[0]) if rows else 0
    if axis_name == "spot":
        if idx < 0 or idx >= rows:
            return False
        return any(regime_mask[idx][j] > 0.5 for j in range(cols))
    if idx < 0 or idx >= cols:
        return False
    return any(regime_mask[i][idx] > 0.5 for i in range(rows))


def _error_score(delta: float, gamma: float, vega: float, config: GridCheckConfig) -> float:
    ratios = []
    if config.refine_delta_threshold > 0.0:
        ratios.append(delta / config.refine_delta_threshold)
    if config.refine_gamma_threshold > 0.0:
        ratios.append(gamma / config.refine_gamma_threshold)
    if config.refine_vega_threshold > 0.0:
        ratios.append(vega / config.refine_vega_threshold)
    return max(ratios) if ratios else 0.0


def _node_in_regime(regime_mask: Sequence[Sequence[float]], i: int, j: int) -> bool:
    if not regime_mask:
        return False
    rows = len(regime_mask)
    cols = len(regime_mask[0]) if rows else 0
    for di in (0, -1):
        for dj in (0, -1):
            ci = i + di
            cj = j + dj
            if 0 <= ci < rows and 0 <= cj < cols and regime_mask[ci][cj] > 0.5:
                return True
    return False


def _add_midpoint(
    xs: Sequence[float],
    container: set[float],
    left: int,
    right: int,
    max_new_nodes: int,
) -> bool:
    if left < 0 or right >= len(xs):
        return False
    if len(container) - len(xs) >= max_new_nodes:
        return False
    mid = 0.5 * (xs[left] + xs[right])
    if mid in container:
        return False
    container.add(mid)
    return True




def write_report(
    result: GridCheckResult,
    config: GridCheckConfig,
    meta: Optional[dict] = None,
) -> Tuple[str, str]:
    os.makedirs(config.output_dir, exist_ok=True)
    stamp = result.summary.asof.strftime("%Y%m%d")
    json_path = os.path.join(config.output_dir, f"grid_check_{stamp}.json")
    html_path = os.path.join(config.output_dir, f"grid_check_{stamp}.html")

    summary_payload = dict(result.summary.__dict__)
    summary_payload["asof"] = summary_payload["asof"].isoformat()

    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "meta": meta or {},
                "summary": summary_payload,
                "delta_error": result.delta_error_map.values,
                "gamma_error": result.gamma_error_map.values,
                "vega_error": result.vega_error_map.values,
                "cross_gamma_error": result.cross_gamma_error_map.values,
                "dividend_error": result.dividend_error_map.values,
                "dividend_cross_error": result.dividend_cross_error_map.values,
                "rate_error": result.rate_error_map.values,
                "rate_cross_error": result.rate_cross_error_map.values,
                "funding_error": result.funding_error_map.values,
                "funding_cross_error": result.funding_cross_error_map.values,
                "borrow_error": result.borrow_error_map.values,
                "borrow_cross_error": result.borrow_cross_error_map.values,
                "regime_crossing": result.regime_map.values,
                "interp_error": result.interp_error_map.values,
                "error_score": result.score_map.values,
                "refine_flags": result.refine_map.values,
                "refinement_plan": result.refinement.__dict__,
            },
            handle,
            indent=2,
        )

    html = render_error_map_html(result)
    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(html)

    return json_path, html_path


def render_error_map_html(result: GridCheckResult) -> str:
    blocks = [
        ("Delta error", result.delta_error_map.values),
        ("Gamma error", result.gamma_error_map.values),
        ("Vega error", result.vega_error_map.values),
        ("Cross-Gamma error", result.cross_gamma_error_map.values),
        ("Dividend error", result.dividend_error_map.values),
        ("Dividend cross error", result.dividend_cross_error_map.values),
        ("Rate error", result.rate_error_map.values),
        ("Rate cross error", result.rate_cross_error_map.values),
        ("Funding error", result.funding_error_map.values),
        ("Funding cross error", result.funding_cross_error_map.values),
        ("Borrow error", result.borrow_error_map.values),
        ("Borrow cross error", result.borrow_cross_error_map.values),
        ("Barrier regime crossings", result.regime_map.values),
        ("Interpolation error", result.interp_error_map.values),
        ("Error score", result.score_map.values),
        ("Refine flags", result.refine_map.values),
    ]

    def color(val: float) -> str:
        if math.isnan(val):
            return "rgb(160,160,160)"
        v = min(max(val, 0.0), 0.5)
        t = v / 0.5
        r = int(40 + 180 * t)
        g = int(180 - 100 * t)
        b = int(220 - 140 * t)
        return f"rgb({r},{g},{b})"

    sections = []
    for title, grid in blocks:
        rows = []
        for row in grid:
            cells = "".join(
                f"<div class='cell' style='background:{color(val)}' title='{val:.4f}'></div>"
                for val in row
            )
            rows.append(f"<div class='row'>{cells}</div>")
        sections.append(
            f"<section><h3>{title}</h3><div class='map'>{''.join(rows)}</div></section>"
        )

    summary = result.summary
    refinement = result.refinement
    return f"""
<!doctype html>
<html lang='en'>
<head>
  <meta charset='utf-8'>
  <title>Grid Sensitivity Checks</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; background: #f7f4ee; }}
    h1 {{ margin-top: 0; }}
    section {{ margin-bottom: 28px; }}
    .map {{ display: inline-block; border: 1px solid #d3c7b9; padding: 6px; background: #fff; }}
    .row {{ display: grid; grid-auto-flow: column; gap: 2px; margin-bottom: 2px; }}
    .cell {{ width: 14px; height: 14px; border-radius: 2px; }}
    .summary {{ display: grid; gap: 6px; font-size: 14px; }}
  </style>
</head>
<body>
  <h1>Daily Grid Sensitivity Check</h1>
    <div class='summary'>
        <div>Date: {summary.asof}</div>
        <div>Base value: {summary.base_value:,.2f}</div>
        <div>Mean rel delta error: {summary.mean_rel_delta_error:.4f}</div>
        <div>Mean rel gamma error: {summary.mean_rel_gamma_error:.4f}</div>
        <div>Mean rel vega error: {summary.mean_rel_vega_error:.4f}</div>
        <div>Mean rel cross-gamma error: {summary.mean_rel_cross_gamma_error:.4f}</div>
        <div>Mean rel dividend error: {summary.mean_rel_dividend_error:.4f}</div>
        <div>Mean rel dividend cross error: {summary.mean_rel_dividend_cross_error:.4f}</div>
        <div>Mean rel rate error: {summary.mean_rel_rate_error:.4f}</div>
        <div>Mean rel rate cross error: {summary.mean_rel_rate_cross_error:.4f}</div>
        <div>Mean rel funding error: {summary.mean_rel_funding_error:.4f}</div>
        <div>Mean rel funding cross error: {summary.mean_rel_funding_cross_error:.4f}</div>
        <div>Mean rel borrow error: {summary.mean_rel_borrow_error:.4f}</div>
        <div>Mean rel borrow cross error: {summary.mean_rel_borrow_cross_error:.4f}</div>
        <div>Interp mean rel error: {summary.interp_mean_rel_error:.4f}</div>
        <div>Interp max rel error: {summary.interp_max_rel_error:.4f}</div>
        <div>Refine thresholds (Delta/Gamma/Vega): {refinement.delta_threshold:.3f} / {refinement.gamma_threshold:.3f} / {refinement.vega_threshold:.3f}</div>
        <div>Flagged cells: {refinement.flagged_cells} (unrefined: {refinement.unrefined_cells})</div>
        <div>RNIV add-on (grid residual): {refinement.rniv_addon:,.2f} (mult={refinement.rniv_multiplier:.2f})</div>
        <div>Suggested spot nodes: {", ".join(f"{x:.4f}" for x in refinement.spot_nodes)}</div>
        <div>Suggested vol shifts: {", ".join(f"{x:.4f}" for x in refinement.vol_shifts)}</div>
        <div>Added spot nodes: {", ".join(f"{x:.4f}" for x in refinement.added_spot_nodes) or "none"}</div>
        <div>Added vol nodes: {", ".join(f"{x:.4f}" for x in refinement.added_vol_nodes) or "none"}</div>
    </div>
    {''.join(sections)}
</body>
</html>
"""
