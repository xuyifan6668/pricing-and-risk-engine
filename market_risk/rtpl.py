"""Risk-theoretical PnL from portfolio sensitivities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.dividends import DividendSchedule

from market_risk.derivatives import (
    cross_derivative,
    first_derivative,
    second_derivative,
)
from market_risk.grid_approx import (
    GridConfig,
    PortfolioGridApproximator,
    apply_spot_vol_shift,
    barrier_cross_map,
    implied_vol_at,
)


@dataclass(frozen=True)
class PortfolioSensitivities:
    delta: float
    gamma: float
    vega: float
    cross_gamma: float
    dividend: float
    rho: float
    funding_rho: float
    borrow_rho: float
    spot_rate_cross: float
    spot_funding_cross: float
    spot_borrow_cross: float
    spot_dividend_cross: float


@dataclass(frozen=True)
class GridRTPLConfig:
    grid_config: GridConfig
    dividend_shifts: Sequence[float]
    rate_shifts: Sequence[float] = (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)
    funding_shifts: Sequence[float] = (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)
    borrow_shifts: Sequence[float] = (-0.005, -0.003, -0.001, 0.0, 0.001, 0.003, 0.005)


def compute_sensitivities(
    portfolio,
    market: MarketDataSnapshot,
    spot_bp: float = 1e-4,
    vol_bp: float = 1e-4,
    div_bp: float = 1e-4,
    rate_bp: float = 1e-4,
    funding_bp: float = 1e-4,
    borrow_bp: float = 1e-4,
) -> PortfolioSensitivities:
    spot = market.spot
    if spot <= 0.0:
        return PortfolioSensitivities(
            delta=0.0,
            gamma=0.0,
            vega=0.0,
            cross_gamma=0.0,
            dividend=0.0,
            rho=0.0,
            funding_rho=0.0,
            borrow_rho=0.0,
            spot_rate_cross=0.0,
            spot_funding_cross=0.0,
            spot_borrow_cross=0.0,
            spot_dividend_cross=0.0,
        )

    base_val = portfolio.value(market)

    market_up = apply_spot_vol_shift(market, 1.0 + spot_bp, 0.0)
    market_dn = apply_spot_vol_shift(market, 1.0 - spot_bp, 0.0)
    p_up = portfolio.value(market_up)
    p_dn = portfolio.value(market_dn)

    delta = (p_up - p_dn) / (2.0 * spot * spot_bp)
    gamma = (p_up - 2.0 * base_val + p_dn) / (spot * spot * spot_bp * spot_bp)

    market_vu = apply_spot_vol_shift(market, 1.0, vol_bp)
    market_vd = apply_spot_vol_shift(market, 1.0, -vol_bp)
    p_vu = portfolio.value(market_vu)
    p_vd = portfolio.value(market_vd)
    vega = (p_vu - p_vd) / (2.0 * vol_bp)

    market_pp = apply_spot_vol_shift(market, 1.0 + spot_bp, vol_bp)
    market_pm = apply_spot_vol_shift(market, 1.0 + spot_bp, -vol_bp)
    market_mp = apply_spot_vol_shift(market, 1.0 - spot_bp, vol_bp)
    market_mm = apply_spot_vol_shift(market, 1.0 - spot_bp, -vol_bp)
    p_pp = portfolio.value(market_pp)
    p_pm = portfolio.value(market_pm)
    p_mp = portfolio.value(market_mp)
    p_mm = portfolio.value(market_mm)
    cross_gamma = (p_pp - p_pm - p_mp + p_mm) / (4.0 * spot * spot_bp * vol_bp)

    div_up = apply_dividend_shift(market, div_bp)
    div_dn = apply_dividend_shift(market, -div_bp)
    p_div_up = portfolio.value(div_up)
    p_div_dn = portfolio.value(div_dn)
    dividend = (p_div_up - p_div_dn) / (2.0 * div_bp)
    spot_div_cross = bump_spot_dividend_cross(portfolio, market, spot_bp, div_bp)

    rho = bump_curve_sensitivity(portfolio, market, rate_bp, curve_name="discount")
    funding_rho = bump_curve_sensitivity(portfolio, market, funding_bp, curve_name="funding")
    borrow_rho = bump_curve_sensitivity(portfolio, market, borrow_bp, curve_name="borrow")
    spot_rate_cross = bump_spot_curve_cross(portfolio, market, spot_bp, rate_bp, curve_name="discount")
    spot_funding_cross = bump_spot_curve_cross(portfolio, market, spot_bp, funding_bp, curve_name="funding")
    spot_borrow_cross = bump_spot_curve_cross(portfolio, market, spot_bp, borrow_bp, curve_name="borrow")

    return PortfolioSensitivities(
        delta=delta,
        gamma=gamma,
        vega=vega,
        cross_gamma=cross_gamma,
        dividend=dividend,
        rho=rho,
        funding_rho=funding_rho,
        borrow_rho=borrow_rho,
        spot_rate_cross=spot_rate_cross,
        spot_funding_cross=spot_funding_cross,
        spot_borrow_cross=spot_borrow_cross,
        spot_dividend_cross=spot_div_cross,
    )


def compute_rtpl_series(
    portfolio,
    history: Sequence[MarketDataSnapshot],
    t_ref: float = 1.0,
    spot_bp: float = 1e-4,
    vol_bp: float = 1e-4,
    div_bp: float = 1e-4,
    rate_bp: float = 1e-4,
    funding_bp: float = 1e-4,
    borrow_bp: float = 1e-4,
) -> List[float]:
    if len(history) < 2:
        return []

    rtpl_series: List[float] = []
    for i in range(1, len(history)):
        base = history[i - 1]
        curr = history[i]
        sens = compute_sensitivities(
            portfolio,
            base,
            spot_bp=spot_bp,
            vol_bp=vol_bp,
            div_bp=div_bp,
            rate_bp=rate_bp,
            funding_bp=funding_bp,
            borrow_bp=borrow_bp,
        )

        d_spot = curr.spot - base.spot
        d_vol = implied_vol_at(curr, t_ref) - implied_vol_at(base, t_ref)
        d_div = curr.dividends.continuous_yield - base.dividends.continuous_yield
        d_rate = curr.discount_curve.zero_rate(t_ref) - base.discount_curve.zero_rate(t_ref)
        d_funding = curr.funding_curve.zero_rate(t_ref) - base.funding_curve.zero_rate(t_ref)
        d_borrow = curr.borrow_curve.zero_rate(t_ref) - base.borrow_curve.zero_rate(t_ref)

        rtpl = (
            sens.delta * d_spot
            + 0.5 * sens.gamma * d_spot * d_spot
            + sens.vega * d_vol
            + sens.cross_gamma * d_spot * d_vol
            + sens.dividend * d_div
            + sens.rho * d_rate
            + sens.funding_rho * d_funding
            + sens.borrow_rho * d_borrow
            + sens.spot_rate_cross * d_spot * d_rate
            + sens.spot_funding_cross * d_spot * d_funding
            + sens.spot_borrow_cross * d_spot * d_borrow
            + sens.spot_dividend_cross * d_spot * d_div
        )
        rtpl_series.append(rtpl)

    return rtpl_series


def apply_dividend_shift(market: MarketDataSnapshot, div_shift: float) -> MarketDataSnapshot:
    dividends = DividendSchedule(
        discrete=market.dividends.discrete,
        continuous_yield=market.dividends.continuous_yield + div_shift,
    )
    return MarketDataSnapshot(
        asof=market.asof,
        spot=market.spot,
        discount_curve=market.discount_curve,
        funding_curve=market.funding_curve,
        borrow_curve=market.borrow_curve,
        vol_surface=market.vol_surface,
        dividends=dividends,
        corporate_actions=market.corporate_actions,
    )


def compute_grid_sensitivities(
    portfolio,
    market: MarketDataSnapshot,
    config: GridRTPLConfig,
) -> PortfolioSensitivities:
    approx = PortfolioGridApproximator(portfolio, market)
    grid = approx.build_grid(config.grid_config)

    regime_mask = barrier_cross_map(portfolio, market, grid.spot_grid, grid.vol_grid)
    if regime_mask and _cell_crosses_regime(grid.spot_grid, grid.vol_grid, regime_mask, 1.0, 0.0):
        return compute_sensitivities(portfolio, market)

    spot_index = _closest_interior_index(grid.spot_grid, 1.0)
    vol_index = _closest_interior_index(grid.vol_grid, 0.0)

    delta = first_derivative(
        grid.spot_grid,
        grid.grid_values,
        spot_index,
        vol_index,
        scale=market.spot,
    )
    gamma = second_derivative(
        grid.spot_grid,
        grid.grid_values,
        spot_index,
        vol_index,
        scale=market.spot,
    )
    vega = first_derivative(
        grid.vol_grid,
        _transpose(grid.grid_values),
        vol_index,
        spot_index,
        scale=1.0,
    )
    cross = cross_derivative(
        grid.spot_grid,
        grid.vol_grid,
        grid.grid_values,
        spot_index,
        vol_index,
        scale_x=market.spot,
        scale_y=1.0,
    )

    dividend, spot_div_cross = compute_dividend_grid_sensitivities(
        portfolio,
        market,
        grid.spot_grid,
        config.dividend_shifts,
    )
    rho, spot_rate_cross = compute_curve_grid_sensitivities(
        portfolio,
        market,
        grid.spot_grid,
        config.rate_shifts,
        curve_name="discount",
    )
    funding_rho, spot_funding_cross = compute_curve_grid_sensitivities(
        portfolio,
        market,
        grid.spot_grid,
        config.funding_shifts,
        curve_name="funding",
    )
    borrow_rho, spot_borrow_cross = compute_curve_grid_sensitivities(
        portfolio,
        market,
        grid.spot_grid,
        config.borrow_shifts,
        curve_name="borrow",
    )

    return PortfolioSensitivities(
        delta=delta,
        gamma=gamma,
        vega=vega,
        cross_gamma=cross,
        dividend=dividend,
        rho=rho,
        funding_rho=funding_rho,
        borrow_rho=borrow_rho,
        spot_rate_cross=spot_rate_cross,
        spot_funding_cross=spot_funding_cross,
        spot_borrow_cross=spot_borrow_cross,
        spot_dividend_cross=spot_div_cross,
    )


def compute_rtpl_series_grid(
    portfolio,
    history: Sequence[MarketDataSnapshot],
    config: GridRTPLConfig,
    t_ref: float = 1.0,
    max_days: int | None = None,
) -> List[float]:
    if len(history) < 2:
        return []

    if max_days is not None and max_days > 0:
        history = history[-(max_days + 1) :]

    rtpl_series: List[float] = []
    for i in range(1, len(history)):
        base = history[i - 1]
        curr = history[i]
        sens = compute_grid_sensitivities(portfolio, base, config)

        d_spot = curr.spot - base.spot
        d_vol = implied_vol_at(curr, t_ref) - implied_vol_at(base, t_ref)
        d_div = curr.dividends.continuous_yield - base.dividends.continuous_yield
        d_rate = curr.discount_curve.zero_rate(t_ref) - base.discount_curve.zero_rate(t_ref)
        d_funding = curr.funding_curve.zero_rate(t_ref) - base.funding_curve.zero_rate(t_ref)
        d_borrow = curr.borrow_curve.zero_rate(t_ref) - base.borrow_curve.zero_rate(t_ref)

        rtpl = (
            sens.delta * d_spot
            + 0.5 * sens.gamma * d_spot * d_spot
            + sens.vega * d_vol
            + sens.cross_gamma * d_spot * d_vol
            + sens.dividend * d_div
            + sens.rho * d_rate
            + sens.funding_rho * d_funding
            + sens.borrow_rho * d_borrow
            + sens.spot_rate_cross * d_spot * d_rate
            + sens.spot_funding_cross * d_spot * d_funding
            + sens.spot_borrow_cross * d_spot * d_borrow
            + sens.spot_dividend_cross * d_spot * d_div
        )
        rtpl_series.append(rtpl)

    return rtpl_series


def compute_dividend_grid_sensitivities(
    portfolio,
    market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    dividend_shifts: Sequence[float],
) -> tuple[float, float]:
    if not dividend_shifts or len(spot_grid) < 3:
        return 0.0, 0.0
    if 0.0 not in dividend_shifts:
        raise ValueError("dividend_shifts must include 0.0 for grid sensitivities")

    values: List[List[float]] = []
    for shift in dividend_shifts:
        row = []
        for spot_mult in spot_grid:
            market_shift = apply_dividend_shift(apply_spot_vol_shift(market, spot_mult, 0.0), shift)
            row.append(portfolio.value(market_shift))
        values.append(row)

    spot_index = _closest_interior_index(spot_grid, 1.0)
    div_index = _closest_interior_index(dividend_shifts, 0.0)
    dividend = first_derivative(dividend_shifts, values, div_index, spot_index, scale=1.0)
    spot_div_cross = cross_derivative(
        spot_grid,
        dividend_shifts,
        _transpose(values),
        spot_index,
        div_index,
        scale_x=market.spot,
        scale_y=1.0,
    )
    return dividend, spot_div_cross


def compute_curve_grid_sensitivities(
    portfolio,
    market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    curve_shifts: Sequence[float],
    curve_name: str,
) -> tuple[float, float]:
    if not curve_shifts or len(spot_grid) < 3:
        return 0.0, 0.0
    if 0.0 not in curve_shifts:
        raise ValueError("curve_shifts must include 0.0 for grid sensitivities")

    values: List[List[float]] = []
    for spot_mult in spot_grid:
        row = []
        for shift in curve_shifts:
            market_shift = apply_curve_shift(apply_spot_vol_shift(market, spot_mult, 0.0), shift, curve_name)
            row.append(portfolio.value(market_shift))
        values.append(row)

    spot_index = _closest_interior_index(spot_grid, 1.0)
    curve_index = _closest_interior_index(curve_shifts, 0.0)
    rho = first_derivative(curve_shifts, _transpose(values), curve_index, spot_index, scale=1.0)
    spot_curve_cross = cross_derivative(
        spot_grid,
        curve_shifts,
        values,
        spot_index,
        curve_index,
        scale_x=market.spot,
        scale_y=1.0,
    )
    return rho, spot_curve_cross


def _closest_interior_index(xs: Sequence[float], target: float) -> int:
    if len(xs) < 3:
        return 0
    closest = min(range(len(xs)), key=lambda i: abs(xs[i] - target))
    return max(1, min(len(xs) - 2, closest))


def _cell_crosses_regime(
    xs: Sequence[float],
    ys: Sequence[float],
    regime_mask: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> bool:
    if len(xs) < 2 or len(ys) < 2:
        return False
    i = _cell_index(xs, x)
    j = _cell_index(ys, y)
    return regime_mask[i][j] > 0.5


def _cell_index(xs: Sequence[float], value: float) -> int:
    if len(xs) < 2:
        return 0
    if value <= xs[0]:
        return 0
    if value >= xs[-1]:
        return len(xs) - 2
    for i in range(len(xs) - 1):
        if xs[i] <= value <= xs[i + 1]:
            return i
    return len(xs) - 2


def _transpose(values: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*values)]


class _ParallelShiftCurve:
    def __init__(self, base_curve, shift: float) -> None:
        self._base = base_curve
        self._shift = shift

    def df(self, t: float) -> float:
        import math

        rate = self._base.zero_rate(t) + self._shift
        return math.exp(-rate * max(t, 0.0))


def apply_curve_shift(
    market: MarketDataSnapshot,
    shift: float,
    curve_name: str,
) -> MarketDataSnapshot:
    discount_curve = market.discount_curve
    funding_curve = market.funding_curve
    borrow_curve = market.borrow_curve

    if curve_name == "discount":
        discount_curve = _ParallelShiftCurve(discount_curve, shift)
    elif curve_name == "funding":
        funding_curve = _ParallelShiftCurve(funding_curve, shift)
    elif curve_name == "borrow":
        borrow_curve = _ParallelShiftCurve(borrow_curve, shift)
    else:
        raise ValueError(f"Unsupported curve_name: {curve_name}")

    return MarketDataSnapshot(
        asof=market.asof,
        spot=market.spot,
        discount_curve=discount_curve,
        funding_curve=funding_curve,
        borrow_curve=borrow_curve,
        vol_surface=market.vol_surface,
        dividends=market.dividends,
        corporate_actions=market.corporate_actions,
    )


def bump_curve_sensitivity(
    portfolio,
    market: MarketDataSnapshot,
    curve_bp: float,
    curve_name: str,
) -> float:
    market_up = apply_curve_shift(market, curve_bp, curve_name)
    market_dn = apply_curve_shift(market, -curve_bp, curve_name)
    p_up = portfolio.value(market_up)
    p_dn = portfolio.value(market_dn)
    return (p_up - p_dn) / (2.0 * curve_bp)


def bump_spot_curve_cross(
    portfolio,
    market: MarketDataSnapshot,
    spot_bp: float,
    curve_bp: float,
    curve_name: str,
) -> float:
    market_pp = apply_curve_shift(apply_spot_vol_shift(market, 1.0 + spot_bp, 0.0), curve_bp, curve_name)
    market_pm = apply_curve_shift(apply_spot_vol_shift(market, 1.0 + spot_bp, 0.0), -curve_bp, curve_name)
    market_mp = apply_curve_shift(apply_spot_vol_shift(market, 1.0 - spot_bp, 0.0), curve_bp, curve_name)
    market_mm = apply_curve_shift(apply_spot_vol_shift(market, 1.0 - spot_bp, 0.0), -curve_bp, curve_name)
    p_pp = portfolio.value(market_pp)
    p_pm = portfolio.value(market_pm)
    p_mp = portfolio.value(market_mp)
    p_mm = portfolio.value(market_mm)
    ds = market.spot * spot_bp * 2.0
    dr = curve_bp * 2.0
    if ds == 0.0 or dr == 0.0:
        return 0.0
    return (p_pp - p_pm - p_mp + p_mm) / (ds * dr)


def bump_spot_dividend_cross(
    portfolio,
    market: MarketDataSnapshot,
    spot_bp: float,
    div_bp: float,
) -> float:
    market_pp = apply_dividend_shift(apply_spot_vol_shift(market, 1.0 + spot_bp, 0.0), div_bp)
    market_pm = apply_dividend_shift(apply_spot_vol_shift(market, 1.0 + spot_bp, 0.0), -div_bp)
    market_mp = apply_dividend_shift(apply_spot_vol_shift(market, 1.0 - spot_bp, 0.0), div_bp)
    market_mm = apply_dividend_shift(apply_spot_vol_shift(market, 1.0 - spot_bp, 0.0), -div_bp)
    p_pp = portfolio.value(market_pp)
    p_pm = portfolio.value(market_pm)
    p_mp = portfolio.value(market_mp)
    p_mm = portfolio.value(market_mm)
    ds = market.spot * spot_bp * 2.0
    dq = div_bp * 2.0
    if ds == 0.0 or dq == 0.0:
        return 0.0
    return (p_pp - p_pm - p_mp + p_mm) / (ds * dq)
