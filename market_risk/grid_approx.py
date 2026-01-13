"""Grid-based approximation and sensitivity checks for portfolio risk."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Iterable, List, Sequence, Tuple

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.surfaces import FlatVolSurface, SmileVolSurface
from pricing_engine.products.exotics import BarrierOption

from market_risk.portfolio import Portfolio
from market_risk.scenarios import VolShift, shift_surface
from market_risk.utils import mean, stdev


@dataclass(frozen=True)
class GridConfig:
    spot_min: float = 0.7
    spot_max: float = 1.3
    spot_nodes: Sequence[float] = (0.7, 0.8, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2, 1.3)
    spot_level_bump: float = 0.02
    vol_shifts: Sequence[float] = (-0.08, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05, 0.08)


@dataclass(frozen=True)
class InterpolationSettings:
    use_log_spot: bool = True
    use_total_variance: bool = True
    t_ref: float = 1.0
    variance_floor: float = 1e-8


@dataclass(frozen=True)
class InterpolationContext:
    spot_axis: List[float]
    vol_axis: List[float]
    base_vol: float
    settings: InterpolationSettings


@dataclass(frozen=True)
class GridSensitivity:
    full_delta: float
    full_gamma: float
    full_vega: float
    grid_delta: float
    grid_gamma: float
    grid_vega: float


@dataclass(frozen=True)
class GridAccuracy:
    mean_abs_error: float
    rms_error: float
    max_error: float


@dataclass(frozen=True)
class GridApproxResult:
    base_value: float
    grid_values: List[List[float]]
    spot_grid: List[float]
    vol_grid: List[float]


class PortfolioGridApproximator:
    def __init__(
        self,
        portfolio: Portfolio,
        base_market: MarketDataSnapshot,
        interp_settings: InterpolationSettings | None = None,
    ) -> None:
        self.portfolio = portfolio
        self.base_market = base_market
        self.base_value = portfolio.value(base_market)
        self.interp_settings = interp_settings or InterpolationSettings()

    def build_grid(self, config: GridConfig) -> GridApproxResult:
        spot_grid = build_spot_grid(self.portfolio, self.base_market, config)
        vol_grid = list(config.vol_shifts)

        values: List[List[float]] = []
        for spot_mult in spot_grid:
            row = []
            for vol_shift in vol_grid:
                market = apply_spot_vol_shift(self.base_market, spot_mult, vol_shift)
                row.append(self.portfolio.value(market))
            values.append(row)

        return GridApproxResult(
            base_value=self.base_value,
            grid_values=values,
            spot_grid=spot_grid,
            vol_grid=vol_grid,
        )

    def interpolate(self, grid: GridApproxResult, spot_mult: float, vol_shift: float) -> float:
        context = build_interpolation_context(
            grid.spot_grid,
            grid.vol_grid,
            self.base_market,
            self.interp_settings,
        )
        return bilinear_interpolate_context(context, grid.grid_values, spot_mult, vol_shift)

    def check_sensitivities(
        self,
        grid: GridApproxResult,
        spot_bump: float = 0.01,
        vol_bump: float = 0.01,
    ) -> GridSensitivity:
        base_spot = self.base_market.spot
        p0 = self.base_value

        market_up = apply_spot_vol_shift(self.base_market, 1.0 + spot_bump, 0.0)
        market_dn = apply_spot_vol_shift(self.base_market, 1.0 - spot_bump, 0.0)
        p_up = self.portfolio.value(market_up)
        p_dn = self.portfolio.value(market_dn)

        full_delta = (p_up - p_dn) / (2.0 * base_spot * spot_bump)
        full_gamma = (p_up - 2.0 * p0 + p_dn) / (base_spot * base_spot * spot_bump * spot_bump)

        market_vu = apply_spot_vol_shift(self.base_market, 1.0, vol_bump)
        market_vd = apply_spot_vol_shift(self.base_market, 1.0, -vol_bump)
        p_vu = self.portfolio.value(market_vu)
        p_vd = self.portfolio.value(market_vd)
        full_vega = (p_vu - p_vd) / (2.0 * vol_bump)

        g_up = self.interpolate(grid, 1.0 + spot_bump, 0.0)
        g_dn = self.interpolate(grid, 1.0 - spot_bump, 0.0)
        grid_delta = (g_up - g_dn) / (2.0 * base_spot * spot_bump)
        grid_gamma = (g_up - 2.0 * p0 + g_dn) / (base_spot * base_spot * spot_bump * spot_bump)

        g_vu = self.interpolate(grid, 1.0, vol_bump)
        g_vd = self.interpolate(grid, 1.0, -vol_bump)
        grid_vega = (g_vu - g_vd) / (2.0 * vol_bump)

        return GridSensitivity(
            full_delta=full_delta,
            full_gamma=full_gamma,
            full_vega=full_vega,
            grid_delta=grid_delta,
            grid_gamma=grid_gamma,
            grid_vega=grid_vega,
        )

    def check_interpolation_accuracy(
        self,
        grid: GridApproxResult,
        samples: int = 25,
        seed: int = 7,
    ) -> GridAccuracy:
        rng = random.Random(seed)
        errors: List[float] = []
        context = build_interpolation_context(
            grid.spot_grid,
            grid.vol_grid,
            self.base_market,
            self.interp_settings,
        )

        for _ in range(samples):
            spot_mult = rng.uniform(min(grid.spot_grid), max(grid.spot_grid))
            vol_shift = rng.uniform(min(grid.vol_grid), max(grid.vol_grid))
            approx = bilinear_interpolate_context(context, grid.grid_values, spot_mult, vol_shift)
            full = self.portfolio.value(apply_spot_vol_shift(self.base_market, spot_mult, vol_shift))
            errors.append(approx - full)

        mean_abs = mean(abs(e) for e in errors)
        rms = math.sqrt(mean([e * e for e in errors])) if errors else 0.0
        max_err = max((abs(e) for e in errors), default=0.0)
        return GridAccuracy(mean_abs_error=mean_abs, rms_error=rms, max_error=max_err)

    def approximate_pnls(
        self,
        grid: GridApproxResult,
        scenario_markets: Iterable[MarketDataSnapshot],
        t_ref: float = 1.0,
    ) -> List[Tuple[float, float]]:
        base_vol = implied_vol_at(self.base_market, t_ref)
        context = build_interpolation_context(
            grid.spot_grid,
            grid.vol_grid,
            self.base_market,
            InterpolationSettings(
                use_log_spot=self.interp_settings.use_log_spot,
                use_total_variance=self.interp_settings.use_total_variance,
                t_ref=t_ref,
                variance_floor=self.interp_settings.variance_floor,
            ),
        )
        regime_mask = barrier_cross_map(self.portfolio, self.base_market, grid.spot_grid, grid.vol_grid)
        pnls: List[Tuple[float, float]] = []
        for market in scenario_markets:
            spot_mult = market.spot / self.base_market.spot if self.base_market.spot else 1.0
            scen_vol = implied_vol_at(market, t_ref)
            vol_shift = scen_vol - base_vol
            approx_val = bilinear_interpolate_regime_context(
                context,
                grid.grid_values,
                spot_mult,
                vol_shift,
                regime_mask,
            )
            if math.isnan(approx_val):
                approx_val = self.portfolio.value(market)
            full_val = self.portfolio.value(market)
            pnls.append((approx_val - grid.base_value, full_val - grid.base_value))
        return pnls


def build_spot_grid(portfolio: Portfolio, base_market: MarketDataSnapshot, config: GridConfig) -> List[float]:
    grid = set(config.spot_nodes)

    spot = base_market.spot
    for pos in portfolio.positions:
        level = extract_critical_levels(pos.product)
        for lv in level:
            if spot <= 0.0 or lv <= 0.0:
                continue
            m = lv / spot
            grid.add(m)
            grid.add(max(m * (1.0 - config.spot_level_bump), config.spot_min))
            grid.add(min(m * (1.0 + config.spot_level_bump), config.spot_max))

    grid = {min(max(x, config.spot_min), config.spot_max) for x in grid}
    return sorted(grid)


def extract_critical_levels(product) -> List[float]:
    levels = []
    for attr in ("strike", "barrier", "lower", "upper"):
        if hasattr(product, attr):
            val = getattr(product, attr)
            if isinstance(val, (int, float)):
                levels.append(float(val))
    return levels


def barrier_ratios(portfolio: Portfolio, base_market: MarketDataSnapshot) -> List[float]:
    ratios = []
    spot = base_market.spot
    if spot <= 0.0:
        return ratios
    for pos in portfolio.positions:
        product = pos.product
        if isinstance(product, BarrierOption):
            ratios.append(product.barrier / spot)
    return ratios


def barrier_cross_map(
    portfolio: Portfolio,
    base_market: MarketDataSnapshot,
    spot_grid: Sequence[float],
    vol_grid: Sequence[float],
) -> List[List[float]]:
    ratios = barrier_ratios(portfolio, base_market)
    rows = len(spot_grid) - 1
    cols = len(vol_grid) - 1
    mask = [[0.0 for _ in range(cols)] for _ in range(rows)]
    if not ratios:
        return mask
    for i in range(rows):
        low = spot_grid[i]
        high = spot_grid[i + 1]
        for ratio in ratios:
            if (low - ratio) * (high - ratio) <= 0.0:
                for j in range(cols):
                    mask[i][j] = 1.0
                break
    return mask


def apply_spot_vol_shift(
    base: MarketDataSnapshot,
    spot_mult: float,
    vol_shift: float,
) -> MarketDataSnapshot:
    spot = base.spot * spot_mult
    vol_surface = base.vol_surface
    if isinstance(vol_surface, SmileVolSurface):
        shift = VolShift(atm=[vol_shift] * len(vol_surface.expiries))
    elif isinstance(vol_surface, FlatVolSurface):
        shift = VolShift(flat=vol_shift)
    else:
        shift = VolShift(flat=vol_shift)
    shifted_surface = shift_surface(vol_surface, shift, spot)
    return MarketDataSnapshot(
        asof=base.asof,
        spot=spot,
        discount_curve=base.discount_curve,
        funding_curve=base.funding_curve,
        borrow_curve=base.borrow_curve,
        vol_surface=shifted_surface,
        dividends=base.dividends,
        corporate_actions=base.corporate_actions,
    )


def implied_vol_at(market: MarketDataSnapshot, t_ref: float) -> float:
    return market.vol_surface.implied_vol(t_ref, max(market.spot, 1e-8))


def bilinear_interpolate(
    xs: Sequence[float],
    ys: Sequence[float],
    grid: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> float:
    return bilinear_interpolate_axes(xs, ys, grid, x, y)


def bilinear_interpolate_axes(
    xs: Sequence[float],
    ys: Sequence[float],
    grid: Sequence[Sequence[float]],
    x: float,
    y: float,
) -> float:
    if not xs or not ys:
        return 0.0
    x = min(max(x, xs[0]), xs[-1])
    y = min(max(y, ys[0]), ys[-1])

    ix = _axis_index(xs, x)
    iy = _axis_index(ys, y)
    ix1 = min(ix + 1, len(xs) - 1)
    iy1 = min(iy + 1, len(ys) - 1)

    x0 = xs[ix]
    x1 = xs[ix1]
    y0 = ys[iy]
    y1 = ys[iy1]

    q11 = grid[ix][iy]
    q12 = grid[ix][iy1]
    q21 = grid[ix1][iy]
    q22 = grid[ix1][iy1]

    if x1 == x0 and y1 == y0:
        return q11
    if x1 == x0:
        wy = (y - y0) / (y1 - y0)
        return q11 + wy * (q12 - q11)
    if y1 == y0:
        wx = (x - x0) / (x1 - x0)
        return q11 + wx * (q21 - q11)

    wx = (x - x0) / (x1 - x0)
    wy = (y - y0) / (y1 - y0)
    return (
        q11 * (1 - wx) * (1 - wy)
        + q21 * wx * (1 - wy)
        + q12 * (1 - wx) * wy
        + q22 * wx * wy
    )


def bilinear_interpolate_context(
    context: InterpolationContext,
    grid: Sequence[Sequence[float]],
    spot_mult: float,
    vol_shift: float,
) -> float:
    x = _transform_spot(spot_mult, context.settings)
    y = _transform_vol_shift(vol_shift, context.base_vol, context.settings)
    return bilinear_interpolate_axes(context.spot_axis, context.vol_axis, grid, x, y)


def bilinear_interpolate_regime_context(
    context: InterpolationContext,
    grid: Sequence[Sequence[float]],
    spot_mult: float,
    vol_shift: float,
    regime_mask: Sequence[Sequence[float]],
) -> float:
    if not context.spot_axis or not context.vol_axis:
        return float("nan")
    ix, iy = _context_cell_index(context, spot_mult, vol_shift)
    if regime_mask[ix][iy] > 0.5:
        return float("nan")
    return bilinear_interpolate_context(context, grid, spot_mult, vol_shift)


def build_interpolation_context(
    spot_grid: Sequence[float],
    vol_grid: Sequence[float],
    base_market: MarketDataSnapshot,
    settings: InterpolationSettings,
) -> InterpolationContext:
    base_vol = implied_vol_at(base_market, settings.t_ref)
    spot_axis = [_transform_spot(x, settings) for x in spot_grid]
    vol_axis = [_transform_vol_shift(y, base_vol, settings) for y in vol_grid]
    return InterpolationContext(
        spot_axis=spot_axis,
        vol_axis=vol_axis,
        base_vol=base_vol,
        settings=settings,
    )


def _transform_spot(spot_mult: float, settings: InterpolationSettings) -> float:
    if not settings.use_log_spot:
        return spot_mult
    return math.log(max(spot_mult, 1e-8))


def _transform_vol_shift(
    vol_shift: float,
    base_vol: float,
    settings: InterpolationSettings,
) -> float:
    if not settings.use_total_variance:
        return vol_shift
    t_ref = max(settings.t_ref, 1e-8)
    vol_floor = math.sqrt(settings.variance_floor / t_ref) if settings.variance_floor > 0.0 else 0.0
    vol = max(base_vol + vol_shift, vol_floor)
    return vol * vol * t_ref


def _context_cell_index(
    context: InterpolationContext,
    spot_mult: float,
    vol_shift: float,
) -> Tuple[int, int]:
    x = _transform_spot(spot_mult, context.settings)
    y = _transform_vol_shift(vol_shift, context.base_vol, context.settings)
    ix = _axis_index(context.spot_axis, x)
    iy = _axis_index(context.vol_axis, y)
    ix = min(ix, len(context.spot_axis) - 2)
    iy = min(iy, len(context.vol_axis) - 2)
    return ix, iy


def _axis_index(xs: Sequence[float], value: float) -> int:
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
