"""Local stochastic volatility leverage calibration."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import List, Sequence, Tuple

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.surfaces import LocalVolSurfaceFromIV
from pricing_engine.models.local_stochastic_vol import LeverageSurface, LocalStochasticVolModel


@dataclass(frozen=True)
class LSVLeverageSettings:
    times: Sequence[float]
    spots: Sequence[float]
    min_leverage: float = 0.1
    max_leverage: float = 5.0
    dt: float = 1e-3
    dk_rel: float = 0.02


@dataclass(frozen=True)
class LSVParticleSettings:
    times: Sequence[float]
    spots: Sequence[float]
    num_paths: int = 20000
    substeps: int = 2
    iterations: int = 4
    bandwidth: float = 0.2
    kernel: str = "log"
    damping: float = 0.5
    min_leverage: float = 0.1
    max_leverage: float = 5.0
    dt: float = 1e-3
    dk_rel: float = 0.02
    seed: int | None = 42


def build_leverage_surface(
    market: MarketDataSnapshot,
    model: LocalStochasticVolModel,
    settings: LSVLeverageSettings,
) -> LeverageSurface:
    local_surface = LocalVolSurfaceFromIV(
        implied_surface=market.vol_surface,
        spot_ref=market.spot,
        dt=settings.dt,
        dk_rel=settings.dk_rel,
    )

    values = []
    for t in settings.times:
        r = market.discount_curve.zero_rate(t)
        q = market.dividends.yield_rate(t) + market.borrow_curve.zero_rate(t)
        ev = max(model.expected_variance(t), 1e-8)
        row = []
        for s in settings.spots:
            sigma_loc = local_surface.local_vol(t, s, r, q)
            leverage = sigma_loc / math.sqrt(ev)
            leverage = min(max(leverage, settings.min_leverage), settings.max_leverage)
            row.append(leverage)
        values.append(row)

    return LeverageSurface(times=settings.times, spots=settings.spots, values=values)


def _simulate_lsv_paths(
    market: MarketDataSnapshot,
    model: LocalStochasticVolModel,
    leverage: LeverageSurface,
    settings: LSVParticleSettings,
    rng: random.Random,
) -> Tuple[List[List[float]], List[List[float]]]:
    times = list(settings.times)
    spots_paths: List[List[float]] = []
    vars_paths: List[List[float]] = []

    num_paths = max(settings.num_paths, 1)
    steps_per = max(settings.substeps, 1)

    s_vals = [market.spot for _ in range(num_paths)]
    v_vals = [max(model.v0, 0.0) for _ in range(num_paths)]

    prev_t = 0.0
    for t in times:
        dt_total = max(t - prev_t, 0.0)
        dt = dt_total / steps_per if steps_per > 0 else 0.0
        for _ in range(steps_per):
            if dt <= 0.0:
                continue
            r = market.funding_curve.zero_rate(max(prev_t + dt, 0.0))
            q = market.dividends.yield_rate(max(prev_t + dt, 0.0)) + market.borrow_curve.zero_rate(max(prev_t + dt, 0.0))
            for i in range(num_paths):
                z1 = rng.gauss(0.0, 1.0)
                z2 = model.rho * z1 + math.sqrt(1.0 - model.rho * model.rho) * rng.gauss(0.0, 1.0)
                v = v_vals[i]
                v = max(v + model.kappa * (model.theta - v) * dt + model.sigma * math.sqrt(max(v, 0.0)) * math.sqrt(dt) * z2, 0.0)
                lever = max(leverage.value(prev_t, s_vals[i]), 1e-6)
                eff_var = max(v * lever * lever, 0.0)
                s_vals[i] *= math.exp((r - q - 0.5 * eff_var) * dt + math.sqrt(max(eff_var, 0.0)) * math.sqrt(dt) * z1)
                v_vals[i] = v
            prev_t += dt
        spots_paths.append(list(s_vals))
        vars_paths.append(list(v_vals))
    return spots_paths, vars_paths


def _conditional_expected_variance(
    spots: List[float],
    vars_: List[float],
    grid_spots: Sequence[float],
    bandwidth: float,
    kernel: str,
    spot_ref: float,
) -> List[float]:
    if not spots:
        return [0.0 for _ in grid_spots]
    unconditional = sum(vars_) / len(vars_)
    results: List[float] = []
    use_log = kernel == "log"
    if use_log:
        h = max(bandwidth, 1e-6)
    else:
        h = max(bandwidth * spot_ref, 1e-6) if bandwidth < 1.0 else max(bandwidth, 1e-6)
    for s in grid_spots:
        num = 0.0
        den = 0.0
        for si, vi in zip(spots, vars_):
            if si <= 0.0 or s <= 0.0:
                continue
            x = math.log(si / s) if use_log else (si - s)
            w = math.exp(-0.5 * (x / h) * (x / h))
            num += w * vi
            den += w
        if den <= 1e-12:
            results.append(max(unconditional, 1e-12))
        else:
            results.append(max(num / den, 1e-12))
    return results


def calibrate_leverage_particle(
    market: MarketDataSnapshot,
    model: LocalStochasticVolModel,
    settings: LSVParticleSettings,
) -> LeverageSurface:
    local_surface = LocalVolSurfaceFromIV(
        implied_surface=market.vol_surface,
        spot_ref=market.spot,
        dt=settings.dt,
        dk_rel=settings.dk_rel,
    )
    times = list(settings.times)
    spots = list(settings.spots)

    if model.leverage is not None:
        current = model.leverage
        values = [list(row) for row in current.values]
    else:
        values = [[1.0 for _ in spots] for _ in times]

    rng = random.Random(settings.seed)

    for _ in range(max(settings.iterations, 1)):
        leverage = LeverageSurface(times=times, spots=spots, values=values)
        spot_paths, var_paths = _simulate_lsv_paths(market, model, leverage, settings, rng)

        updated: List[List[float]] = []
        for idx, t in enumerate(times):
            r = market.discount_curve.zero_rate(t)
            q = market.dividends.yield_rate(t) + market.borrow_curve.zero_rate(t)
            expected_var = _conditional_expected_variance(
                spot_paths[idx],
                var_paths[idx],
                spots,
                settings.bandwidth,
                settings.kernel,
                market.spot,
            )
            row = []
            for s, ev in zip(spots, expected_var):
                sigma_loc = local_surface.local_vol(t, s, r, q)
                leverage_val = sigma_loc / math.sqrt(max(ev, 1e-12))
                leverage_val = min(max(leverage_val, settings.min_leverage), settings.max_leverage)
                row.append(leverage_val)
            updated.append(row)

        damp = min(max(settings.damping, 0.0), 1.0)
        values = [
            [
                (1.0 - damp) * values[i][j] + damp * updated[i][j]
                for j in range(len(spots))
            ]
            for i in range(len(times))
        ]

    return LeverageSurface(times=times, spots=spots, values=values)
