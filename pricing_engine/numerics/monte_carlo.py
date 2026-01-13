"""Monte Carlo pricing engine skeleton."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Dict, Iterable, List, Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.base import Engine, EngineSettings
from pricing_engine.products.base import Product
from pricing_engine.products.exotics import BarrierOption, VarianceSwap, VolSwap
from pricing_engine.products.vanilla import AmericanOption
from pricing_engine.utils.types import BarrierType, OptionType


@dataclass
class MonteCarloEngine(Engine):
    num_paths: int = 20000
    timesteps: int = 252
    use_brownian_bridge: bool = True
    barrier_monitoring: str = "continuous"
    lsm_basis_degree: int = 2
    random_mode: str = "pseudo"
    use_antithetic: bool = False
    use_moment_matching: bool = False
    use_control_variate: bool = False
    halton_skip: int = 0
    halton_leap: int = 0

    @property
    def name(self) -> str:
        return "monte_carlo"

    def _effective_sigma(self, model: Model, market: MarketDataSnapshot, horizon: float) -> float:
        if hasattr(model, "sigma"):
            return float(getattr(model, "sigma"))
        if hasattr(model, "v0"):
            return math.sqrt(max(float(getattr(model, "v0")), 0.0))
        if hasattr(model, "theta"):
            return math.sqrt(max(float(getattr(model, "theta")), 0.0))
        return market.vol_surface.implied_vol(max(horizon, 1e-6), market.spot)

    def _simulate_bs_paths(
        self,
        market: MarketDataSnapshot,
        model: BlackScholesModel,
        horizon: float,
        settings: EngineSettings,
    ) -> Sequence[Sequence[float]]:
        steps = max(self.timesteps, 1)
        dt = max(horizon, 0.0) / steps
        r = market.funding_curve.zero_rate(max(horizon, 0.0))
        q = market.dividends.yield_rate(max(horizon, 0.0))
        b = market.borrow_curve.zero_rate(max(horizon, 0.0))
        drift = (r - q - b - 0.5 * model.sigma * model.sigma) * dt
        normals = self._normal_matrix(self.num_paths, steps, settings.seed)
        paths = []
        for row in normals:
            spot = market.spot
            path = [spot]
            for z in row:
                spot *= math.exp(drift + model.sigma * math.sqrt(dt) * z)
                path.append(spot)
            paths.append(path)
        return paths

    def _barrier_hit(self, path: Iterable[float], barrier: float, barrier_type: BarrierType, sigma: float, dt: float) -> bool:
        if dt <= 0.0 or sigma <= 0.0 or self.barrier_monitoring == "discrete" or not self.use_brownian_bridge:
            if barrier_type in (BarrierType.UP_IN, BarrierType.UP_OUT):
                return any(s >= barrier for s in path)
            return any(s <= barrier for s in path)

        path_list = list(path)
        if barrier_type in (BarrierType.UP_IN, BarrierType.UP_OUT):
            for i in range(1, len(path_list)):
                s0 = path_list[i - 1]
                s1 = path_list[i]
                if s0 >= barrier or s1 >= barrier:
                    return True
                if s0 <= 0.0 or s1 <= 0.0:
                    continue
                log_ratio0 = math.log(barrier / s0)
                log_ratio1 = math.log(barrier / s1)
                prob = math.exp(-2.0 * log_ratio0 * log_ratio1 / (sigma * sigma * dt))
                if random.random() < prob:
                    return True
            return False

        for i in range(1, len(path_list)):
            s0 = path_list[i - 1]
            s1 = path_list[i]
            if s0 <= barrier or s1 <= barrier:
                return True
            if s0 <= 0.0 or s1 <= 0.0:
                continue
            log_ratio0 = math.log(s0 / barrier)
            log_ratio1 = math.log(s1 / barrier)
            prob = math.exp(-2.0 * log_ratio0 * log_ratio1 / (sigma * sigma * dt))
            if random.random() < prob:
                return True
        return False

    def _normal_matrix(self, num_paths: int, steps: int, seed: int | None) -> List[List[float]]:
        total_paths = max(num_paths, 1)
        base_paths = (total_paths + 1) // 2 if self.use_antithetic else total_paths
        if self.random_mode == "halton":
            normals = self._halton_normals(base_paths, steps, seed)
        elif self.random_mode == "pseudo":
            rng = random.Random(seed)
            normals = [[rng.gauss(0.0, 1.0) for _ in range(steps)] for _ in range(base_paths)]
        else:
            raise ValueError(f"Unsupported random_mode: {self.random_mode}")

        if self.use_antithetic:
            normals = normals + [[-z for z in row] for row in normals]
            normals = normals[:total_paths]

        if self.use_moment_matching:
            normals = self._moment_match(normals)

        return normals

    def _moment_match(self, normals: List[List[float]]) -> List[List[float]]:
        if not normals:
            return normals
        rows = len(normals)
        cols = len(normals[0])
        for j in range(cols):
            col = [normals[i][j] for i in range(rows)]
            mean = sum(col) / rows
            var = sum((x - mean) ** 2 for x in col) / max(rows - 1, 1)
            std = math.sqrt(var) if var > 0.0 else 1.0
            for i in range(rows):
                normals[i][j] = (normals[i][j] - mean) / std
        return normals

    def _halton_normals(self, num_paths: int, steps: int, seed: int | None) -> List[List[float]]:
        start = max((seed or 0) + 1 + self.halton_skip, 1)
        leap = max(self.halton_leap, 0)
        primes = self._first_primes(steps)
        normals: List[List[float]] = []
        index = start
        for _ in range(num_paths):
            row = []
            for base in primes:
                u = self._halton_value(index, base)
                u = min(max(u, 1e-12), 1.0 - 1e-12)
                row.append(self._norm_inv(u))
            normals.append(row)
            index += 1 + leap
        return normals

    def _halton_value(self, index: int, base: int) -> float:
        f = 1.0
        r = 0.0
        i = index
        while i > 0:
            f /= base
            r += f * (i % base)
            i //= base
        return r

    def _first_primes(self, count: int) -> List[int]:
        primes: List[int] = []
        n = 2
        while len(primes) < count:
            is_prime = True
            limit = int(math.sqrt(n))
            for p in primes:
                if p > limit:
                    break
                if n % p == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(n)
            n += 1
        return primes

    def _norm_inv(self, p: float) -> float:
        # Acklam's approximation for inverse normal CDF.
        a = [
            -3.969683028665376e01,
            2.209460984245205e02,
            -2.759285104469687e02,
            1.383577518672690e02,
            -3.066479806614716e01,
            2.506628277459239e00,
        ]
        b = [
            -5.447609879822406e01,
            1.615858368580409e02,
            -1.556989798598866e02,
            6.680131188771972e01,
            -1.328068155288572e01,
        ]
        c = [
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e00,
            -2.549732539343734e00,
            4.374664141464968e00,
            2.938163982698783e00,
        ]
        d = [
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e00,
            3.754408661907416e00,
        ]
        plow = 0.02425
        phigh = 1.0 - plow
        if p < plow:
            q = math.sqrt(-2.0 * math.log(p))
            return (
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            )
        if p > phigh:
            q = math.sqrt(-2.0 * math.log(1.0 - p))
            return -(
                (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
            )
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )

    def _basis(self, s: float) -> List[float]:
        basis = [1.0]
        if self.lsm_basis_degree >= 1:
            basis.append(s)
        if self.lsm_basis_degree >= 2:
            basis.append(s * s)
        if self.lsm_basis_degree >= 3:
            basis.append(s * s * s)
        return basis

    def _solve_linear(self, a: List[List[float]], b: List[float]) -> List[float]:
        n = len(b)
        aug = [row[:] + [b[i]] for i, row in enumerate(a)]
        for i in range(n):
            pivot = i
            for j in range(i + 1, n):
                if abs(aug[j][i]) > abs(aug[pivot][i]):
                    pivot = j
            if abs(aug[pivot][i]) < 1e-12:
                return [0.0] * n
            aug[i], aug[pivot] = aug[pivot], aug[i]
            factor = aug[i][i]
            for k in range(i, n + 1):
                aug[i][k] /= factor
            for j in range(n):
                if j == i:
                    continue
                factor = aug[j][i]
                for k in range(i, n + 1):
                    aug[j][k] -= factor * aug[i][k]
        return [aug[i][-1] for i in range(n)]

    def _least_squares(self, xs: List[List[float]], ys: List[float]) -> List[float]:
        if not xs:
            return []
        m = len(xs[0])
        xtx = [[0.0 for _ in range(m)] for _ in range(m)]
        xty = [0.0 for _ in range(m)]
        for x, y in zip(xs, ys):
            for i in range(m):
                xty[i] += x[i] * y
                for j in range(m):
                    xtx[i][j] += x[i] * x[j]
        return self._solve_linear(xtx, xty)

    def _lsm_american(self, product: AmericanOption, paths: List[List[float]], r: float, dt: float) -> float:
        if not paths or dt <= 0.0:
            return 0.0
        disc = math.exp(-r * dt)
        cashflows = [product.payoff(path[-1]) for path in paths]
        steps = len(paths[0]) - 1
        for step in range(steps - 1, 0, -1):
            cashflows = [cf * disc for cf in cashflows]
            spots = [path[step] for path in paths]
            exercise = [product.payoff(s) for s in spots]
            itm_indices = [i for i, ex in enumerate(exercise) if ex > 0.0]
            if len(itm_indices) >= 2:
                xs = [self._basis(spots[i]) for i in itm_indices]
                ys = [cashflows[i] for i in itm_indices]
                if len(xs) >= len(xs[0]):
                    beta = self._least_squares(xs, ys)
                    continuation = [sum(b * x for b, x in zip(beta, self._basis(spots[i]))) for i in itm_indices]
                else:
                    avg = sum(ys) / len(ys)
                    continuation = [avg] * len(itm_indices)
                for idx, cont in zip(itm_indices, continuation):
                    if exercise[idx] >= cont:
                        cashflows[idx] = exercise[idx]
            else:
                for idx in itm_indices:
                    cashflows[idx] = exercise[idx]
        cashflows = [cf * disc for cf in cashflows]
        return sum(cashflows) / len(cashflows)

    def price(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        settings: EngineSettings,
    ) -> Dict[str, float]:
        horizon = market.time_to(product.maturity)
        if isinstance(model, BlackScholesModel):
            paths = self._simulate_bs_paths(market, model, horizon, settings)
        else:
            paths = model.simulate_paths(
                market=market,
                timesteps=self.timesteps,
                num_paths=self.num_paths,
                horizon=horizon,
                seed=settings.seed,
            )
        if isinstance(product, AmericanOption):
            r = market.discount_curve.zero_rate(max(horizon, 0.0))
            dt = max(horizon, 0.0) / max(self.timesteps, 1)
            price = self._lsm_american(product, list(paths), r, dt)
            return {"price": price}
        payoffs = []
        terminal_spots = []
        sigma = self._effective_sigma(model, market, horizon)
        dt = max(horizon, 0.0) / max(self.timesteps, 1)
        for path in paths:
            terminal_spots.append(path[-1] if path else 0.0)
            if isinstance(product, VarianceSwap):
                if len(path) < 2 or horizon <= 0.0:
                    payoffs.append(0.0)
                    continue
                log_returns = [
                    math.log(path[i] / path[i - 1]) for i in range(1, len(path)) if path[i - 1] > 0.0
                ]
                realized_var = sum(lr * lr for lr in log_returns) / max(horizon, 1e-12)
                payoffs.append(product.notional * (realized_var - product.strike))
                continue
            if isinstance(product, VolSwap):
                if len(path) < 2 or horizon <= 0.0:
                    payoffs.append(0.0)
                    continue
                log_returns = [
                    math.log(path[i] / path[i - 1]) for i in range(1, len(path)) if path[i - 1] > 0.0
                ]
                realized_var = sum(lr * lr for lr in log_returns) / max(horizon, 1e-12)
                realized_vol = math.sqrt(max(realized_var, 0.0))
                payoffs.append(product.notional * (realized_vol - product.strike))
                continue
            if isinstance(product, BarrierOption):
                hit = self._barrier_hit(path, product.barrier, product.barrier_type, sigma, dt)
                is_in = product.barrier_type in (BarrierType.UP_IN, BarrierType.DOWN_IN)
                active = hit if is_in else not hit
                if not active:
                    payoffs.append(product.rebate)
                    continue
                spot = path[-1]
                if product.option_type == OptionType.CALL:
                    payoffs.append(max(spot - product.strike, 0.0))
                else:
                    payoffs.append(max(product.strike - spot, 0.0))
                continue

            if product.is_path_dependent():
                payoffs.append(product.payoff(path))
            else:
                payoffs.append(product.payoff(path[-1]))

        disc = market.discount_curve.df(horizon)
        mean_payoff = sum(payoffs) / len(payoffs)
        diagnostics: Dict[str, float] = {}

        if self.use_control_variate and not isinstance(product, AmericanOption):
            r = market.discount_curve.zero_rate(max(horizon, 0.0))
            q = market.dividends.yield_rate(max(horizon, 0.0))
            b = market.borrow_curve.zero_rate(max(horizon, 0.0))
            forward = market.spot * math.exp((r - q - b) * max(horizon, 0.0))
            cov = 0.0
            var_s = 0.0
            for payoff, st in zip(payoffs, terminal_spots):
                cov += (payoff - mean_payoff) * (st - forward)
                var_s += (st - forward) ** 2
            if var_s > 0.0:
                beta = cov / var_s
                diagnostics["control_variate_beta"] = beta
                payoffs = [p + beta * (forward - st) for p, st in zip(payoffs, terminal_spots)]
                mean_payoff = sum(payoffs) / len(payoffs)

        var = sum((p - mean_payoff) ** 2 for p in payoffs) / max(len(payoffs) - 1, 1)
        stderr = disc * math.sqrt(var / max(len(payoffs), 1))
        price = disc * mean_payoff
        result = {"price": price, "stderr": stderr}
        if diagnostics:
            result.update({k: float(v) for k, v in diagnostics.items()})
        return result
