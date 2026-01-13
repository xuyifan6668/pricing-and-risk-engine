"""FRTB IMA capital aggregation and desk approval logic."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Iterable, List, Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot

from market_risk.backtesting import BacktestResult
from market_risk.es import ExpectedShortfallCalculator, ESResult
from market_risk.pla import PLATestResult
from market_risk.portfolio import Portfolio
from market_risk.rtpl import GridRTPLConfig, compute_rtpl_series_grid
from market_risk.scenarios import ScenarioSet


@dataclass(frozen=True)
class LiquidityHorizon:
    name: str
    days: int


DEFAULT_LH_MAP: Dict[str, int] = {
    "spot": 10,
    "forward": 10,
    "swap": 10,
    "european_option": 20,
    "american_option": 40,
    "barrier_option": 40,
    "digital_option": 20,
    "asian_option": 40,
    "lookback_option": 60,
    "cliquet": 60,
    "corridor": 60,
    "variance_swap": 20,
    "vol_swap": 20,
    "basket_option": 20,
}


@dataclass(frozen=True)
class IMACapitalResult:
    es_by_lh: Dict[int, float]
    total_es: float
    details: Dict[int, ESResult]


class IMACapitalCalculator:
    def __init__(
        self,
        es_level: float = 0.975,
        base_horizon_days: int = 10,
    ) -> None:
        self.es_level = es_level
        self.base_horizon_days = base_horizon_days
        self.es_calc = ExpectedShortfallCalculator()

    def _lh_for_position(self, product_type: str, lh_map: Dict[str, int]) -> int:
        return lh_map.get(product_type, 20)

    def compute(
        self,
        portfolio: Portfolio,
        base_market: MarketDataSnapshot,
        scenarios: ScenarioSet,
        lh_map: Dict[str, int] | None = None,
    ) -> IMACapitalResult:
        lh_map = lh_map or DEFAULT_LH_MAP
        buckets: Dict[int, List[int]] = {}
        for idx, pos in enumerate(portfolio.positions):
            lh = self._lh_for_position(pos.product.product_type, lh_map)
            buckets.setdefault(lh, []).append(idx)

        es_by_lh: Dict[int, float] = {}
        details: Dict[int, ESResult] = {}
        scenario_markets = scenarios.markets(base_market)

        for lh, indices in buckets.items():
            bucket_positions = [portfolio.positions[i] for i in indices]
            bucket_portfolio = Portfolio(positions=bucket_positions)
            pnls = bucket_portfolio.scenario_pnls(base_market, scenario_markets)
            es = self.es_calc.compute(pnls, self.es_level, scenarios.horizon_days)
            scale = math.sqrt(lh / self.base_horizon_days)
            es_scaled = es.value * scale
            es_by_lh[lh] = es_scaled
            details[lh] = es

        total = math.sqrt(sum(val * val for val in es_by_lh.values())) if es_by_lh else 0.0
        return IMACapitalResult(es_by_lh=es_by_lh, total_es=total, details=details)


@dataclass(frozen=True)
class RTPLCapitalResult:
    es_by_lh: Dict[int, float]
    total_es: float
    details: Dict[int, ESResult]


class RTPLCapitalCalculator:
    def __init__(
        self,
        es_level: float = 0.975,
        base_horizon_days: int = 10,
        t_ref: float = 1.0,
    ) -> None:
        self.es_level = es_level
        self.base_horizon_days = base_horizon_days
        self.t_ref = t_ref
        self.es_calc = ExpectedShortfallCalculator()

    def _lh_for_position(self, product_type: str, lh_map: Dict[str, int]) -> int:
        return lh_map.get(product_type, 20)

    def compute(
        self,
        portfolio: Portfolio,
        history: Sequence[MarketDataSnapshot],
        grid_config: GridRTPLConfig,
        lh_map: Dict[str, int] | None = None,
        max_days: int | None = None,
    ) -> RTPLCapitalResult:
        lh_map = lh_map or DEFAULT_LH_MAP
        if max_days is not None and max_days > 0:
            history = history[-(max_days + 1) :]

        buckets: Dict[int, List[int]] = {}
        for idx, pos in enumerate(portfolio.positions):
            lh = self._lh_for_position(pos.product.product_type, lh_map)
            buckets.setdefault(lh, []).append(idx)

        es_by_lh: Dict[int, float] = {}
        details: Dict[int, ESResult] = {}

        for lh, indices in buckets.items():
            bucket_positions = [portfolio.positions[i] for i in indices]
            bucket_portfolio = Portfolio(positions=bucket_positions)
            pnls = compute_rtpl_series_grid(
                bucket_portfolio,
                history,
                config=grid_config,
                t_ref=self.t_ref,
            )
            es = self.es_calc.compute(pnls, self.es_level, horizon_days=1)
            scale = math.sqrt(lh / self.base_horizon_days)
            es_scaled = es.value * scale
            es_by_lh[lh] = es_scaled
            details[lh] = es

        total = math.sqrt(sum(val * val for val in es_by_lh.values())) if es_by_lh else 0.0
        return RTPLCapitalResult(es_by_lh=es_by_lh, total_es=total, details=details)


@dataclass(frozen=True)
class DeskApprovalResult:
    backtest: BacktestResult
    pla: PLATestResult
    status: str
    reason: str


class DeskApproval:
    def assess(self, backtest: BacktestResult, pla: PLATestResult) -> DeskApprovalResult:
        if backtest.traffic_light == "red":
            return DeskApprovalResult(
                backtest=backtest,
                pla=pla,
                status="fail",
                reason="backtest red",
            )
        if pla.status != "pass":
            return DeskApprovalResult(
                backtest=backtest,
                pla=pla,
                status="fail",
                reason="pla fail",
            )
        if backtest.traffic_light == "yellow":
            return DeskApprovalResult(
                backtest=backtest,
                pla=pla,
                status="conditional",
                reason="backtest yellow",
            )
        return DeskApprovalResult(
            backtest=backtest,
            pla=pla,
            status="pass",
            reason="approved",
        )
