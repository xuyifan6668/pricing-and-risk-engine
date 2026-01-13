"""Market risk engine for FRTB IMA (historical VaR/ES)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot

from market_risk.backtesting import Backtester, BacktestResult
from market_risk.es import ExpectedShortfallCalculator, ESResult
from market_risk.frtb import IMACapitalCalculator, IMACapitalResult
from market_risk.pla import PLATester, PLATestResult
from market_risk.portfolio import Portfolio
from market_risk.scenarios import HistoricalScenarioBuilder, ScenarioSet
from market_risk.var import HistoricalVaRCalculator, VaRResult


@dataclass(frozen=True)
class RiskReport:
    var_result: VaRResult
    es_result: ESResult
    ima_capital: IMACapitalResult


class MarketRiskEngine:
    def __init__(
        self,
        scenario_builder: HistoricalScenarioBuilder,
        var_level: float = 0.99,
        es_level: float = 0.975,
    ) -> None:
        self.scenario_builder = scenario_builder
        self.var_calc = HistoricalVaRCalculator()
        self.es_calc = ExpectedShortfallCalculator()
        self.ima_calc = IMACapitalCalculator(es_level=es_level)
        self.var_level = var_level
        self.es_level = es_level

    def build_scenarios(self, history: Sequence[MarketDataSnapshot]) -> ScenarioSet:
        return self.scenario_builder.build(history)

    def compute_risk(
        self,
        portfolio: Portfolio,
        base_market: MarketDataSnapshot,
        scenarios: ScenarioSet,
    ) -> RiskReport:
        scenario_markets = scenarios.markets(base_market)
        pnls = portfolio.scenario_pnls(base_market, scenario_markets)
        var = self.var_calc.compute(pnls, self.var_level, scenarios.horizon_days)
        es = self.es_calc.compute(pnls, self.es_level, scenarios.horizon_days)
        ima = self.ima_calc.compute(portfolio, base_market, scenarios)
        return RiskReport(var_result=var, es_result=es, ima_capital=ima)

    def backtest(self, actual_pnls: Sequence[float], var_series: Sequence[float], window: int = 250) -> BacktestResult:
        return Backtester().run(actual_pnls, var_series, window=window)

    def pla(self, hpl: Sequence[float], rtpl: Sequence[float]) -> PLATestResult:
        return PLATester().run(hpl, rtpl)
