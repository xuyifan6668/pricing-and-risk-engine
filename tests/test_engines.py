from datetime import timedelta

import pytest

from market_scenarios import market_scenarios
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.numerics.monte_carlo import MonteCarloEngine
from pricing_engine.numerics.trees import TreeEngine
from pricing_engine.products.base import Underlying
from pricing_engine.products.exotics import AsianOption, BarrierOption, VarianceSwap
from pricing_engine.products.vanilla import AmericanOption
from pricing_engine.utils.types import AveragingType, BarrierType, OptionType


SCENARIOS = market_scenarios()


def _market(name: str = "normal"):
    return SCENARIOS[name]


@pytest.mark.parametrize("scenario", ["normal", "steep"])
def test_tree_american_call_positive(scenario):
    market = _market(scenario)
    maturity = market.asof + timedelta(days=365)
    product = AmericanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=100.0,
        option_type=OptionType.CALL,
    )
    model = BlackScholesModel(sigma=0.2)
    engine = TreeEngine(steps=200)

    result = engine.price(product, market, model, EngineSettings())
    assert result["price"] > 0.0


def test_mc_asian_option_positive():
    market = _market("normal")
    maturity = market.asof + timedelta(days=180)
    product = AsianOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=100.0,
        option_type=OptionType.CALL,
        averaging=AveragingType.PRICE,
        observation_dates=(),
    )
    model = BlackScholesModel(sigma=0.2)
    engine = MonteCarloEngine(num_paths=2000, timesteps=50)

    result = engine.price(product, market, model, EngineSettings(seed=42))
    assert result["price"] >= 0.0


def test_mc_lsm_american_put_positive():
    market = _market("normal")
    maturity = market.asof + timedelta(days=365)
    product = AmericanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=105.0,
        option_type=OptionType.PUT,
    )
    model = BlackScholesModel(sigma=0.2)
    engine = MonteCarloEngine(num_paths=5000, timesteps=50)

    result = engine.price(product, market, model, EngineSettings(seed=7))
    assert result["price"] >= 0.0


def test_mc_barrier_continuous_below_discrete():
    market = _market("volatile")
    maturity = market.asof + timedelta(days=180)
    product = BarrierOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=100.0,
        barrier=105.0,
        barrier_type=BarrierType.UP_OUT,
        option_type=OptionType.CALL,
    )
    model = BlackScholesModel(sigma=0.2)
    engine_discrete = MonteCarloEngine(num_paths=4000, timesteps=50, use_brownian_bridge=False)
    engine_cont = MonteCarloEngine(num_paths=4000, timesteps=50, use_brownian_bridge=True)

    price_discrete = engine_discrete.price(product, market, model, EngineSettings(seed=11))["price"]
    price_cont = engine_cont.price(product, market, model, EngineSettings(seed=11))["price"]
    assert price_cont <= price_discrete + 1e-6


@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_analytic_variance_swap_fair_strike(scenario):
    market = _market(scenario)
    maturity = market.asof + timedelta(days=365)
    product = VarianceSwap(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=0.04,
        notional=1.0,
    )
    model = BlackScholesModel(sigma=0.2)
    engine = AnalyticEngine()

    result = engine.price(product, market, model, EngineSettings())
    assert 0.01 <= result["fair_var"] <= 0.2


@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_market_scenarios_forward_positive(scenario):
    market = _market(scenario)
    maturity = market.asof + timedelta(days=365)
    assert market.forward_price(maturity) > 0.0
