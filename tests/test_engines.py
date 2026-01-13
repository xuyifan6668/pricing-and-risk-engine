from datetime import timedelta

import pytest

from market_scenarios import market_scenarios
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.numerics.monte_carlo import MonteCarloEngine
from pricing_engine.numerics.pde import PDEEngine
from pricing_engine.numerics.trees import TreeEngine
from pricing_engine.products.base import Underlying
from pricing_engine.products.exotics import AsianOption, BarrierOption, DigitalOption, VarianceSwap
from pricing_engine.products.vanilla import AmericanOption, EuropeanOption
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


@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_european_call_parity_across_engines(scenario):
    market = _market(scenario)
    maturity = market.asof + timedelta(days=365)
    sigma_atm = market.vol_surface.implied_vol(1.0, market.spot)
    product = EuropeanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=market.spot,
        option_type=OptionType.CALL,
    )
    model = BlackScholesModel(sigma=sigma_atm)
    analytic = AnalyticEngine().price(product, market, model, EngineSettings())["price"]
    tree = TreeEngine(steps=400).price(product, market, model, EngineSettings())["price"]
    pde = PDEEngine(grid_points=200, time_steps=200).price(product, market, model, EngineSettings())["price"]

    assert tree == pytest.approx(analytic, rel=1e-3, abs=1e-2)
    assert pde == pytest.approx(analytic, rel=1e-3, abs=1e-2)


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


def test_mc_barrier_deterministic_seed():
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
    engine = MonteCarloEngine(num_paths=2000, timesteps=50, use_brownian_bridge=True)
    settings = EngineSettings(seed=123, deterministic=True)

    price1 = engine.price(product, market, model, settings)["price"]
    price2 = engine.price(product, market, model, settings)["price"]
    assert price1 == pytest.approx(price2)


def test_pde_barrier_matches_mc():
    market = _market("normal")
    maturity = market.asof + timedelta(days=365)
    product = BarrierOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=market.spot,
        barrier=market.spot * 1.1,
        barrier_type=BarrierType.UP_OUT,
        option_type=OptionType.CALL,
    )
    model = BlackScholesModel(sigma=0.2)
    pde = PDEEngine(grid_points=240, time_steps=240)
    mc = MonteCarloEngine(
        num_paths=20000,
        timesteps=80,
        use_brownian_bridge=True,
        use_antithetic=True,
        use_control_variate=True,
    )

    pde_price = pde.price(product, market, model, EngineSettings())["price"]
    mc_price = mc.price(product, market, model, EngineSettings(seed=12))["price"]
    assert pde_price == pytest.approx(mc_price, rel=0.02, abs=0.07)


def test_digital_parity_mc_vs_analytic():
    market = _market("normal")
    maturity = market.asof + timedelta(days=180)
    product = DigitalOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=market.spot,
        option_type=OptionType.CALL,
        payout=10.0,
    )
    model = BlackScholesModel(sigma=0.2)
    analytic = AnalyticEngine().price(product, market, model, EngineSettings())["price"]
    mc = MonteCarloEngine(
        num_paths=30000,
        timesteps=80,
        use_antithetic=True,
        use_control_variate=True,
    )

    mc_price = mc.price(product, market, model, EngineSettings(seed=21))["price"]
    assert mc_price == pytest.approx(analytic, rel=0.02, abs=0.05)


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
