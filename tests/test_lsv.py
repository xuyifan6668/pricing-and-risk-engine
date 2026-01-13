from datetime import timedelta

from market_scenarios import market_scenarios
from pricing_engine.calibration.lsv import (
    LSVLeverageSettings,
    LSVParticleSettings,
    build_leverage_surface,
    calibrate_leverage_particle,
)
from pricing_engine.models.local_stochastic_vol import LocalStochasticVolModel
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.numerics.monte_carlo import MonteCarloEngine
from pricing_engine.products.base import Underlying
from pricing_engine.products.exotics import BarrierOption
from pricing_engine.utils.types import BarrierType, OptionType


def test_lsv_barrier_price_positive():
    market = market_scenarios()["normal"]
    maturity = market.asof + timedelta(days=365)
    product = BarrierOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=100.0,
        barrier=115.0,
        barrier_type=BarrierType.UP_OUT,
        option_type=OptionType.CALL,
    )

    model = LocalStochasticVolModel(
        kappa=1.5,
        theta=0.04,
        v0=0.04,
        sigma=0.5,
        rho=-0.5,
    )
    settings = LSVLeverageSettings(
        times=[0.25, 0.5, 1.0, 2.0],
        spots=[0.7 * market.spot, market.spot, 1.3 * market.spot],
    )
    model.leverage = build_leverage_surface(market, model, settings)

    engine = MonteCarloEngine(num_paths=3000, timesteps=80)
    result = engine.price(product, market, model, EngineSettings(seed=123))
    assert result["price"] >= 0.0


def test_lsv_particle_calibration_bounds():
    market = market_scenarios()["normal"]
    model = LocalStochasticVolModel(
        kappa=1.5,
        theta=0.04,
        v0=0.04,
        sigma=0.5,
        rho=-0.5,
    )
    settings = LSVParticleSettings(
        times=[0.5, 1.0],
        spots=[0.8 * market.spot, market.spot, 1.2 * market.spot],
        num_paths=2000,
        substeps=2,
        iterations=2,
        bandwidth=0.25,
        seed=7,
        min_leverage=0.2,
        max_leverage=4.0,
    )
    surface = calibrate_leverage_particle(market, model, settings)
    for row in surface.values:
        for val in row:
            assert settings.min_leverage <= val <= settings.max_leverage
