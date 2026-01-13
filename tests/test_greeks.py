from datetime import timedelta

from market_scenarios import market_scenarios
from pricing_engine.api import PricingSettings, price
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.products.base import Underlying
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.risk.greeks import GreekRequest
from pricing_engine.utils.types import OptionType


def _assert_stable(a: float, b: float, rel_tol: float = 0.05, abs_tol: float = 1e-6) -> None:
    diff = abs(a - b)
    scale = max(abs(a), abs(b), abs_tol)
    assert diff / scale <= rel_tol


def test_bump_greeks_stability_halved_bumps():
    market = market_scenarios()["normal"]
    maturity = market.asof + timedelta(days=365)
    product = EuropeanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=market.spot,
        option_type=OptionType.CALL,
    )
    model = BlackScholesModel(sigma=0.2)
    engine = AnalyticEngine()

    request = GreekRequest(
        greeks=("delta", "gamma", "vega"),
        method="bump",
        bump_size=1e-3,
        vol_bump=1e-3,
    )
    request_half = GreekRequest(
        greeks=("delta", "gamma", "vega"),
        method="bump",
        bump_size=5e-4,
        vol_bump=5e-4,
    )

    settings = PricingSettings(
        engine_settings=EngineSettings(deterministic=True),
        compute_greeks=True,
        greek_request=request,
    )
    settings_half = PricingSettings(
        engine_settings=EngineSettings(deterministic=True),
        compute_greeks=True,
        greek_request=request_half,
    )

    greeks = price(product, market, model, engine, settings).greeks
    greeks_half = price(product, market, model, engine, settings_half).greeks

    for key in ("delta", "gamma", "vega"):
        _assert_stable(greeks[key], greeks_half[key])
