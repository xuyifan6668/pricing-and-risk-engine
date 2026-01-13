from datetime import timedelta

from market_scenarios import market_scenarios
from pricing_engine.api import PricingSettings
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.products.base import Underlying
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.risk.greeks import GreekRequest
from pricing_engine.services.cache import SimpleCache
from pricing_engine.services.pricing_service import PricingService
from pricing_engine.utils.types import OptionType


SCENARIOS = market_scenarios()


def _market(name: str = "normal"):
    return SCENARIOS[name]


def test_pricing_service_cache_key_changes_with_product_and_model():
    market = _market()
    maturity = market.asof + timedelta(days=365)
    product_one = EuropeanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=100.0,
        option_type=OptionType.CALL,
    )
    product_two = EuropeanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=105.0,
        option_type=OptionType.CALL,
    )
    model_one = BlackScholesModel(sigma=0.2)
    model_two = BlackScholesModel(sigma=0.25)
    engine = AnalyticEngine()
    settings = PricingSettings(engine_settings=EngineSettings())
    service = PricingService(cache=SimpleCache())

    key_one = service._cache_key(product_one, market, model_one, engine, settings)
    key_two = service._cache_key(product_two, market, model_one, engine, settings)
    key_three = service._cache_key(product_one, market, model_two, engine, settings)

    assert key_one != key_two
    assert key_one != key_three


def test_pricing_service_cache_key_changes_with_greek_request():
    market = _market()
    maturity = market.asof + timedelta(days=365)
    product = EuropeanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=maturity,
        strike=100.0,
        option_type=OptionType.CALL,
    )
    model = BlackScholesModel(sigma=0.2)
    engine = AnalyticEngine()
    service = PricingService(cache=SimpleCache())

    settings_one = PricingSettings(
        engine_settings=EngineSettings(),
        compute_greeks=True,
        greek_request=GreekRequest(greeks=("delta", "vega")),
    )
    settings_two = PricingSettings(
        engine_settings=EngineSettings(),
        compute_greeks=True,
        greek_request=GreekRequest(greeks=("delta", "gamma")),
    )

    key_one = service._cache_key(product, market, model, engine, settings_one)
    key_two = service._cache_key(product, market, model, engine, settings_two)

    assert key_one != key_two
