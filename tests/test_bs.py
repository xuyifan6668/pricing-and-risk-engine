from datetime import timedelta

import pytest

from market_scenarios import market_scenarios
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.products.base import Underlying
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.utils.types import OptionType


SCENARIOS = market_scenarios()


@pytest.mark.parametrize("scenario", list(SCENARIOS.keys()))
def test_bs_price_positive(scenario):
    market = SCENARIOS[scenario]
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

    result = engine.price(product, market, model, EngineSettings())
    assert result["price"] > 0.0
