from datetime import date

from market_scenarios import market_scenarios
from pricing_engine.market_data import MarketDataSnapshot, SmileVolSurface
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.api import PricingSettings
from pricing_engine.products.base import Underlying
from pricing_engine.products.vanilla import EuropeanOption
from pricing_engine.utils.types import OptionType

from market_risk.backtesting import Backtester
from market_risk.engine import MarketRiskEngine
from market_risk.es import ExpectedShortfallCalculator
from market_risk.pla import PLATester
from market_risk.portfolio import Portfolio, Position
from market_risk.scenarios import HistoricalScenarioBuilder, default_tenor_grid
from market_risk.var import HistoricalVaRCalculator


def _history():
    base = market_scenarios()["normal"]
    v = base.vol_surface
    if not isinstance(v, SmileVolSurface):
        return [base, base]

    def bump_surface(surface, bump):
        return SmileVolSurface(
            expiries=surface.expiries,
            atm_vols=[max(x + bump, 1e-4) for x in surface.atm_vols],
            skew=surface.skew,
            curvature=surface.curvature,
            spot_ref=surface.spot_ref,
        )

    snap1 = MarketDataSnapshot(
        asof=base.asof,
        spot=base.spot,
        discount_curve=base.discount_curve,
        funding_curve=base.funding_curve,
        borrow_curve=base.borrow_curve,
        vol_surface=base.vol_surface,
        dividends=base.dividends,
        corporate_actions=base.corporate_actions,
    )
    snap2 = MarketDataSnapshot(
        asof=date(2024, 1, 3),
        spot=base.spot * 0.98,
        discount_curve=base.discount_curve,
        funding_curve=base.funding_curve,
        borrow_curve=base.borrow_curve,
        vol_surface=bump_surface(v, 0.01),
        dividends=base.dividends,
        corporate_actions=base.corporate_actions,
    )
    snap3 = MarketDataSnapshot(
        asof=date(2024, 1, 4),
        spot=base.spot * 1.02,
        discount_curve=base.discount_curve,
        funding_curve=base.funding_curve,
        borrow_curve=base.borrow_curve,
        vol_surface=bump_surface(v, -0.005),
        dividends=base.dividends,
        corporate_actions=base.corporate_actions,
    )
    return [snap1, snap2, snap3]


def test_var_es_basic():
    pnls = [-1.0, 0.5, -2.5, 1.2, -0.3]
    var = HistoricalVaRCalculator().compute(pnls, 0.99, 1)
    es = ExpectedShortfallCalculator().compute(pnls, 0.975, 1)
    assert var.value >= 0.0
    assert es.value >= 0.0


def test_backtester_traffic_light():
    actual = [1, -2, 0.5, -3, -1, 0.2]
    var_series = [1, 1, 1, 1, 1, 1]
    result = Backtester(green=1, yellow=2).run(actual, var_series, window=6)
    assert result.traffic_light in {"green", "yellow", "red"}


def test_pla_pass_fail():
    hpl = [1, 2, 3, 4, 5]
    rtpl = [1.1, 1.9, 3.1, 3.9, 4.8]
    result = PLATester().run(hpl, rtpl)
    assert result.status in {"pass", "fail"}


def test_risk_engine_outputs():
    history = _history()
    base_market = history[0]

    product = EuropeanOption(
        underlying=Underlying("XYZ"),
        currency="USD",
        maturity=date(2025, 1, 2),
        strike=100.0,
        option_type=OptionType.CALL,
    )
    position = Position(
        product=product,
        model=BlackScholesModel(sigma=0.2),
        engine=AnalyticEngine(),
        quantity=10.0,
        pricing_settings=PricingSettings(engine_settings=EngineSettings()),
    )
    portfolio = Portfolio([position])

    builder = HistoricalScenarioBuilder(tenor_grid=default_tenor_grid())
    engine = MarketRiskEngine(builder)
    scenarios = engine.build_scenarios(history)
    report = engine.compute_risk(portfolio, base_market, scenarios)

    assert report.var_result.value >= 0.0
    assert report.es_result.value >= 0.0
    assert report.ima_capital.total_es >= 0.0
