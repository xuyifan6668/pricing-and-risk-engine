"""Mock portfolio and historical market data for FRTB IMA workflow."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
import math
import random
from typing import List, Sequence

from pricing_engine.api import PricingSettings
from pricing_engine.market_data import Dividend, DividendSchedule, MarketDataSnapshot, SmileVolSurface, ZeroCurve
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.models.heston import HestonModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.numerics.monte_carlo import MonteCarloEngine
from pricing_engine.numerics.trees import TreeEngine
from pricing_engine.products.base import Underlying
from pricing_engine.products.exotics import (
    AsianOption,
    BarrierOption,
    BasketOption,
    Cliquet,
    Corridor,
    DigitalOption,
    LookbackOption,
    VarianceSwap,
    VolSwap,
)
from pricing_engine.products.vanilla import AmericanOption, EquitySwap, EuropeanOption, Forward, Spot
from pricing_engine.utils.types import AveragingType, BarrierType, OptionType

from market_risk.portfolio import Portfolio, Position


@dataclass(frozen=True)
class MockHistorySettings:
    start: date
    days: int = 260
    seed: int = 7
    spot: float = 102.0
    drift: float = 0.02
    spot_vol: float = 0.18


@dataclass(frozen=True)
class MockPortfolioSettings:
    quantity_scale: float = 1.0
    mc_paths: int = 800
    mc_steps: int = 60


def _mean_revert(x: float, theta: float, kappa: float, sigma: float, dt: float, rng: random.Random) -> float:
    return x + kappa * (theta - x) * dt + sigma * math.sqrt(dt) * rng.gauss(0.0, 1.0)


def generate_market_history(settings: MockHistorySettings) -> List[MarketDataSnapshot]:
    rng = random.Random(settings.seed)
    dt = 1.0 / 252.0
    tenors = [0.25, 0.5, 1.0, 2.0, 5.0]

    spot = settings.spot
    short_rate = 0.02
    funding_spread = 0.002
    borrow_spread = 0.004

    atm = 0.22
    skew = -0.12
    curvature = 0.16

    history: List[MarketDataSnapshot] = []
    for day in range(settings.days):
        log_ret = (settings.drift - 0.5 * settings.spot_vol * settings.spot_vol) * dt + settings.spot_vol * math.sqrt(dt) * rng.gauss(0.0, 1.0)
        spot *= math.exp(log_ret)

        short_rate = _mean_revert(short_rate, 0.02, 1.2, 0.005, dt, rng)
        funding_spread = _mean_revert(funding_spread, 0.002, 1.0, 0.002, dt, rng)
        borrow_spread = _mean_revert(borrow_spread, 0.004, 0.8, 0.002, dt, rng)

        atm = max(_mean_revert(atm, 0.21, 1.0, 0.03, dt, rng), 0.05)
        skew = _mean_revert(skew, -0.12, 1.0, 0.02, dt, rng)
        curvature = max(_mean_revert(curvature, 0.16, 1.0, 0.02, dt, rng), 0.05)

        discount_curve = ZeroCurve(tenors, [short_rate + 0.001 * t for t in tenors])
        funding_curve = ZeroCurve(tenors, [short_rate + funding_spread + 0.0012 * t for t in tenors])
        borrow_curve = ZeroCurve(tenors, [borrow_spread + 0.0005 * t for t in tenors])

        vol_surface = SmileVolSurface(
            expiries=tenors,
            atm_vols=[max(atm - 0.015 * math.log(1 + t), 0.05) for t in tenors],
            skew=[skew + 0.01 * t for t in tenors],
            curvature=[curvature + 0.01 * t for t in tenors],
            spot_ref=spot,
        )

        dividends = DividendSchedule(
            discrete=[
                Dividend(settings.start + timedelta(days=70), 0.6),
                Dividend(settings.start + timedelta(days=160), 0.7),
                Dividend(settings.start + timedelta(days=240), 1.4),
            ],
            continuous_yield=0.005,
        )

        history.append(
            MarketDataSnapshot(
                asof=settings.start + timedelta(days=day),
                spot=spot,
                discount_curve=discount_curve,
                funding_curve=funding_curve,
                borrow_curve=borrow_curve,
                vol_surface=vol_surface,
                dividends=dividends,
            )
        )

    return history


def build_mock_portfolio(base_market: MarketDataSnapshot, settings: MockPortfolioSettings) -> Portfolio:
    asof = base_market.asof
    underlying = Underlying("EQ_US")
    basket_underlying = Underlying("BASKET_TECH")

    analytic = AnalyticEngine()
    tree = TreeEngine(steps=200)
    mc = MonteCarloEngine(num_paths=settings.mc_paths, timesteps=settings.mc_steps)

    pricing = PricingSettings(engine_settings=EngineSettings(seed=42))

    sigma_atm = base_market.vol_surface.implied_vol(1.0, base_market.spot)

    portfolio = Portfolio(
        positions=[
            Position(
                product=Spot(underlying, "USD", asof),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=5_000_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Cash Equity",
            ),
            Position(
                product=Forward(underlying, "USD", asof + timedelta(days=90), base_market.spot * 1.01),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=-2_000_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Forward Hedge",
            ),
            Position(
                product=EuropeanOption(
                    underlying, "USD", asof + timedelta(days=365), base_market.spot, OptionType.CALL
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=15_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="ATM Call",
            ),
            Position(
                product=EuropeanOption(
                    underlying, "USD", asof + timedelta(days=365), base_market.spot * 0.9, OptionType.PUT
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=12_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="OTM Put",
            ),
            Position(
                product=AmericanOption(
                    underlying, "USD", asof + timedelta(days=365), base_market.spot * 0.95, OptionType.PUT
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=tree,
                quantity=8_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="American Put",
            ),
            Position(
                product=BarrierOption(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    base_market.spot,
                    barrier=base_market.spot * 1.2,
                    barrier_type=BarrierType.UP_OUT,
                    option_type=OptionType.CALL,
                ),
                model=HestonModel(kappa=1.8, theta=0.04, v0=0.04, sigma=0.6, rho=-0.5),
                engine=mc,
                quantity=6_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Up-and-Out",
            ),
            Position(
                product=DigitalOption(
                    underlying,
                    "USD",
                    asof + timedelta(days=180),
                    base_market.spot * 1.05,
                    OptionType.CALL,
                    payout=10.0,
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=20_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Digital Call",
            ),
            Position(
                product=AsianOption(
                    underlying,
                    "USD",
                    asof + timedelta(days=240),
                    base_market.spot,
                    OptionType.CALL,
                    AveragingType.PRICE,
                    observation_dates=(),
                ),
                model=HestonModel(kappa=1.5, theta=0.05, v0=0.05, sigma=0.7, rho=-0.4),
                engine=mc,
                quantity=4_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Asian Call",
            ),
            Position(
                product=LookbackOption(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    base_market.spot * 0.95,
                    OptionType.PUT,
                ),
                model=HestonModel(kappa=1.4, theta=0.04, v0=0.04, sigma=0.6, rho=-0.3),
                engine=mc,
                quantity=2_500 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Lookback Put",
            ),
            Position(
                product=Cliquet(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    local_cap=0.05,
                    local_floor=-0.03,
                    global_cap=0.12,
                    observation_dates=(),
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=mc,
                quantity=3_000 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Cliquet",
            ),
            Position(
                product=Corridor(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    lower=base_market.spot * 0.9,
                    upper=base_market.spot * 1.1,
                    notional=100_000,
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=mc,
                quantity=1.0 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Corridor",
            ),
            Position(
                product=VarianceSwap(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    strike=sigma_atm * sigma_atm,
                    notional=1_000_000,
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=1.0 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Variance Swap",
            ),
            Position(
                product=VolSwap(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    strike=sigma_atm,
                    notional=500_000,
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=-1.0 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Vol Swap",
            ),
            Position(
                product=EquitySwap(
                    underlying,
                    "USD",
                    asof + timedelta(days=365),
                    notional=2_000_000,
                    fixed_rate=0.01,
                ),
                model=BlackScholesModel(sigma=sigma_atm),
                engine=analytic,
                quantity=1.0 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Equity Swap",
            ),
            Position(
                product=BasketOption(
                    basket_underlying,
                    "USD",
                    asof + timedelta(days=365),
                    strike=base_market.spot * 1.02,
                    option_type=OptionType.CALL,
                    weights=(0.5, 0.3, 0.2),
                    components=("TECH_A", "TECH_B", "TECH_C"),
                ),
                model=BlackScholesModel(sigma=sigma_atm * 1.1),
                engine=analytic,
                quantity=7_500 * settings.quantity_scale,
                pricing_settings=pricing,
                name="Basket Call (Proxy)",
            ),
        ]
    )

    return portfolio
