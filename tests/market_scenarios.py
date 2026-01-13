from datetime import date

from pricing_engine.market_data import Dividend, DividendSchedule, MarketDataSnapshot, SmileVolSurface, ZeroCurve
from pricing_engine.market_data.corporate_actions import SpecialDividend


def market_scenarios():
    asof = date(2024, 1, 2)
    times = [0.25, 0.5, 1.0, 2.0, 5.0]
    div_dates = [date(2024, 3, 15), date(2024, 6, 14), date(2024, 9, 20)]

    scenarios = {
        "normal": {
            "spot": 102.0,
            "discount": [0.015, 0.017, 0.02, 0.024, 0.028],
            "funding": [0.017, 0.019, 0.022, 0.026, 0.03],
            "borrow": [0.003, 0.004, 0.005, 0.006, 0.007],
            "atm_vols": [0.22, 0.21, 0.2, 0.195, 0.19],
            "skew": [-0.15, -0.12, -0.1, -0.08, -0.06],
            "curvature": [0.2, 0.18, 0.16, 0.14, 0.12],
            "div_cont": 0.005,
            "div_disc": [0.6, 0.65, 2.5],
        },
        "steep": {
            "spot": 98.0,
            "discount": [0.01, 0.015, 0.025, 0.035, 0.045],
            "funding": [0.012, 0.017, 0.027, 0.037, 0.047],
            "borrow": [0.006, 0.007, 0.008, 0.009, 0.01],
            "atm_vols": [0.2, 0.195, 0.19, 0.185, 0.18],
            "skew": [-0.12, -0.1, -0.09, -0.07, -0.05],
            "curvature": [0.18, 0.16, 0.14, 0.12, 0.1],
            "div_cont": 0.006,
            "div_disc": [0.5, 0.55, 1.8],
        },
        "volatile": {
            "spot": 105.0,
            "discount": [0.012, 0.015, 0.018, 0.022, 0.027],
            "funding": [0.014, 0.017, 0.02, 0.024, 0.029],
            "borrow": [0.004, 0.006, 0.007, 0.009, 0.011],
            "atm_vols": [0.35, 0.32, 0.3, 0.28, 0.26],
            "skew": [-0.25, -0.22, -0.2, -0.17, -0.15],
            "curvature": [0.3, 0.28, 0.25, 0.22, 0.2],
            "div_cont": 0.004,
            "div_disc": [0.7, 0.75, 3.0],
        },
    }

    snapshots = {}
    for name, data in scenarios.items():
        dividends = DividendSchedule(
            discrete=[
                Dividend(div_dates[0], data["div_disc"][0]),
                Dividend(div_dates[1], data["div_disc"][1]),
                Dividend(div_dates[2], data["div_disc"][2]),
            ],
            continuous_yield=data["div_cont"],
        )
        corporate_actions = [SpecialDividend(div_dates[2], data["div_disc"][2])]
        snapshots[name] = MarketDataSnapshot(
            asof=asof,
            spot=data["spot"],
            discount_curve=ZeroCurve(times, data["discount"]),
            funding_curve=ZeroCurve(times, data["funding"]),
            borrow_curve=ZeroCurve(times, data["borrow"]),
            vol_surface=SmileVolSurface(
                expiries=times,
                atm_vols=data["atm_vols"],
                skew=data["skew"],
                curvature=data["curvature"],
                spot_ref=data["spot"],
            ),
            dividends=dividends,
            corporate_actions=corporate_actions,
        )
    return snapshots
