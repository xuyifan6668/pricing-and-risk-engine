"""Product registry exports."""

from pricing_engine.products.base import Product, Underlying
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

__all__ = [
    "Product",
    "Underlying",
    "Spot",
    "Forward",
    "EquitySwap",
    "EuropeanOption",
    "AmericanOption",
    "BarrierOption",
    "BasketOption",
    "DigitalOption",
    "AsianOption",
    "LookbackOption",
    "Cliquet",
    "Corridor",
    "VarianceSwap",
    "VolSwap",
]
