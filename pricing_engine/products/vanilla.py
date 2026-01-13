"""Vanilla equity products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Optional

from pricing_engine.products.base import Product, Underlying
from pricing_engine.utils.types import OptionType


@dataclass(frozen=True)
class Spot(Product):
    underlying: Underlying
    currency: str
    settlement_date: date

    @property
    def product_type(self) -> str:
        return "spot"

    @property
    def maturity(self) -> date:
        return self.settlement_date

    def payoff(self, path_or_spot) -> float:
        return float(path_or_spot)


@dataclass(frozen=True)
class Forward(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float

    @property
    def product_type(self) -> str:
        return "forward"

    def payoff(self, path_or_spot) -> float:
        return float(path_or_spot) - self.strike


@dataclass(frozen=True)
class EquitySwap(Product):
    underlying: Underlying
    currency: str
    maturity: date
    notional: float
    fixed_rate: float

    @property
    def product_type(self) -> str:
        return "swap"

    def payoff(self, path_or_spot) -> float:
        # Stub: equity leg minus fixed leg.
        return 0.0


@dataclass(frozen=True)
class EuropeanOption(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    option_type: OptionType

    @property
    def product_type(self) -> str:
        return "european_option"

    def payoff(self, path_or_spot) -> float:
        spot = float(path_or_spot)
        if self.option_type == OptionType.CALL:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)


@dataclass(frozen=True)
class AmericanOption(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    option_type: OptionType
    exercise_start: Optional[date] = None

    @property
    def product_type(self) -> str:
        return "american_option"

    def payoff(self, path_or_spot) -> float:
        spot = float(path_or_spot)
        if self.option_type == OptionType.CALL:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)
