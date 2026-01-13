"""Exotic equity products."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable, Optional, Sequence

from pricing_engine.products.base import Product, Underlying
from pricing_engine.utils.types import AveragingType, BarrierType, OptionType


@dataclass(frozen=True)
class BarrierOption(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    barrier: float
    barrier_type: BarrierType
    option_type: OptionType
    rebate: float = 0.0

    @property
    def product_type(self) -> str:
        return "barrier_option"

    def payoff(self, path_or_spot) -> float:
        # Placeholder payoff; barrier logic handled in engine.
        spot = float(path_or_spot)
        if self.option_type == OptionType.CALL:
            return max(spot - self.strike, 0.0)
        return max(self.strike - spot, 0.0)

    def is_path_dependent(self) -> bool:
        return True


@dataclass(frozen=True)
class DigitalOption(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    option_type: OptionType
    payout: float = 1.0

    @property
    def product_type(self) -> str:
        return "digital_option"

    def payoff(self, path_or_spot) -> float:
        spot = float(path_or_spot)
        if self.option_type == OptionType.CALL:
            return self.payout if spot > self.strike else 0.0
        return self.payout if spot < self.strike else 0.0


@dataclass(frozen=True)
class AsianOption(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    option_type: OptionType
    averaging: AveragingType
    observation_dates: Sequence[date]

    @property
    def product_type(self) -> str:
        return "asian_option"

    def payoff(self, path_or_spot) -> float:
        path = list(path_or_spot) if isinstance(path_or_spot, Iterable) else [float(path_or_spot)]
        avg = sum(path) / len(path)
        spot = path[-1]
        if self.averaging == AveragingType.PRICE:
            if self.option_type == OptionType.CALL:
                return max(avg - self.strike, 0.0)
            return max(self.strike - avg, 0.0)
        if self.option_type == OptionType.CALL:
            return max(spot - avg, 0.0)
        return max(avg - spot, 0.0)

    def is_path_dependent(self) -> bool:
        return True


@dataclass(frozen=True)
class LookbackOption(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    option_type: OptionType

    @property
    def product_type(self) -> str:
        return "lookback_option"

    def payoff(self, path_or_spot) -> float:
        path = list(path_or_spot) if isinstance(path_or_spot, Iterable) else [float(path_or_spot)]
        if self.option_type == OptionType.CALL:
            return max(max(path) - self.strike, 0.0)
        return max(self.strike - min(path), 0.0)

    def is_path_dependent(self) -> bool:
        return True


@dataclass(frozen=True)
class Cliquet(Product):
    underlying: Underlying
    currency: str
    maturity: date
    local_cap: float
    local_floor: float
    global_cap: Optional[float] = None
    observation_dates: Sequence[date] = ()

    @property
    def product_type(self) -> str:
        return "cliquet"

    def payoff(self, path_or_spot) -> float:
        path = list(path_or_spot) if isinstance(path_or_spot, Iterable) else [float(path_or_spot)]
        if len(path) < 2:
            return 0.0
        returns = []
        for i in range(1, len(path)):
            raw = path[i] / path[i - 1] - 1.0
            capped = min(max(raw, self.local_floor), self.local_cap)
            returns.append(capped)
        total = sum(returns)
        if self.global_cap is not None:
            total = min(total, self.global_cap)
        return total

    def is_path_dependent(self) -> bool:
        return True


@dataclass(frozen=True)
class Corridor(Product):
    underlying: Underlying
    currency: str
    maturity: date
    lower: float
    upper: float
    notional: float = 1.0

    @property
    def product_type(self) -> str:
        return "corridor"

    def payoff(self, path_or_spot) -> float:
        path = list(path_or_spot) if isinstance(path_or_spot, Iterable) else [float(path_or_spot)]
        if not path:
            return 0.0
        inside = sum(1 for s in path if self.lower <= s <= self.upper)
        return self.notional * inside / len(path)

    def is_path_dependent(self) -> bool:
        return True


@dataclass(frozen=True)
class VarianceSwap(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    notional: float

    @property
    def product_type(self) -> str:
        return "variance_swap"

    def payoff(self, path_or_spot) -> float:
        return 0.0

    def is_path_dependent(self) -> bool:
        return True


@dataclass(frozen=True)
class BasketOption(Product):
    """Basket option proxy (single-spot approximation)."""

    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    option_type: OptionType
    weights: Sequence[float] = ()
    components: Sequence[str] = ()

    @property
    def product_type(self) -> str:
        return "basket_option"

    def payoff(self, path_or_spot) -> float:
        spot = path_or_spot
        if isinstance(path_or_spot, Iterable) and not isinstance(path_or_spot, (str, bytes)):
            seq = list(path_or_spot)
            if self.weights and len(seq) == len(self.weights):
                spot = sum(w * float(s) for w, s in zip(self.weights, seq))
            elif seq:
                spot = float(seq[-1])
            else:
                spot = 0.0
        else:
            spot = float(path_or_spot)
        if self.option_type == OptionType.CALL:
            return max(float(spot) - self.strike, 0.0)
        return max(self.strike - float(spot), 0.0)

@dataclass(frozen=True)
class VolSwap(Product):
    underlying: Underlying
    currency: str
    maturity: date
    strike: float
    notional: float

    @property
    def product_type(self) -> str:
        return "vol_swap"

    def payoff(self, path_or_spot) -> float:
        return 0.0

    def is_path_dependent(self) -> bool:
        return True
