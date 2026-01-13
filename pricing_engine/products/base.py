"""Product definitions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import date
from typing import Dict, Iterable, Optional


@dataclass(frozen=True)
class Underlying:
    symbol: str
    exchange: Optional[str] = None


class Product(ABC):
    """Abstract product definition."""

    @property
    @abstractmethod
    def product_type(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def payoff(self, path_or_spot) -> float:
        """Payoff given spot or path representation."""
        raise NotImplementedError

    def is_path_dependent(self) -> bool:
        return False

    def required_market_data(self) -> Iterable[str]:
        return []

    def metadata(self) -> Dict[str, str]:
        return {}
