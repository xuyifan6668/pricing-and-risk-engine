"""Numerical engine interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from pricing_engine.models.base import Model
from pricing_engine.products.base import Product
from pricing_engine.market_data.snapshot import MarketDataSnapshot


@dataclass(frozen=True)
class EngineSettings:
    deterministic: bool = True
    tolerance: float = 1e-8
    max_steps: int = 2000
    seed: Optional[int] = None


class Engine(ABC):
    """Pricing engine interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def price(
        self,
        product: Product,
        market: MarketDataSnapshot,
        model: Model,
        settings: EngineSettings,
    ) -> Dict[str, float]:
        """Return pricing outputs including price and optional diagnostics."""
        raise NotImplementedError
