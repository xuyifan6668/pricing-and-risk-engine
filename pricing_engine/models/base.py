"""Model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Iterable, Sequence

from pricing_engine.market_data.snapshot import MarketDataSnapshot


@dataclass(frozen=True)
class ModelConstraints:
    bounds: Dict[str, tuple[float, float]]
    linear: Iterable[str] = ()


class Model(ABC):
    """Abstract model interface."""

    @property
    @abstractmethod
    def name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def params(self) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def set_params(self, params: Dict[str, float]) -> None:
        raise NotImplementedError

    @abstractmethod
    def initial_guess(self, market: MarketDataSnapshot) -> Dict[str, float]:
        raise NotImplementedError

    @abstractmethod
    def constraints(self, market: MarketDataSnapshot) -> ModelConstraints:
        raise NotImplementedError

    def simulate_paths(
        self,
        market: MarketDataSnapshot,
        timesteps: int,
        num_paths: int,
        horizon: float,
        seed: int | None,
    ) -> Sequence[Sequence[float]]:
        raise NotImplementedError
