"""Calibration objective definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class WeightedLeastSquares:
    weights: Sequence[float]

    def evaluate(self, errors: Iterable[float]) -> float:
        return sum(w * e * e for w, e in zip(self.weights, errors))
