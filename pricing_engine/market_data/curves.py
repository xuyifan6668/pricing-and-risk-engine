"""Curve abstractions."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import math
from typing import Sequence


class Curve(ABC):
    """Abstract curve interface."""

    @abstractmethod
    def df(self, t: float) -> float:
        raise NotImplementedError

    def zero_rate(self, t: float) -> float:
        if t <= 0.0:
            return 0.0
        return -math.log(self.df(t)) / t

    def fwd_rate(self, t1: float, t2: float) -> float:
        if t2 <= t1:
            raise ValueError("t2 must be greater than t1")
        df1 = self.df(t1)
        df2 = self.df(t2)
        return (df1 / df2 - 1.0) / (t2 - t1)


@dataclass(frozen=True)
class FlatCurve(Curve):
    rate: float

    def df(self, t: float) -> float:
        return math.exp(-self.rate * max(t, 0.0))


def _linear_interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty curve")
    if len(xs) != len(ys):
        raise ValueError("curve size mismatch")
    if x <= xs[0]:
        return ys[0]
    if x >= xs[-1]:
        return ys[-1]
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0 = xs[i - 1]
            x1 = xs[i]
            y0 = ys[i - 1]
            y1 = ys[i]
            w = (x - x0) / (x1 - x0)
            return y0 + w * (y1 - y0)
    return ys[-1]


@dataclass(frozen=True)
class ZeroCurve(Curve):
    times: Sequence[float]
    zero_rates: Sequence[float]

    def df(self, t: float) -> float:
        if t <= 0.0:
            return 1.0
        r = _linear_interp(t, list(self.times), list(self.zero_rates))
        return math.exp(-r * t)
