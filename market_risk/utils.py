"""Market risk utilities."""

from __future__ import annotations

import math
from typing import Iterable, Sequence


def linear_interp(x: float, xs: Sequence[float], ys: Sequence[float]) -> float:
    if not xs:
        raise ValueError("empty grid")
    if len(xs) != len(ys):
        raise ValueError("grid size mismatch")
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


def quantile(data: Sequence[float], level: float) -> float:
    if not data:
        return 0.0
    if level <= 0.0:
        return min(data)
    if level >= 1.0:
        return max(data)
    ordered = sorted(data)
    idx = level * (len(ordered) - 1)
    low = int(math.floor(idx))
    high = int(math.ceil(idx))
    if low == high:
        return ordered[low]
    frac = idx - low
    return ordered[low] + frac * (ordered[high] - ordered[low])


def mean(data: Iterable[float]) -> float:
    values = list(data)
    if not values:
        return 0.0
    return sum(values) / len(values)


def stdev(data: Iterable[float]) -> float:
    values = list(data)
    if len(values) < 2:
        return 0.0
    mu = mean(values)
    var = sum((x - mu) ** 2 for x in values) / (len(values) - 1)
    return math.sqrt(var)


def correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0
    mu_x = mean(xs)
    mu_y = mean(ys)
    num = sum((x - mu_x) * (y - mu_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mu_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mu_y) ** 2 for y in ys))
    denom = den_x * den_y
    if denom <= 0.0:
        return 0.0
    return num / denom
