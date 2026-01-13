"""Shared derivative helpers for grid calculations."""

from __future__ import annotations

import math
from typing import List, Sequence


def first_derivative(
    xs: Sequence[float],
    values: Sequence[Sequence[float]],
    i: int,
    j: int,
    scale: float,
) -> float:
    x0 = xs[i - 1]
    x1 = xs[i + 1]
    f0 = values[i - 1][j]
    f1 = values[i + 1][j]
    denom = (x1 - x0) * scale
    if denom == 0.0:
        return 0.0
    return (f1 - f0) / denom


def second_derivative(
    xs: Sequence[float],
    values: Sequence[Sequence[float]],
    i: int,
    j: int,
    scale: float,
) -> float:
    x_im1 = xs[i - 1]
    x_i = xs[i]
    x_ip1 = xs[i + 1]

    f_im1 = values[i - 1][j]
    f_i = values[i][j]
    f_ip1 = values[i + 1][j]

    denom = (x_im1 - x_i) * (x_im1 - x_ip1) * (x_i - x_ip1)
    if denom == 0.0:
        return 0.0
    second = 2.0 * (
        f_im1 * (x_i - x_ip1)
        + f_i * (x_ip1 - x_im1)
        + f_ip1 * (x_im1 - x_i)
    ) / denom
    scale_sq = scale * scale
    if scale_sq == 0.0:
        return 0.0
    return second / scale_sq


def cross_derivative(
    xs: Sequence[float],
    ys: Sequence[float],
    values: Sequence[Sequence[float]],
    i: int,
    j: int,
    scale_x: float,
    scale_y: float,
) -> float:
    x_im1 = xs[i - 1]
    x_ip1 = xs[i + 1]
    y_jm1 = ys[j - 1]
    y_jp1 = ys[j + 1]

    f_pp = values[i + 1][j + 1]
    f_pm = values[i + 1][j - 1]
    f_mp = values[i - 1][j + 1]
    f_mm = values[i - 1][j - 1]

    denom_x = (x_ip1 - x_im1) * scale_x
    denom_y = (y_jp1 - y_jm1) * scale_y
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    return (f_pp - f_pm - f_mp + f_mm) / (4.0 * denom_x * denom_y)


def transpose(values: Sequence[Sequence[float]]) -> List[List[float]]:
    return [list(row) for row in zip(*values)]


def build_grid_map(xs: Sequence[float], ys: Sequence[float], fill: float) -> List[List[float]]:
    return [[fill for _ in ys] for _ in xs]


def flatten(values: Sequence[Sequence[float]]) -> List[float]:
    return [v for row in values for v in row if not math.isnan(v)]


def mean_ignore_nan(values: Sequence[float]) -> float:
    clean = [v for v in values if not math.isnan(v)]
    if not clean:
        return 0.0
    return sum(clean) / len(clean)
