"""Profit & Loss attribution (PLA) tests for FRTB."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from market_risk.utils import correlation, mean, stdev


@dataclass(frozen=True)
class PLATestResult:
    correlation: float
    std_ratio: float
    beta: float
    r2: float
    mean_error: float
    mean_abs_error: float
    rmse: float
    explained_ratio: float
    observations: int
    correlation_pass: bool
    std_ratio_pass: bool
    status: str


class PLATester:
    def __init__(self, min_corr: float = 0.8, std_ratio_min: float = 0.8, std_ratio_max: float = 1.25) -> None:
        self.min_corr = min_corr
        self.std_ratio_min = std_ratio_min
        self.std_ratio_max = std_ratio_max

    def run(self, hpl: Sequence[float], rtpl: Sequence[float]) -> PLATestResult:
        if len(hpl) != len(rtpl) or len(hpl) < 2:
            return PLATestResult(
                correlation=0.0,
                std_ratio=0.0,
                beta=0.0,
                r2=0.0,
                mean_error=0.0,
                mean_abs_error=0.0,
                rmse=0.0,
                explained_ratio=0.0,
                observations=len(hpl),
                correlation_pass=False,
                std_ratio_pass=False,
                status="fail",
            )
        corr = correlation(hpl, rtpl)
        std_hpl = stdev(hpl)
        std_rtpl = stdev(rtpl)
        ratio = std_hpl / std_rtpl if std_rtpl > 0.0 else 0.0
        beta = _beta(hpl, rtpl)
        residuals = [h - r for h, r in zip(hpl, rtpl)]
        mean_error = mean(residuals)
        mean_abs_error = mean(abs(r) for r in residuals)
        rmse = (mean(r * r for r in residuals)) ** 0.5 if residuals else 0.0
        r2 = corr * corr
        explained_ratio = _explained_ratio(hpl, residuals)
        corr_pass = corr >= self.min_corr
        ratio_pass = self.std_ratio_min <= ratio <= self.std_ratio_max
        status = "pass" if (corr_pass and ratio_pass) else "fail"
        return PLATestResult(
            correlation=corr,
            std_ratio=ratio,
            beta=beta,
            r2=r2,
            mean_error=mean_error,
            mean_abs_error=mean_abs_error,
            rmse=rmse,
            explained_ratio=explained_ratio,
            observations=len(hpl),
            correlation_pass=corr_pass,
            std_ratio_pass=ratio_pass,
            status=status,
        )


def _beta(hpl: Sequence[float], rtpl: Sequence[float]) -> float:
    if len(hpl) < 2:
        return 0.0
    mu_r = mean(rtpl)
    var_r = mean((r - mu_r) ** 2 for r in rtpl)
    if var_r <= 0.0:
        return 0.0
    cov = mean((h - mean(hpl)) * (r - mu_r) for h, r in zip(hpl, rtpl))
    return cov / var_r


def _explained_ratio(hpl: Sequence[float], residuals: Sequence[float]) -> float:
    var_h = mean((h - mean(hpl)) ** 2 for h in hpl)
    var_res = mean((r - mean(residuals)) ** 2 for r in residuals)
    if var_h <= 0.0:
        return 0.0
    return max(0.0, 1.0 - var_res / var_h)
