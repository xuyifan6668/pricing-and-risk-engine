"""Adjoint/automatic differentiation hooks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class AdjointConfig:
    enabled: bool = False
    tape_memory_limit_mb: int = 512


def adjoint_supported(engine_name: str) -> bool:
    return engine_name in {"monte_carlo", "pde"}


def compute_adjoint_greeks(payload: Dict[str, float]) -> Dict[str, float]:
    # Placeholder for AAD integration.
    return {}
