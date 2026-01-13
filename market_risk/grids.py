"""Sensitivity grid configuration and updates."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Sequence


@dataclass(frozen=True)
class RiskGrid:
    name: str
    moneyness: Sequence[float]
    expiries: Sequence[float]
    version: int = 1
    updated_at: date = field(default_factory=date.today)

    def update(self, moneyness: Sequence[float] | None = None, expiries: Sequence[float] | None = None) -> "RiskGrid":
        return RiskGrid(
            name=self.name,
            moneyness=moneyness if moneyness is not None else self.moneyness,
            expiries=expiries if expiries is not None else self.expiries,
            version=self.version + 1,
            updated_at=date.today(),
        )


@dataclass(frozen=True)
class GridUpdateRecord:
    grid_name: str
    old_version: int
    new_version: int
    updated_at: date
    note: str = ""
