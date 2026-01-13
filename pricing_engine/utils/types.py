"""Shared types and enums."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class OptionType(str, Enum):
    CALL = "call"
    PUT = "put"


class BarrierType(str, Enum):
    UP_IN = "up_in"
    UP_OUT = "up_out"
    DOWN_IN = "down_in"
    DOWN_OUT = "down_out"


class AveragingType(str, Enum):
    PRICE = "avg_price"
    STRIKE = "avg_strike"


@dataclass(frozen=True)
class Quote:
    instrument_id: str
    value: float
    quote_type: str
    asof: Optional[str] = None
