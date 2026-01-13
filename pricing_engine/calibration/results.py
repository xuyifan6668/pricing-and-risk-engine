"""Calibration results."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass(frozen=True)
class CalibrationResult:
    status: str
    params: Optional[Dict[str, float]] = None
    diagnostics: Optional[Dict[str, float]] = None
