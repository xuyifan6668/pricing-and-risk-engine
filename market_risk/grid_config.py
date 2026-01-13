"""Grid config persistence helpers."""

from __future__ import annotations

from datetime import datetime
import json
import os
from typing import Any, Dict, Optional


DEFAULT_GRID_CONFIG_PATH = "market_risk/reports/grid_config_latest.json"


def load_grid_config(path: str = DEFAULT_GRID_CONFIG_PATH) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload


def save_grid_config(path: str, payload: Dict[str, Any]) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = dict(payload)
    payload.setdefault("updated_at", datetime.utcnow().isoformat())
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return path
