"""Run reporting and persistence utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
import hashlib
import json
import os
from typing import Any, Dict, Mapping, Optional
from uuid import uuid4


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    name: str
    asof: str
    started_at: str
    finished_at: str
    params: Dict[str, Any]
    outputs: Dict[str, str]
    checksums: Dict[str, str]
    notes: str = ""
    version: str = ""


class RunLogger:
    def __init__(self, output_root: str = "market_risk/reports/runs") -> None:
        self.output_root = output_root

    def log_run(
        self,
        name: str,
        asof: date,
        params: Mapping[str, Any],
        outputs: Mapping[str, str],
        notes: str = "",
        version: str = "",
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
    ) -> str:
        run_id = str(uuid4())
        started_at = started_at or datetime.utcnow()
        finished_at = finished_at or datetime.utcnow()
        run_dir = os.path.join(self.output_root, asof.strftime("%Y%m%d"))
        os.makedirs(run_dir, exist_ok=True)
        record = RunRecord(
            run_id=run_id,
            name=name,
            asof=asof.isoformat(),
            started_at=started_at.isoformat(),
            finished_at=finished_at.isoformat(),
            params=_safe_json(params),
            outputs={k: str(v) for k, v in outputs.items()},
            checksums={k: _sha256(v) for k, v in outputs.items() if os.path.exists(v)},
            notes=notes,
            version=version,
        )
        path = os.path.join(run_dir, f"run_{run_id}.json")
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(record.__dict__, handle, indent=2)
        return path


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, date):
            return obj.isoformat()
        if isinstance(obj, Mapping):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return str(obj)

    return {str(k): _convert(v) for k, v in payload.items()}
