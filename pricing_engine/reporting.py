"""Reporting helpers for pricing workflows."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import date, datetime, timezone
from typing import Any, Dict, Mapping, Sequence
from uuid import uuid4

from pricing_engine.market_data.curves import FlatCurve, ZeroCurve
from pricing_engine.market_data.dividends import DividendSchedule
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.surfaces import FlatVolSurface, SmileVolSurface
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine, EngineSettings
from pricing_engine.products.base import Product


SCHEMA_VERSION = "1.1"


def new_run_id() -> str:
    return str(uuid4())


def build_report_meta(
    report_type: str,
    run_id: str,
    asof: date,
    inputs: Mapping[str, Any],
    generator: str,
    notes: str = "",
    assumptions: Sequence[str] = (),
    model: str = "",
    engine: str = "",
) -> Dict[str, Any]:
    meta = {
        "schema_version": SCHEMA_VERSION,
        "report_type": report_type,
        "run_id": run_id,
        "asof": asof.isoformat(),
        "created_at": _utc_now().isoformat(),
        "generator": generator,
        "inputs": _safe_json(inputs),
    }
    if model:
        meta["model"] = model
    if engine:
        meta["engine"] = engine
    if assumptions:
        meta["assumptions"] = list(assumptions)
    if notes:
        meta["notes"] = notes
    return meta


def serialize_product(product: Product) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"product_type": product.product_type}
    if is_dataclass(product):
        payload.update(_safe_json(asdict(product)))
    else:
        payload.update(_safe_json(getattr(product, "__dict__", {})))
    metadata = product.metadata() if hasattr(product, "metadata") else {}
    if metadata:
        payload["metadata"] = _safe_json(metadata)
    return payload


def serialize_model(model: Model) -> Dict[str, Any]:
    return {"name": model.name, "params": _safe_json(model.params())}


def serialize_engine(engine: Engine, settings: EngineSettings) -> Dict[str, Any]:
    params: Dict[str, Any] = {}
    if is_dataclass(engine):
        params = asdict(engine)
    else:
        params = getattr(engine, "__dict__", {}) or {}
    payload = {
        "name": engine.name,
        "settings": _safe_json(getattr(settings, "__dict__", {})),
    }
    if params:
        payload["params"] = _safe_json(params)
    return payload


def serialize_market(market: MarketDataSnapshot) -> Dict[str, Any]:
    return {
        "snapshot_id": market.snapshot_id,
        "asof": market.asof.isoformat(),
        "spot": market.spot,
        "discount_curve": _serialize_curve(market.discount_curve),
        "funding_curve": _serialize_curve(market.funding_curve),
        "borrow_curve": _serialize_curve(market.borrow_curve),
        "vol_surface": _serialize_surface(market.vol_surface),
        "dividends": _serialize_dividends(market.dividends),
    }


def _serialize_curve(curve) -> Dict[str, Any]:
    if isinstance(curve, FlatCurve):
        return {"type": "flat", "rate": curve.rate}
    if isinstance(curve, ZeroCurve):
        return {"type": "zero", "times": list(curve.times), "zero_rates": list(curve.zero_rates)}
    return {"type": curve.__class__.__name__}


def _serialize_surface(surface) -> Dict[str, Any]:
    if isinstance(surface, FlatVolSurface):
        return {"type": "flat", "vol": surface.vol}
    if isinstance(surface, SmileVolSurface):
        return {
            "type": "smile",
            "expiries": list(surface.expiries),
            "atm_vols": list(surface.atm_vols),
            "skew": list(surface.skew),
            "curvature": list(surface.curvature),
            "spot_ref": surface.spot_ref,
        }
    return {"type": surface.__class__.__name__}


def _serialize_dividends(dividends: DividendSchedule) -> Dict[str, Any]:
    return {
        "continuous_yield": dividends.continuous_yield,
        "discrete": [
            {"ex_date": div.ex_date.isoformat(), "amount": div.amount} for div in dividends.discrete
        ],
    }


def _safe_json(payload: Mapping[str, Any]) -> Dict[str, Any]:
    def _convert(obj: Any) -> Any:
        if isinstance(obj, (str, int, float, bool)) or obj is None:
            return obj
        if isinstance(obj, date):
            return obj.isoformat()
        if is_dataclass(obj):
            return _convert(asdict(obj))
        if isinstance(obj, Mapping):
            return {str(k): _convert(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_convert(v) for v in obj]
        return str(obj)

    return {str(k): _convert(v) for k, v in payload.items()}


def _utc_now() -> datetime:
    tz = getattr(datetime, "UTC", timezone.utc)
    return datetime.now(tz)
