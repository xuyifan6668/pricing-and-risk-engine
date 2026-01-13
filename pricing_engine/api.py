"""Core pricing API."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional
from uuid import uuid4

from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.models.base import Model
from pricing_engine.numerics.base import Engine, EngineSettings
from pricing_engine.products.base import Product
from pricing_engine.risk.greeks import GreeksCalculator, GreekRequest


@dataclass(frozen=True)
class PricingSettings:
    engine_settings: EngineSettings = EngineSettings()
    compute_greeks: bool = False
    greek_request: Optional[GreekRequest] = None
    diagnostics: bool = True


@dataclass(frozen=True)
class PricingContext:
    product: Product
    market: MarketDataSnapshot
    model: Model
    engine: Engine
    settings: PricingSettings
    run_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(frozen=True)
class PricingResult:
    price: float
    greeks: Dict[str, float]
    metrics: Dict[str, float]
    diagnostics: Dict[str, str]
    run_id: str


def price(
    product: Product,
    market: MarketDataSnapshot,
    model: Model,
    engine: Engine,
    settings: Optional[PricingSettings] = None,
) -> PricingResult:
    """Price a product and optionally compute Greeks."""

    settings = settings or PricingSettings()
    context = PricingContext(
        product=product,
        market=market,
        model=model,
        engine=engine,
        settings=settings,
    )

    raw = engine.price(product, market, model, settings.engine_settings)
    price_value = raw.get("price")
    if price_value is None:
        raise ValueError("engine returned no price")
    price_value = float(price_value)

    metrics: Dict[str, float] = {}
    for key, value in raw.items():
        if key == "price":
            continue
        if isinstance(value, (int, float)):
            metrics[key] = float(value)
    stderr = metrics.get("stderr")
    if stderr is not None and stderr >= 0.0:
        z_score = 1.96
        metrics["ci_level"] = 0.95
        metrics["ci_lower"] = price_value - z_score * stderr
        metrics["ci_upper"] = price_value + z_score * stderr

    greeks: Dict[str, float] = {}
    if settings.compute_greeks:
        request = settings.greek_request or GreekRequest.default()
        greeks = GreeksCalculator().calculate(context, request)

    diagnostics: Dict[str, str] = {}
    if settings.diagnostics:
        diagnostics = {
            "engine": engine.name,
            "model": model.name,
            "run_id": context.run_id,
            "market_snapshot": market.snapshot_id,
        }

    return PricingResult(
        price=price_value,
        greeks=greeks,
        metrics=metrics,
        diagnostics=diagnostics,
        run_id=context.run_id,
    )
