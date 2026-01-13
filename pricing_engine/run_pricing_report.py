"""Generate a pricing report for a small sample portfolio."""

from __future__ import annotations

import argparse
import csv
import os
import sys
from dataclasses import dataclass
from datetime import date, timedelta

if __package__ is None and __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pricing_engine.api import PricingSettings, price
from pricing_engine.market_data.curves import FlatCurve
from pricing_engine.market_data.dividends import DividendSchedule
from pricing_engine.market_data.snapshot import MarketDataSnapshot
from pricing_engine.market_data.surfaces import FlatVolSurface
from pricing_engine.models.black_scholes import BlackScholesModel
from pricing_engine.numerics.analytic import AnalyticEngine
from pricing_engine.numerics.base import EngineSettings
from pricing_engine.numerics.trees import TreeEngine
from pricing_engine.products.base import Underlying
from pricing_engine.products.vanilla import AmericanOption, EuropeanOption
from pricing_engine.reporting import (
    SCHEMA_VERSION,
    build_report_meta,
    new_run_id,
    serialize_engine,
    serialize_market,
    serialize_model,
    serialize_product,
)
from pricing_engine.utils.types import OptionType


@dataclass(frozen=True)
class TradeSpec:
    trade_id: str
    product: object
    model: object
    engine: object
    settings: PricingSettings


def build_trade_results(trades: list[TradeSpec], market: MarketDataSnapshot) -> list[dict]:
    trade_results = []
    for trade in trades:
        result = price(
            product=trade.product,
            market=market,
            model=trade.model,
            engine=trade.engine,
            settings=trade.settings,
        )
        trade_results.append(
            {
                "trade_id": trade.trade_id,
                "product": serialize_product(trade.product),
                "model": serialize_model(trade.model),
                "engine": serialize_engine(trade.engine, trade.settings.engine_settings),
                "settings": {
                    "compute_greeks": trade.settings.compute_greeks,
                    "engine_settings": trade.settings.engine_settings.__dict__,
                },
                "result": {
                    "price": result.price,
                    "greeks": result.greeks,
                    "metrics": result.metrics,
                    "diagnostics": result.diagnostics,
                },
            }
        )
    return trade_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a pricing report for sample trades.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="pricing_engine/reports",
        help="Directory to write JSON/CSV reports.",
    )
    args = parser.parse_args()

    market = build_sample_market()
    trades = build_sample_trades(market.asof)

    run_id = new_run_id()
    meta = build_report_meta(
        report_type="pricing.run",
        run_id=run_id,
        asof=market.asof,
        inputs={"trade_count": len(trades), "market_snapshot_id": market.snapshot_id},
        generator="pricing_engine/run_pricing_report.py",
        assumptions=(
            "Flat curves and flat vol surface for sample pricing runs.",
            "Black-Scholes model for all trades in this demo.",
        ),
    )

    trade_results = build_trade_results(trades, market)

    payload = {
        "meta": meta,
        "summary": {"trade_count": len(trade_results), "currency": "USD"},
        "market": serialize_market(market),
        "trades": trade_results,
    }

    json_path, csv_path, html_path = write_report_files(
        args.output_dir,
        market.asof,
        payload,
        trade_results,
        report_run_id=run_id,
    )

    print("=== Pricing Report ===")
    print(f"Asof: {market.asof}")
    print(f"Trades: {len(trade_results)}")
    print(f"JSON report: {json_path}")
    print(f"CSV report: {csv_path}")
    print(f"HTML report: {html_path}")
    print(f"Schema version: {SCHEMA_VERSION}")


def build_sample_market(asof: date | None = None) -> MarketDataSnapshot:
    asof = asof or date(2024, 1, 2)
    return MarketDataSnapshot(
        asof=asof,
        spot=100.0,
        discount_curve=FlatCurve(0.02),
        funding_curve=FlatCurve(0.02),
        borrow_curve=FlatCurve(0.0),
        vol_surface=FlatVolSurface(0.2),
        dividends=DividendSchedule(continuous_yield=0.01),
    )


def build_sample_trades(asof: date) -> list[TradeSpec]:
    underlying = Underlying("XYZ")
    model = BlackScholesModel(sigma=0.2)
    trades = [
        TradeSpec(
            trade_id="T-001",
            product=EuropeanOption(
                underlying=underlying,
                currency="USD",
                maturity=asof + timedelta(days=180),
                strike=100.0,
                option_type=OptionType.CALL,
            ),
            model=model,
            engine=AnalyticEngine(),
            settings=PricingSettings(
                engine_settings=EngineSettings(deterministic=True, max_steps=200),
                compute_greeks=True,
            ),
        ),
        TradeSpec(
            trade_id="T-002",
            product=AmericanOption(
                underlying=underlying,
                currency="USD",
                maturity=asof + timedelta(days=365),
                strike=95.0,
                option_type=OptionType.PUT,
            ),
            model=model,
            engine=TreeEngine(steps=200),
            settings=PricingSettings(
                engine_settings=EngineSettings(deterministic=True, max_steps=200),
                compute_greeks=False,
            ),
        ),
    ]
    return trades


def write_report_files(
    output_dir: str,
    asof: date,
    payload: dict,
    trades: list[dict],
    report_run_id: str,
) -> tuple[str, str, str]:
    os.makedirs(output_dir, exist_ok=True)
    stamp = asof.strftime("%Y%m%d")
    json_path = os.path.join(output_dir, f"pricing_report_{stamp}.json")
    csv_path = os.path.join(output_dir, f"pricing_report_{stamp}.csv")
    html_path = os.path.join(output_dir, f"pricing_report_{stamp}.html")

    with open(json_path, "w", encoding="utf-8") as handle:
        import json

        json.dump(payload, handle, indent=2)

    with open(html_path, "w", encoding="utf-8") as handle:
        handle.write(render_pricing_report_html(payload))

    with open(csv_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "report_run_id",
                "trade_id",
                "product_type",
                "option_type",
                "strike",
                "maturity",
                "price",
                "stderr",
                "ci_lower",
                "ci_upper",
                "delta",
                "gamma",
                "vega",
                "theta",
                "rho",
                "engine",
                "model",
                "run_id",
            ]
        )
        for trade in trades:
            product = trade["product"]
            greeks = trade["result"]["greeks"]
            metrics = trade["result"].get("metrics", {})
            diagnostics = trade["result"]["diagnostics"]
            writer.writerow(
                [
                    report_run_id,
                    trade["trade_id"],
                    product.get("product_type", ""),
                    product.get("option_type", ""),
                    product.get("strike", ""),
                    product.get("maturity", ""),
                    trade["result"]["price"],
                    metrics.get("stderr", ""),
                    metrics.get("ci_lower", ""),
                    metrics.get("ci_upper", ""),
                    greeks.get("delta", ""),
                    greeks.get("gamma", ""),
                    greeks.get("vega", ""),
                    greeks.get("theta", ""),
                    greeks.get("rho", ""),
                    trade["engine"]["name"],
                    trade["model"]["name"],
                    diagnostics.get("run_id", ""),
                ]
            )
    return json_path, csv_path, html_path


def render_pricing_report_html(payload: dict) -> str:
    meta = payload.get("meta", {})
    summary = payload.get("summary", {})
    market = payload.get("market", {})
    trades = payload.get("trades", [])

    def fmt(val: object, digits: int = 4) -> str:
        if val is None:
            return ""
        if isinstance(val, float):
            return f"{val:,.{digits}f}"
        return str(val)

    rows = []
    for trade in trades:
        product = trade.get("product", {})
        result = trade.get("result", {})
        greeks = result.get("greeks", {})
        metrics = result.get("metrics", {})
        diagnostics = result.get("diagnostics", {})
        rows.append(
            "<tr>"
            f"<td>{trade.get('trade_id', '')}</td>"
            f"<td>{product.get('product_type', '')}</td>"
            f"<td>{product.get('option_type', '')}</td>"
            f"<td>{fmt(product.get('strike', ''), 4)}</td>"
            f"<td>{product.get('maturity', '')}</td>"
            f"<td>{fmt(result.get('price', ''), 4)}</td>"
            f"<td>{fmt(metrics.get('stderr', ''), 6)}</td>"
            f"<td>{fmt(metrics.get('ci_lower', ''), 4)}</td>"
            f"<td>{fmt(metrics.get('ci_upper', ''), 4)}</td>"
            f"<td>{fmt(greeks.get('delta', ''), 6)}</td>"
            f"<td>{fmt(greeks.get('gamma', ''), 6)}</td>"
            f"<td>{fmt(greeks.get('vega', ''), 6)}</td>"
            f"<td>{fmt(greeks.get('theta', ''), 6)}</td>"
            f"<td>{fmt(greeks.get('rho', ''), 6)}</td>"
            f"<td>{trade.get('engine', {}).get('name', '')}</td>"
            f"<td>{trade.get('model', {}).get('name', '')}</td>"
            f"<td>{diagnostics.get('run_id', '')}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>Pricing Report</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; background: #f5f3ef; color: #1f1f1f; }}
    h1 {{ margin-top: 0; }}
    table {{ border-collapse: collapse; width: 100%; background: #fff; }}
    th, td {{ border: 1px solid #d7d0c5; padding: 6px 8px; font-size: 13px; text-align: left; }}
    th {{ background: #ece7df; }}
    .section {{ margin-bottom: 24px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 6px; }}
    .meta-item {{ background: #fff; border: 1px solid #d7d0c5; padding: 8px; font-size: 13px; }}
  </style>
</head>
<body>
  <h1>Pricing Report</h1>
  <div class="section meta-grid">
    <div class="meta-item">Run ID: {meta.get("run_id", "")}</div>
    <div class="meta-item">As-of: {meta.get("asof", "")}</div>
    <div class="meta-item">Created: {meta.get("created_at", "")}</div>
    <div class="meta-item">Trades: {summary.get("trade_count", "")}</div>
    <div class="meta-item">Currency: {summary.get("currency", "")}</div>
    <div class="meta-item">Market Snapshot: {market.get("snapshot_id", "")}</div>
  </div>
  <div class="section">
    <h2>Trades</h2>
    <table>
      <thead>
        <tr>
          <th>Trade ID</th>
          <th>Product</th>
          <th>Option</th>
          <th>Strike</th>
          <th>Maturity</th>
          <th>Price</th>
          <th>StdErr</th>
          <th>CI Low</th>
          <th>CI High</th>
          <th>Delta</th>
          <th>Gamma</th>
          <th>Vega</th>
          <th>Theta</th>
          <th>Rho</th>
          <th>Engine</th>
          <th>Model</th>
          <th>Pricing Run ID</th>
        </tr>
      </thead>
      <tbody>
        {''.join(rows)}
      </tbody>
    </table>
  </div>
</body>
</html>
"""


if __name__ == "__main__":
    main()
