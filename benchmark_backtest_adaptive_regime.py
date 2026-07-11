#!/usr/bin/env python3
"""
Benchmark script: backtest cooldown under adaptive volatility regimes.

Quantifies how the dynamic trade cap and adaptive stop-loss change the
number of allowed trades and stop-loss overrides in a synthetic multi-ticker
scenario. This mirrors the live PositionCooldownManager behavior so that
backtests and production use the same guardrail logic.

Usage:
    python benchmark_backtest_adaptive_regime.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest.backtest_cooldown import BacktestCooldownManager, CooldownConfig


def run_multi_ticker_scenario(regime: str, days: int = 60):
    """
    Simulate daily decisions for five tickers.

    Each day we try to buy one of the rotating tickers (no existing position).
    Once a position is held, we attempt to sell it every day. The price path is
    chosen so that some positions are stopped out within the minimum holding
    period, while others are held to maturity.
    """
    config = CooldownConfig(
        min_hold_days=5,
        flip_cooldown_days=0,  # disable flip cooldown so the weekly cap is the constraint
        max_trades_per_week=2,  # normal regime baseline
        max_trades_high_vol=1,
        max_trades_normal_vol=2,
        max_trades_low_vol=4,
        allow_stop_loss_override=True,
        stop_loss_threshold_pct=5.0,
        stop_loss_high_vol=3.0,
        stop_loss_normal_vol=5.0,
        stop_loss_low_vol=7.0,
        current_vol_regime=regime,
    )
    mgr = BacktestCooldownManager(config=config)

    tickers = ["AAPL", "TSLA", "MSFT", "GOOG", "AMZN"]
    start = datetime(2024, 1, 1)
    allowed_buys = 0
    allowed_sells = 0
    stop_loss_overrides = 0
    blocked = 0

    for i in range(days):
        current_date = start + timedelta(days=i)
        ticker = tickers[i % len(tickers)]
        # Price starts at 100 and drops sharply to trigger stop-loss overrides
        # within the minimum holding period (5 days).
        price = 100.0 - (i * 1.0)  # ~-1% per day, >3% by day 4, >5% by day 6
        avg_price = 100.0

        # Try to buy if we have no active position for this ticker
        if ticker not in mgr.entries:
            allowed, _ = mgr.can_buy(ticker, current_date)
            if allowed:
                mgr.record_entry(ticker, current_date)
                allowed_buys += 1
            else:
                blocked += 1

        # Try to sell every held position
        for held_ticker in list(mgr.entries.keys()):
            allowed, reason = mgr.can_sell(held_ticker, current_date, price, avg_price)
            if allowed:
                if "Stop-loss override" in reason:
                    stop_loss_overrides += 1
                allowed_sells += 1
                mgr.record_exit(held_ticker, current_date)
            else:
                blocked += 1

    metrics = mgr.get_metrics()
    return {
        "regime": regime,
        "allowed_buys": allowed_buys,
        "allowed_sells": allowed_sells,
        "stop_loss_overrides": stop_loss_overrides,
        "blocked": blocked,
        "trade_attempts": metrics["trade_attempts"],
        "block_rate": metrics["block_rate"],
    }


def main():
    print("\n" + "=" * 70)
    print("BACKTEST ADAPTIVE REGIME COOLDOWN BENCHMARK")
    print("=" * 70)
    print("Synthetic scenario: 60 days, 5 rotating tickers, sharp price drop")
    print("Goal: compare allowed trades and stop-loss overrides by regime")
    print("=" * 70 + "\n")

    regimes = ["high", "normal", "low"]
    results = []
    for regime in regimes:
        t0 = time.perf_counter()
        result = run_multi_ticker_scenario(regime)
        result["runtime"] = time.perf_counter() - t0
        results.append(result)

    print(f"{'Regime':<10} {'Buys':>6} {'Sells':>6} {'SL Ovr':>7} {'Blocked':>8} {'Attempts':>9} {'Block %':>8} {'Time (s)':>10}")
    print("-" * 70)
    for r in results:
        print(
            f"{r['regime']:<10} "
            f"{r['allowed_buys']:>6} "
            f"{r['allowed_sells']:>6} "
            f"{r['stop_loss_overrides']:>7} "
            f"{r['blocked']:>8} "
            f"{r['trade_attempts']:>9} "
            f"{r['block_rate']*100:>7.1f}% "
            f"{r['runtime']:>10.4f}"
        )

    print("\n" + "=" * 70)
    print("OBSERVATIONS")
    print("=" * 70)
    high = next(r for r in results if r["regime"] == "high")
    low = next(r for r in results if r["regime"] == "low")
    normal = next(r for r in results if r["regime"] == "normal")

    print(f"  Trade cap:        high=1, normal=2, low=4")
    print(f"  Allowed buys:     high={high['allowed_buys']}, normal={normal['allowed_buys']}, low={low['allowed_buys']}")
    print(f"  Stop-loss regime: high=3%, normal=5%, low=7%")
    print(f"  SL overrides:     high={high['stop_loss_overrides']}, normal={normal['stop_loss_overrides']}, low={low['stop_loss_overrides']}")
    print(f"  Block rate:       high={high['block_rate']*100:.1f}%, normal={normal['block_rate']*100:.1f}%, low={low['block_rate']*100:.1f}%")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
