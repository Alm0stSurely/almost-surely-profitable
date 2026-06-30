#!/usr/bin/env python3
"""
Benchmark script: backtest with vs without cooldown guardrails.

Compares strategy performance on historical data to quantify the
impact of position cooldown constraints.

Usage:
    python benchmark_backtest_cooldowns.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent / "src"))

from backtest.backtest import BacktestEngine
from backtest.backtest_cooldown import CooldownConfig


def run_benchmark(
    start_date: str = "2024-01-01",
    end_date: str = "2024-12-31",
    tickers: list = None,
    strategy: str = "random",
    seed: int = 42
):
    """Run backtest with and without cooldowns, compare results."""
    tickers = tickers or ["SPY", "QQQ", "GLD", "IWM", "TLT"]

    print(f"\n{'='*70}")
    print(f"BACKTEST COOLDOWN BENCHMARK")
    print(f"{'='*70}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Strategy: {strategy}")
    print(f"Tickers: {', '.join(tickers)}")
    print(f"{'='*70}\n")

    # Baseline: no cooldowns
    print("Running baseline (no cooldowns)...")
    t0 = time.time()
    engine_baseline = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        rebalance_frequency="daily",
        enable_cooldowns=False
    )
    result_baseline = engine_baseline.run_backtest(strategy=strategy, random_seed=seed)
    t_baseline = time.time() - t0

    # With cooldowns
    print("Running with cooldown guardrails...")
    config = CooldownConfig(
        min_hold_days=5,
        flip_cooldown_days=10,
        max_trades_per_week=2,
        stop_loss_threshold_pct=5.0
    )
    t0 = time.time()
    engine_cooldown = BacktestEngine(
        start_date=start_date,
        end_date=end_date,
        tickers=tickers,
        rebalance_frequency="daily",
        enable_cooldowns=True,
        cooldown_config=config
    )
    result_cooldown = engine_cooldown.run_backtest(strategy=strategy, random_seed=seed)
    t_cooldown = time.time() - t0

    # Print comparison
    print(f"\n{'='*70}")
    print("RESULTS COMPARISON")
    print(f"{'='*70}")
    print(f"{'Metric':<25} {'Baseline':>15} {'+Cooldowns':>15} {'Delta':>12}")
    print("-" * 70)

    metrics = [
        ("Final Value", result_baseline['final_value'], result_cooldown['final_value'], "€{:,.2f}"),
        ("Total Return", result_baseline['total_return']*100, result_cooldown['total_return']*100, "{:>8.2f}%"),
        ("Sharpe Ratio", result_baseline['sharpe_ratio'], result_cooldown['sharpe_ratio'], "{:>8.2f}"),
        ("Max Drawdown", result_baseline['max_drawdown']*100, result_cooldown['max_drawdown']*100, "{:>8.2f}%"),
        ("Volatility", result_baseline['volatility']*100, result_cooldown['volatility']*100, "{:>8.2f}%"),
        ("Num Trades", result_baseline['num_trades'], result_cooldown['num_trades'], "{:>8d}"),
        ("Win Rate", result_baseline['win_rate']*100, result_cooldown['win_rate']*100, "{:>8.2f}%"),
    ]

    for name, baseline, cooldown, fmt in metrics:
        delta = cooldown - baseline
        baseline_str = fmt.format(baseline)
        cooldown_str = fmt.format(cooldown)
        delta_str = f"{delta:+>8.2f}"
        if "Return" in name or "Drawdown" in name or "Rate" in name or "Volatility" in name:
            delta_str = f"{delta:+>8.2f}"
        elif "Value" in name:
            delta_str = f"€{delta:+,.2f}"
        else:
            delta_str = f"{delta:+>8.2f}"
        print(f"{name:<25} {baseline_str:>15} {cooldown_str:>15} {delta_str:>12}")

    # Cooldown-specific metrics
    if 'cooldown_metrics' in result_cooldown:
        cm = result_cooldown['cooldown_metrics']
        print(f"\n{'='*70}")
        print("COOLDOWN GUARDRAIL EFFECTIVENESS")
        print(f"{'='*70}")
        print(f"  Blocked buys:          {cm['blocked_buys']:>6}")
        print(f"  Blocked sells:         {cm['blocked_sells']:>6}")
        print(f"  Total blocked:         {cm['total_blocked']:>6}")
        print(f"  Stop-loss overrides:   {cm['stop_loss_overrides']:>6}")
        print(f"  Trade attempts:        {cm['trade_attempts']:>6}")
        print(f"  Block rate:            {cm['block_rate']*100:>6.1f}%")

    print(f"\n{'='*70}")
    print("PERFORMANCE")
    print(f"{'='*70}")
    print(f"  Baseline runtime:      {t_baseline:>6.3f}s")
    print(f"  Cooldown runtime:      {t_cooldown:>6.3f}s")
    print(f"  Overhead:              {(t_cooldown/t_baseline - 1)*100:>6.1f}%")
    print(f"{'='*70}\n")

    return {
        "baseline": result_baseline,
        "cooldown": result_cooldown,
        "runtime_baseline": t_baseline,
        "runtime_cooldown": t_cooldown,
    }


if __name__ == "__main__":
    results = run_benchmark()
