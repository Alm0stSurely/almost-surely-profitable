"""
Benchmark for backtest non-finite serialization guards.

Measures the cost of the serialization guard paths without relying on mocked
market data or directory side effects. We benchmark the recursive sanitizer on
a realistic backtest result dict and the degenerate metric guard path in
isolation.
"""

import sys
import time
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd

from backtest.backtest import BacktestEngine
from portfolio.portfolio import Portfolio
from utils import dump_json_safe, sanitize_for_json


def _make_result_dict() -> dict:
    """Build a realistic backtest result dict with curves and per-day records."""
    n_days = 65
    return {
        "start_date": "2024-01-01",
        "end_date": "2024-03-31",
        "strategy": "buy_and_hold",
        "initial_capital": 10000.0,
        "final_value": 10450.0,
        "total_return": 0.045,
        "annualized_return": 0.18,
        "sharpe_ratio": 1.25,
        "sortino_ratio": 1.80,
        "max_drawdown": 0.03,
        "calmar_ratio": 6.0,
        "omega_ratio": 2.5,
        "win_rate": 0.55,
        "profit_factor": 1.8,
        "beta": 0.95,
        "alpha": 0.01,
        "volatility": 0.12,
        "num_trades": 3,
        "equity_curve": [10000.0 + i * 7.0 for i in range(n_days)],
        "drawdown_curve": [0.0] * n_days,
        "daily_returns": [0.0007] * (n_days - 1),
        "daily_results": [
            {
                "date": f"2024-01-{(i % 31) + 1:02d}",
                "total_value": 10000.0 + i * 7.0,
                "cash": 5000.0,
                "positions_value": 5000.0 + i * 7.0,
                "total_return_pct": i * 0.01,
                "num_positions": 1,
            }
            for i in range(n_days)
        ],
    }


class SuppressOutput:
    """Discard both stdout and stderr during a benchmark run."""
    class NullIO:
        def write(self, _): pass
        def flush(self): pass

    def __enter__(self):
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = self.NullIO()
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr


def _run(func):
    with SuppressOutput(), warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        return func()


def benchmark_backtest_serialization(rounds=100):
    """Benchmark recursive sanitize + dump_json_safe on a backtest-sized dict."""
    result = _make_result_dict()

    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        cleaned = sanitize_for_json(result)
        dump_json_safe(cleaned, open("/dev/null", "w"), indent=2, default=str)
        times.append(time.perf_counter() - start)

    print(f"Backtest result serialization: {min(times)*1e6:.2f} μs min, "
          f"{sum(times)/len(times)*1e6:.2f} μs avg over {rounds} rounds")


def benchmark_comparison_backtest_serialization(rounds=50):
    """Benchmark sanitize + dump on a dict of three strategy results."""
    results = {s: _make_result_dict() for s in ("buy_and_hold", "equal_weight", "random")}

    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        cleaned = sanitize_for_json(results)
        dump_json_safe(cleaned, open("/dev/null", "w"), indent=2, default=str)
        times.append(time.perf_counter() - start)

    print(f"Comparison backtest serialization: {min(times)*1e6:.2f} μs min, "
          f"{sum(times)/len(times)*1e6:.2f} μs avg over {rounds} rounds")


def benchmark_degenerate_metrics(rounds=10000):
    """Benchmark the cost of the zero-loss metric guard path."""
    with tempfile.TemporaryDirectory() as tmp:
        engine = BacktestEngine(
            start_date="2024-01-01",
            end_date="2024-01-05",
            tickers=["SPY"],
        )
        engine.results = [
            {"total_value": 10000.0 + i * 100, "cash": 5000.0,
             "positions_value": 5000.0 + i * 100, "total_return_pct": i * 1.0,
             "num_positions": 1}
            for i in range(5)
        ]
        engine.portfolio = Portfolio(
            state_file="bt_test.json",
            trades_file="bt_trades.json",
            data_dir=tmp,
        )
        engine.portfolio.trades = []

        times = []
        for _ in range(rounds):
            start = time.perf_counter()
            engine._calculate_metrics(benchmark_returns=[0.0, 0.0, 0.0, 0.0])
            times.append(time.perf_counter() - start)

    print(f"Degenerate metric calculation: {min(times)*1e6:.2f} μs min, "
          f"{sum(times)/len(times)*1e6:.2f} μs avg over {rounds} rounds")


if __name__ == "__main__":
    benchmark_backtest_serialization(rounds=100)
    benchmark_comparison_backtest_serialization(rounds=50)
    benchmark_degenerate_metrics(rounds=10000)
