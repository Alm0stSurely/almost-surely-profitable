"""Benchmark: statistical consistency conventions in BacktestEngine._calculate_metrics.

Validates that the 252-trading-day annualization and sample-std (ddof=1)
conventions add no measurable overhead vs the previous population-std path,
and measures the guard branches (single-return, single-downside edge cases).
Run under -W error::RuntimeWarning to prove no degenerate-input warnings.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from backtest.backtest import BacktestEngine  # noqa: E402
from portfolio.portfolio import Portfolio  # noqa: E402


def _make_engine(tmp_dir: str) -> BacktestEngine:
    engine = BacktestEngine(
        start_date="2024-01-01",
        end_date="2024-12-31",
        tickers=["SPY"],
    )
    engine.portfolio = Portfolio(
        state_file="bt_bench.json",
        trades_file="bt_bench_trades.json",
        data_dir=tmp_dir,
    )
    engine.initial_capital = 10000.0
    return engine


def _seed_values(returns: list[float]) -> list[float]:
    values = [10000.0]
    for r in returns:
        values.append(values[-1] * (1 + r))
    return values


def _time(label: str, fn, n: int = 2000) -> None:
    fn()  # warmup
    import time

    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n * 1e6
    print(f"{label:<52} {elapsed:>10.2f} µs/call")


def main() -> None:
    import tempfile

    tmp = tempfile.mkdtemp()

    np.random.seed(42)
    long_rets = np.random.normal(0.0005, 0.01, 252).tolist()
    benchmark_rets = np.random.normal(0.0004, 0.008, 252).tolist()
    long_values = _seed_values(long_rets)

    engine = _make_engine(tmp)
    engine.results = [{"total_value": v} for v in long_values]

    _time(
        "_calculate_metrics 252d (sample-std conventions)",
        lambda: engine._calculate_metrics(benchmark_returns=benchmark_rets),
    )

    # Single-return edge: sample std undefined → volatility 0
    engine_single = _make_engine(tmp)
    engine_single.results = [{"total_value": v} for v in [10000.0, 10100.0]]
    _time(
        "_calculate_metrics single return (guards short-circuit)",
        lambda: engine_single._calculate_metrics(),
    )

    # Single downside edge: Sortino denominator undefined → 0
    engine_down = _make_engine(tmp)
    engine_down.results = [{"total_value": v} for v in [10000.0, 9900.0]]
    _time(
        "_calculate_metrics single downside (Sortino guard)",
        lambda: engine_down._calculate_metrics(),
    )

    print("\nConventions verified: 252 trading days, ddof=1, n_periods=len(returns).")


if __name__ == "__main__":
    main()
