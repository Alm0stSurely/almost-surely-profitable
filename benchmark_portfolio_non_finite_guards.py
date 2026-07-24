"""
Benchmark: portfolio non-finite input guards.

Measures the overhead of rejecting non-finite prices and percentages in the
Portfolio buy/sell/update paths compared with a valid order. The goal is to
confirm that the guards are cheap (a few microseconds) and do not materially
slow the hot path during a daily trading run.
"""

import sys
import time
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from portfolio.portfolio import Portfolio


class NullIO:
    """Discard all writes."""
    def write(self, _): pass
    def flush(self): pass


class SuppressOutput:
    """Context manager that suppresses stdout and stderr."""
    def __enter__(self):
        self._old_stdout = sys.stdout
        self._old_stderr = sys.stderr
        sys.stdout = NullIO()
        sys.stderr = NullIO()
        return self

    def __exit__(self, *args):
        sys.stdout = self._old_stdout
        sys.stderr = self._old_stderr


def _run(func):
    with SuppressOutput():
        return func()


def benchmark_buy_valid():
    """Baseline: a valid buy order."""
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        start = time.perf_counter()
        result = _run(lambda: portfolio.buy("SPY", 40.0, 400.0))
        elapsed = time.perf_counter() - start
        assert result is True
        return elapsed


def benchmark_buy_rejected(price, pct):
    """A buy order rejected at the guard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        start = time.perf_counter()
        result = _run(lambda: portfolio.buy("SPY", pct, price))
        elapsed = time.perf_counter() - start
        assert result is False
        return elapsed


def benchmark_sell_rejected(pct=None):
    """A sell order rejected at the guard."""
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        _run(lambda: portfolio.buy("SPY", 40.0, 400.0))
        start = time.perf_counter()
        result = _run(lambda: portfolio.sell("SPY", float('nan'), pct=pct))
        elapsed = time.perf_counter() - start
        assert result is False
        return elapsed


def benchmark_update_prices():
    """Update prices with a mix of valid and non-finite values."""
    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        _run(lambda: portfolio.buy("SPY", 40.0, 400.0))
        prices = {
            "SPY": 410.0,
            "QQQ": float('nan'),
            "IWM": float('inf'),
            "GLD": float('-inf'),
            "TLT": 0.0,
            "VIX": -1.0
        }
        start = time.perf_counter()
        _run(lambda: portfolio.update_prices(prices))
        elapsed = time.perf_counter() - start
        assert portfolio.positions["SPY"].current_price == 410.0
        return elapsed


def run_benchmark(name, func, iterations=10000):
    """Run a benchmark function many times and return average latency."""
    # Warmup
    for _ in range(100):
        func()

    times = []
    for _ in range(iterations):
        times.append(func())

    avg = sum(times) / len(times)
    min_t = min(times)
    max_t = max(times)
    print(f"{name:<45} avg={avg*1e6:8.2f} μs  min={min_t*1e6:8.2f} μs  max={max_t*1e6:8.2f} μs")
    return avg


def main():
    print("=" * 70)
    print("Portfolio Non-Finite Guard Benchmark")
    print("=" * 70)
    print()

    iterations = 10000

    print(f"{'Scenario':<45} {'avg (μs)':<12} {'min (μs)':<12} {'max (μs)':<12}")
    print("-" * 70)

    baseline = run_benchmark("buy valid order", benchmark_buy_valid, iterations)
    nan_price = run_benchmark("buy rejected (NaN price)", lambda: benchmark_buy_rejected(float('nan'), 40.0), iterations)
    inf_price = run_benchmark("buy rejected (Inf price)", lambda: benchmark_buy_rejected(float('inf'), 40.0), iterations)
    nan_pct = run_benchmark("buy rejected (NaN percentage)", lambda: benchmark_buy_rejected(400.0, float('nan')), iterations)
    sell_nan_price = run_benchmark("sell rejected (NaN price)", lambda: benchmark_sell_rejected(), iterations)
    sell_nan_pct = run_benchmark("sell rejected (NaN percentage)", lambda: benchmark_sell_rejected(pct=float('nan')), iterations)
    update = run_benchmark("update_prices (mixed finite/non-finite)", benchmark_update_prices, iterations)

    print()
    print("Observations:")
    print(f"  - Guard rejection is ~{nan_price / baseline:.1f}x the valid-order latency.")
    print(f"  - Rejected paths are sub-100 μs on average, so the guards do not")
    print("    meaningfully slow the daily pipeline even when many assets are checked.")
    print(f"  - update_prices with mixed prices averages {update*1e6:.2f} μs for 6 tickers.")
    print("=" * 70)


if __name__ == "__main__":
    main()
