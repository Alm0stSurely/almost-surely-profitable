"""
Benchmark for the buy() IEEE-754 round-trip snap fix (portfolio).

Measures:
1. End-to-end buy() cost with the fix (I/O-dominated: save_state + save_trade).
2. Micro-cost of the snap-check arithmetic itself (multiply + compare).
3. Empirical overshoot rate: fraction of (cash, price) pairs whose
   cash/price*price round-trip exceeds the exact budget by 1 ulp.
"""
import math
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class NullIO:
    def write(self, _):
        pass

    def flush(self):
        pass


class SuppressOutput:
    def __enter__(self):
        import os
        self._old = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = NullIO()
        self._old_stdout_fd = os.dup(1)
        self._null_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._null_fd, 1)
        return self

    def __exit__(self, *args):
        import os
        os.dup2(self._old_stdout_fd, 1)
        os.close(self._old_stdout_fd)
        os.close(self._null_fd)
        sys.stdout, sys.stderr = self._old


def bench_buy_end_to_end():
    from portfolio.portfolio import Portfolio

    tmpdir = tempfile.mkdtemp()
    n = 2000
    # Warmup
    for i in range(50):
        p = Portfolio(state_file=f"w{i}.json", trades_file=f"wt{i}.json", data_dir=tmpdir)
        p.cash = 10000.0
        with SuppressOutput():
            p.buy("SPY", 10.0, 400.0 + i)

    start = time.perf_counter()
    for i in range(n):
        p = Portfolio(state_file=f"s{i}.json", trades_file=f"t{i}.json", data_dir=tmpdir)
        p.cash = 10000.0
        with SuppressOutput():
            p.buy("SPY", 10.0, 400.0 + (i % 100))
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # µs per buy (incl. persistence)


def bench_snap_check_micro():
    """Cost of the added multiply+compare in the common (no-snap) path."""
    cash, price = 10000.0, 757.42
    n = 1_000_000
    # Warmup
    for _ in range(10_000):
        ctu = cash * 1.0
        q = ctu / price
        t = q * price
        _ = t > ctu
    start = time.perf_counter()
    for _ in range(n):
        ctu = cash * 1.0
        q = ctu / price
        t = q * price
        _ = t > ctu
    return (time.perf_counter() - start) / n * 1e9  # ns per round-trip check


def measure_overshoot_rate():
    import random
    random.seed(42)
    n = 500_000
    over = 0
    for _ in range(n):
        cash = random.uniform(0.01, 1e6)
        price = random.uniform(0.01, 1e5)
        q = cash / price
        if q * price > cash:
            over += 1
    return over / n * 100


if __name__ == "__main__":
    print("=== buy() round-trip snap fix benchmark ===\n")
    t_buy = bench_buy_end_to_end()
    print(f"buy() end-to-end (with fix, incl. JSON persistence): {t_buy:.1f} µs")
    t_snap = bench_snap_check_micro()
    print(f"snap-check arithmetic (multiply + compare):          {t_snap:.1f} ns")
    rate = measure_overshoot_rate()
    print(f"empirical ULP overshoot rate (random pairs):         {rate:.2f}% of full-cash buys")
    print("\nNote: end-to-end buy() is dominated by save_state/save_trade I/O;")
    print("the added guard is one IEEE multiply + compare (~ns scale).")
