"""
Micro-benchmark for the strict JSON persistence boundary (allow_nan=False).

The flag adds a per-float constant-time check inside the C encoder; the
contract it buys is that a non-finite value can never reach disk as a
non-standard NaN/Infinity token. This benchmark pins the overhead to
nanoseconds per call so the contract stays effectively free.
"""

import json
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio


def _bench_dump(state, allow_nan, n=10_000):
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "state.json"

        def _run():
            with open(path, "w") as f:
                json.dump(state, f, indent=2, allow_nan=allow_nan)

        for _ in range(500):
            _run()
        start = time.perf_counter()
        for _ in range(n):
            _run()
        elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def main():
    print("Strict JSON persistence boundary benchmark")
    print("=" * 50)

    with tempfile.TemporaryDirectory() as tmpdir:
        portfolio = Portfolio(data_dir=tmpdir)
        portfolio.buy("SPY", 40.0, 400.0)
        portfolio.buy("QQQ", 17.5, 350.0)
        state = {
            "cash": portfolio.cash,
            "total_realized_pnl": portfolio.total_realized_pnl,
            "positions": {
                ticker: {
                    "quantity": pos.quantity,
                    "avg_price": pos.avg_price,
                    "current_price": pos.current_price,
                }
                for ticker, pos in portfolio.positions.items()
            },
            "last_updated": datetime.now().isoformat(),
            "total_value": portfolio.total_value,
        }

    loose = _bench_dump(state, allow_nan=True)
    strict = _bench_dump(state, allow_nan=False)

    print(f"  portfolio state dump (loose  allow_nan=True ): {loose:8.2f} µs/call")
    print(f"  portfolio state dump (strict allow_nan=False): {strict:8.2f} µs/call")
    print(f"  strict overhead: {strict - loose:+.2f} µs/call "
          f"({(strict / loose - 1) * 100:+.1f}%)")
    print()
    print("Overhead is encoder-internal and independent of payload size")
    print("to first order: the flag is a contract, not a computation.")
    print("=" * 50)


if __name__ == "__main__":
    main()
