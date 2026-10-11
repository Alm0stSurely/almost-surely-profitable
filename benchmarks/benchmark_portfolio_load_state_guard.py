"""Timing run for the Portfolio.load_state ingress guard (informational).

The guard adds three predicate calls (isinstance/isfinite comparisons) per
position field on a cold, disk-dominated path — no perf claim is made. This
run exists to demonstrate the healthy-path overhead is not measurable above
noise, and to serve as an A/B harness (stash the source file to compare).

The functional gate is benchmarks/repro_portfolio_load_state_guard.py.
"""

import json
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio.portfolio import Portfolio

N_POSITIONS = 100
ROUNDS = 5


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        positions = {
            f"T{i:03d}": {"quantity": 1.0 + i, "avg_price": 100.0 + i, "current_price": 101.0 + i}
            for i in range(N_POSITIONS)
        }
        Path(tmpdir, "portfolio_state.json").write_text(
            json.dumps({"cash": 10000.0, "total_realized_pnl": 0.0, "positions": positions})
        )

        samples = []
        for _ in range(ROUNDS):
            start = time.perf_counter()
            portfolio = Portfolio(data_dir=tmpdir)
            samples.append((time.perf_counter() - start) * 1e3)
            assert len(portfolio.positions) == N_POSITIONS

        # warm the disk cache once and re-time: isolates parse+validate from I/O
        samples_cached = []
        for _ in range(ROUNDS):
            start = time.perf_counter()
            portfolio = Portfolio(data_dir=tmpdir)
            samples_cached.append((time.perf_counter() - start) * 1e3)
            assert len(portfolio.positions) == N_POSITIONS

    print(f"load_state healthy path, {N_POSITIONS} positions (best of {ROUNDS}):")
    print(f"  cold-ish : {min(samples):8.3f} ms  (median {statistics.median(samples):8.3f})")
    print(f"  cached   : {min(samples_cached):8.3f} ms  (median {statistics.median(samples_cached):8.3f})")
    print("No perf claim: cold path dominated by disk read + json parse; the")
    print("guard cost is three predicate calls per field (isinstance/isfinite).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
