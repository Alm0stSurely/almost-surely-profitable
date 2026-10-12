"""Timing harness for the load_state monetary-base guard (PR #81).

Honest no-claim benchmark, per the PR #75/#80 doctrine: the change adds
two predicate calls on a cold (once-per-process) path, so a timing delta
is not the point — the functional gate repro_portfolio_load_state_monetary_guard.py
is the real gate. This harness exists so the PR reports observed numbers
instead of an unverified "no regression" assertion.

Methodology: N-round healthy-state load, best-of-R repetitions, temp dir,
no output noise. A/B by stashing/restoring src/portfolio/portfolio.py.

Run: .venv/bin/python benchmarks/benchmark_portfolio_load_state_monetary_guard.py
"""

import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio

N_ROUNDS = 200
REPS = 5

STATE = {
    "cash": 2295.17,
    "total_realized_pnl": -112.50,
    "positions": {
        f"T{i}": {"quantity": 1.0 + i, "avg_price": 100.0, "current_price": 101.0}
        for i in range(10)
    },
}


def bench() -> float:
    """Return best wall time (ms) for N_ROUNDS fresh Portfolio loads."""
    tmp = Path(tempfile.mkdtemp(prefix="bench-monetary-guard-"))
    try:
        data_dir = tmp / "data"
        data_dir.mkdir()
        with open(data_dir / "portfolio_state.json", "w") as f:
            json.dump(STATE, f)
        best = float("inf")
        for _ in range(REPS):
            start = time.perf_counter()
            for _ in range(N_ROUNDS):
                Portfolio(data_dir=str(data_dir))
            best = min(best, (time.perf_counter() - start) / N_ROUNDS * 1000)
        return best
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    import contextlib
    import io

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        best = bench()
    print(f"Healthy load (10 positions), {N_ROUNDS} rounds x {REPS} reps, "
          f"best: {best:.4f} ms/load")
    print("No perf claim: two predicate calls on a once-per-process cold "
          "path; the functional gate is the embedded benchmark for this PR.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
