"""Timing harness for the benchmark load_state cash-refusal guard (PR #82).

Honest no-claim benchmark, per the PR #75/#80/#81 doctrine: the change adds
one predicate call on a cold (once-per-process) construction path, so a
timing delta is not the point — the functional gate
repro_benchmark_load_state_cash_refusal.py is the real gate. This harness
exists so the PR reports observed numbers instead of an unverified
"no regression" assertion.

Methodology: N-round healthy-state construction, best-of-R repetitions,
temp dir, no output noise. A/B by stashing/restoring
src/benchmark/__init__.py.

Run: python benchmarks/benchmark_benchmark_load_state_cash_guard.py
"""

import contextlib
import io
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark

N_ROUNDS = 200
REPS = 5

STATE = {
    "shares": {f"T{i}": 1.0 + i for i in range(20)},
    "cash": 2295.17,
    "start_date": "2026-07-31",
    "last_rebalanced": "2026-07-31T10:00:00",
}


def bench() -> float:
    """Return best wall time (ms) for N_ROUNDS fresh constructions."""
    tmp = Path(tempfile.mkdtemp(prefix="bench-bench-cash-guard-"))
    try:
        data_dir = tmp / "data"
        data_dir.mkdir()
        with open(data_dir / "equalweight_benchmark_state.json", "w") as f:
            json.dump(STATE, f)
        best = float("inf")
        for _ in range(REPS):
            start = time.perf_counter()
            # Construction logs nothing on the healthy path; wrap anyway so a
            # loud guard cannot flood the report (PR #64 lesson).
            with contextlib.redirect_stderr(io.StringIO()):
                for _ in range(N_ROUNDS):
                    LiveEqualWeightBenchmark(
                        initial_capital=10000.0, data_dir=str(data_dir)
                    )
            best = min(best, (time.perf_counter() - start) / N_ROUNDS * 1000)
        return best
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    print(f"healthy-state construction, best-of-{REPS}, n={N_ROUNDS} per rep")
    post = bench()
    print(f"current working tree: {post:.4f} ms/load")
    print("\nA/B against origin/dev requires stash; numbers above are post-fix.")
    print("Cold-path predicate: one _is_valid_cash call per construction.")


if __name__ == "__main__":
    main()
