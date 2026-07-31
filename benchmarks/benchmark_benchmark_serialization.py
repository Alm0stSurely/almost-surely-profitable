"""Benchmark for the live equal-weight benchmark state save/load path.

Shows that ``LiveEqualWeightBenchmark.save_state`` writes strict JSON even when
non-finite floats leak into the internal state, and that the load path sanitizes
them back to safe defaults.
"""

import json
import sys
import timeit
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark


def _benchmark(name: str, stmt: str, setup: str = "pass", number: int = 10000, globs=None):
    full_setup = f"import warnings; warnings.simplefilter('error', RuntimeWarning); {setup}"
    elapsed = timeit.timeit(stmt, setup=full_setup, number=number, globals=globs)
    mean_us = elapsed / number * 1e6
    print(f"{name:45s} | {number:6d} runs | {mean_us:8.2f} µs/run")
    return elapsed


if __name__ == "__main__":
    import tempfile

    print("Benchmark: equal-weight benchmark serialization")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmpdir, warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        data_dir = Path(tmpdir) / "data"
        bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(data_dir))
        bm.rebalance({"SPY": 100.0, "GLD": 200.0, "TLT": 50.0})

        state_path = data_dir / "equalweight_benchmark_state.json"

        # 1. Save state repeatedly.
        _benchmark(
            "save_state (clean data)",
            "bm.save_state()",
            globs={"bm": bm},
        )

        # 2. Load state repeatedly.
        _benchmark(
            "load_state on clean file",
            "LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(data_dir))",
            setup=f"data_dir = Path('{tmpdir}') / 'data'",
            globs={"LiveEqualWeightBenchmark": LiveEqualWeightBenchmark, "Path": Path},
        )

        # 3. Save state with non-finite values and verify strict JSON output.
        bm.shares["BAD"] = float("nan")
        bm.cash = float("inf")
        bm.save_state()
        raw = state_path.read_text()
        assert "NaN" not in raw and "Infinity" not in raw
        loaded = json.loads(raw)
        assert loaded["shares"].get("BAD") is None
        assert loaded["cash"] is None

        _benchmark(
            "save_state (non-finite data)",
            "bm.save_state()",
            globs={"bm": bm},
        )

        # 4. Reload from the sanitized file.
        _benchmark(
            "load_state after non-finite file",
            "LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(data_dir))",
            setup=f"data_dir = Path('{tmpdir}') / 'data'",
            globs={"LiveEqualWeightBenchmark": LiveEqualWeightBenchmark, "Path": Path},
        )

    print("-" * 80)
    print("OK - benchmark state serialization is strict and fast.")
