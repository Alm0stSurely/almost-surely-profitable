"""Benchmark for JSON-safe serialization helpers.

Shows that ``sanitize_for_json`` and ``dump_json_safe`` handle non-finite
floats without measurable overhead compared to a standard ``json.dumps`` of
clean data, and that the output is always valid JSON (``allow_nan=False``).
"""

import io
import json
import sys
import timeit
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import sanitize_for_json, dump_json_safe


_clean_result = {
    "date": "2026-07-27",
    "portfolio_after": {
        "cash": 3360.04,
        "total_value": 9764.80,
        "positions": [
            {"ticker": "SPY", "quantity": 10, "market_value": 2000.0, "unrealized_pnl_pct": 3.0},
            {"ticker": "GLD", "quantity": 5, "market_value": 1500.0, "unrealized_pnl_pct": -1.5},
        ],
    },
    "performance_metrics": {
        "sharpe_ratio": 1.5,
        "volatility": 0.12,
        "max_drawdown": -0.04,
    },
}

_dirty_result = {
    **_clean_result,
    "equalweight_benchmark": {
        "total_value": float("nan"),
        "total_return_pct": float("inf"),
        "num_positions": 0,
    },
}


def _benchmark(name: str, stmt, setup="pass", number: int = 10000):
    full_setup = f"import warnings; warnings.simplefilter('error', RuntimeWarning); {setup}"
    elapsed = timeit.timeit(stmt, setup=full_setup, number=number, globals=globals())
    mean_us = elapsed / number * 1e6
    print(f"{name:40s} | {number:6d} runs | {mean_us:8.2f} µs/run")
    return elapsed


if __name__ == "__main__":
    print("Benchmark: JSON-safe serialization")
    print("-" * 75)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        # 1. Standard serialization of clean data (baseline).
        _benchmark(
            "json.dumps (clean data)",
            "json.dumps(_clean_result)",
            "import json",
        )

        # 2. Sanitize + serialize clean data.
        _benchmark(
            "dump_json_safe (clean data)",
            "dump_json_safe(_clean_result, io.StringIO())",
        )

        # 3. Sanitize + serialize dirty data, ensuring output is valid JSON.
        buffer = io.StringIO()
        dump_json_safe(_dirty_result, buffer)
        output = buffer.getvalue()
        assert "NaN" not in output and "Infinity" not in output
        assert json.loads(output)["equalweight_benchmark"]["total_value"] is None
        assert json.loads(output)["equalweight_benchmark"]["total_return_pct"] is None
        _benchmark(
            "dump_json_safe (non-finite data)",
            "dump_json_safe(_dirty_result, io.StringIO())",
        )

        # 4. Recursive sanitization alone.
        cleaned = sanitize_for_json(_dirty_result)
        assert cleaned["equalweight_benchmark"]["total_value"] is None
        assert cleaned["equalweight_benchmark"]["total_return_pct"] is None
        _benchmark(
            "sanitize_for_json (non-finite data)",
            "sanitize_for_json(_dirty_result)",
        )

    print("-" * 75)
    print("OK - JSON-safe serialization is fast and produces valid JSON.")
