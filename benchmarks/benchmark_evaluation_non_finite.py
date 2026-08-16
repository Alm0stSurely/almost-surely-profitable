"""
Benchmark the non-finite guards in evaluation.py.

Compares runtime before/after guarding against zero and non-finite portfolio
values in the comprehensive report generator.
"""

import json
import logging
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from evaluation import generate_comprehensive_report

# Silence noisy yfinance/network logging so benchmark output stays readable.
logging.getLogger("data.fetch_market_data").setLevel(logging.CRITICAL)

REPO_ROOT = Path(__file__).parent.parent


class NullIO:
    def write(self, _):
        pass

    def flush(self):
        pass


class SuppressOutput:
    def __enter__(self):
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = NullIO()
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr


def _setup_tmp_portfolio(tmp_path: Path, total_value):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    portfolio = {
        "cash": total_value / 2 if total_value and total_value > 0 else 0.0,
        "total_value": total_value,
        "total_realized_pnl": 0.0,
        "positions": {},
    }
    with open(data_dir / "portfolio_state.json", "w") as f:
        json.dump(portfolio, f)


def _setup_tmp_results(tmp_path: Path, values):
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    for i, val in enumerate(values):
        date_str = f"2026-01-{i + 1:02d}"
        with open(results_dir / f"{date_str}.json", "w") as f:
            json.dump(
                {
                    "date": date_str,
                    "portfolio_after": {
                        "total_value": val,
                        "cash": 0.0,
                        "num_positions": 0,
                    },
                },
                f,
            )


def _run(name: str, tmp_path: Path, iterations: int = 100):
    """Run the report generator repeatedly and time it."""
    fake_benchmark = {"SPY": pd.DataFrame({"Close": [100.0, 105.0]})}

    # Patch fetchers so the benchmark does not hit the network.
    with (
        patch(
            "data.fetch_market_data.fetch_current_prices",
            return_value={"SPY": 450.0},
        ),
        patch(
            "evaluation.fetch_historical_data",
            return_value=fake_benchmark,
        ),
    ):
        start = time.perf_counter()
        for _ in range(iterations):
            with SuppressOutput():
                generate_comprehensive_report()
        elapsed = time.perf_counter() - start

    print(
        f"{name:40s} {iterations:6d} iter  {elapsed:.3f}s  {elapsed / iterations * 1e3:.3f} ms/iter"
    )
    return elapsed


def main():
    print("=" * 70)
    print("Benchmark: evaluation.py non-finite guards")
    print("=" * 70)

    # Scenario 1: healthy portfolio
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _setup_tmp_portfolio(tmp_path, 10000.0)
        _setup_tmp_results(tmp_path, [10000, 10100, 10200, 10150, 10300])
        _run("Healthy portfolio", tmp_path)

    # Scenario 2: zero total value
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _setup_tmp_portfolio(tmp_path, 0.0)
        _run("Zero total value", tmp_path)

    # Scenario 3: NaN in daily returns
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _setup_tmp_portfolio(tmp_path, 10000.0)
        _setup_tmp_results(tmp_path, [10000, float("nan"), 10100])
        _run("NaN daily returns", tmp_path)

    # Scenario 4: infinite total value
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _setup_tmp_portfolio(tmp_path, float("inf"))
        _run("Infinite total value", tmp_path)


if __name__ == "__main__":
    main()
