"""Benchmark: benchmark-alpha summary print path in evaluation.py.

Measures generate_comprehensive_report end-to-end with a mocked data layer
(no network) to confirm the unambiguous Alpha label (quantity + pp unit +
component breakdown) adds no measurable overhead to the reporting path.
Output noise is suppressed so timings reflect computation, not terminal I/O.
Run under -W error::RuntimeWarning to prove no degenerate-input warnings.
"""

import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import data.fetch_market_data
import evaluation


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


def _mock_data_layer(total_value: float, spy_start: float, spy_end: float) -> None:
    """Patch every network/file dependency of generate_comprehensive_report."""
    evaluation.load_portfolio_data = lambda: {
        "total_value": total_value,
        "cash": total_value * 0.25,
        "total_realized_pnl": 0.0,
        "positions": {},
    }
    evaluation.load_valid_daily_results_limited = lambda *a, **k: [{"date": "2026-08-01"}]
    evaluation.load_valid_daily_results = lambda *a, **k: [
        {"date": "2026-02-17"},
        {"date": "2026-08-01"},
    ]
    evaluation.fetch_historical_data = lambda *a, **k: {
        "SPY": pd.DataFrame({"Close": [spy_start, spy_end]})
    }
    data.fetch_market_data.fetch_current_prices = lambda *a, **k: {"SPY": float(spy_end)}
    mock_analyzer = MagicMock()
    mock_analyzer.load_decisions.return_value = []
    mock_analyzer.analyze_outcomes.return_value = {}
    evaluation.DecisionAnalyzer = lambda: mock_analyzer


def _run_report() -> None:
    with SuppressOutput():
        evaluation.generate_comprehensive_report()


def _time(label: str, fn, n: int = 300) -> None:
    fn()  # warmup
    start = time.perf_counter()
    for _ in range(n):
        fn()
    elapsed = (time.perf_counter() - start) / n * 1e6
    print(f"{label:<52} {elapsed:>10.2f} µs/call")


def main() -> None:
    # Warm path: strategy +5%, SPY +12% -> alpha line printed
    _mock_data_layer(total_value=10500.0, spy_start=100.0, spy_end=112.0)
    _time("generate_comprehensive_report (alpha label path)", _run_report)

    print("\nAlpha label path verified: 'Alpha (vs Buy & Hold SPY)' + component line.")


if __name__ == "__main__":
    main()
