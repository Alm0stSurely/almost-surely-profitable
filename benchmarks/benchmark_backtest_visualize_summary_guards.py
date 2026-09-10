"""
Micro-benchmark for the finite-safe guards in print_summary_table().

The summary table is printed on the backtest comparison path, so the
defensive ``n/a`` fallback must stay on the same order of magnitude as the
happy-path formatting it replaces. Also measures the shared formatting
helpers directly, since both console tables now route through them.
"""

import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.formatting import _fmt_finite, _fmt_pct
from backtest.visualize import print_summary_table


def _make_row(**overrides):
    base = {
        "total_return": 0.0428,
        "annualized_return": 0.0871,
        "sharpe_ratio": 0.76,
        "max_drawdown": -0.0312,
        "num_trades": 12,
        "win_rate": 0.5833,
    }
    base.update(overrides)
    return base


def _bench(func, n=20_000):
    for _ in range(500):
        func()

    start = time.perf_counter()
    for _ in range(n):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def main():
    print("Backtest summary table guards benchmark")
    print("=" * 50)

    finite_results = {"buy_and_hold": _make_row(), "llm": _make_row(total_return=0.10)}
    nan_results = {
        "degenerate": _make_row(
            total_return=float("nan"),
            annualized_return=None,
            sharpe_ratio=float("inf"),
            max_drawdown=float("-inf"),
            num_trades=None,
            win_rate=float("nan"),
        )
    }

    def _run(results):
        with redirect_stdout(StringIO()):
            print_summary_table(results)

    table_cases = [
        ("table finite (2 strategies)", lambda: _run(finite_results)),
        ("table non-finite (1 strategy)", lambda: _run(nan_results)),
    ]
    helper_cases = [
        ("_fmt_pct finite", lambda: _fmt_pct(0.0428, ">9.2f")),
        ("_fmt_pct NaN", lambda: _fmt_pct(float("nan"), ">9.2f")),
        ("_fmt_pct None", lambda: _fmt_pct(None, ">9.2f")),
        ("_fmt_finite finite", lambda: _fmt_finite(0.76, ">7.2f")),
        ("_fmt_finite inf", lambda: _fmt_finite(float("inf"), ">7.2f")),
        ("_fmt_finite int", lambda: _fmt_finite(12, ">7")),
    ]

    for label, func in table_cases + helper_cases:
        us = _bench(func)
        print(f"  {label:<32} {us:8.2f} µs/call")


if __name__ == "__main__":
    main()
