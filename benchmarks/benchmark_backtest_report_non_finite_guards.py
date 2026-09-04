"""
Micro-benchmark for the non-finite guards in print_backtest_report().

The report is printed to the console after every backtest run, so the
defensive ``n/a`` fallback must stay on the same order of magnitude as the
happy-path formatting it replaces.
"""

import sys
import time
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.backtest import _fmt_finite, _fmt_pct, print_backtest_report


def _make_result(**overrides):
    base = {
        "start_date": "2026-01-01",
        "end_date": "2026-06-30",
        "initial_capital": 10000.0,
        "final_value": 10427.55,
        "total_return": 0.0428,
        "annualized_return": 0.0871,
        "volatility": 0.1142,
        "max_drawdown": 0.0312,
        "sharpe_ratio": 0.76,
        "sortino_ratio": 1.12,
        "calmar_ratio": 2.79,
        "omega_ratio": 1.31,
        "num_trades": 12,
        "win_rate": 0.5833,
        "profit_factor": 1.42,
        "beta": 0.88,
        "alpha": 0.0111,
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
    print("Backtest report non-finite guards benchmark")
    print("=" * 50)

    finite_result = _make_result()
    nan_result = _make_result(
        final_value=float("nan"),
        total_return=float("nan"),
        annualized_return=float("inf"),
        volatility=float("nan"),
        max_drawdown=float("-inf"),
        sharpe_ratio=float("nan"),
        sortino_ratio=float("nan"),
        calmar_ratio=float("nan"),
        omega_ratio=float("nan"),
        win_rate=float("nan"),
        profit_factor=float("inf"),
        beta=float("nan"),
        alpha=float("nan"),
    )

    def render(result):
        with redirect_stdout(StringIO()):
            print_backtest_report(result, "bench")

    cases = [
        ("report finite", lambda: render(finite_result)),
        ("report non-finite", lambda: render(nan_result)),
        ("_fmt_finite finite", lambda: _fmt_finite(10427.55, ",.2f")),
        ("_fmt_finite NaN", lambda: _fmt_finite(float("nan"), ",.2f")),
        ("_fmt_pct finite", lambda: _fmt_pct(0.0428, ">8.2f")),
        ("_fmt_pct inf", lambda: _fmt_pct(float("inf"), ">8.2f")),
        ("_fmt_pct None", lambda: _fmt_pct(None, ">8.2f")),
    ]

    for label, func in cases:
        us = _bench(func)
        print(f"{label:<28} {us:>10.3f} µs/call")


if __name__ == "__main__":
    main()
