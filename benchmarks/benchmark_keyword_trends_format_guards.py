"""Micro-benchmark for keyword_trends formatting guards."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.keyword_trends import _safe_pct_str, _safe_signed_str, format_report


def _make_rates(n_weeks=52, concepts=("loss aversion", "CVaR", "tail risk")):
    return {
        f"2026-W{i:02d}": {
            **{"_n": 5},
            **{c: (i + j) % 100 for j, c in enumerate(concepts)},
        }
        for i in range(1, n_weeks + 1)
    }


def _run(func):
    start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - start) * 1e6
    return result, elapsed


def main():
    rates = _make_rates()
    highlight = ["loss aversion", "CVaR", "tail risk"]

    _, us_full = _run(lambda: format_report(rates, highlight_concepts=highlight, window=4))
    _, us_finite = _run(lambda: _safe_pct_str(12.34, width=9, precision=1))
    _, us_nan = _run(lambda: _safe_pct_str(float("nan"), width=9, precision=1))
    _, us_signed = _run(lambda: _safe_signed_str(0.42, width=9, precision=2))
    _, us_signed_nan = _run(lambda: _safe_signed_str(float("nan"), width=9, precision=2))

    print("| Helper | Input | µs/call |")
    print("|---|---|---|")
    print(f"| `format_report` (52 weeks, 3 concepts) | full report | {us_full:.3f} |")
    print(f"| `_safe_pct_str` | finite | {us_finite:.3f} |")
    print(f"| `_safe_pct_str` | NaN | {us_nan:.3f} |")
    print(f"| `_safe_signed_str` | finite | {us_signed:.3f} |")
    print(f"| `_safe_signed_str` | NaN | {us_signed_nan:.3f} |")


if __name__ == "__main__":
    main()
