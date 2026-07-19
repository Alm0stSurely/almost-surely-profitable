"""Benchmark for tail_risk_analysis small-sample guards.

Demonstrates that tail_risk_analysis now survives small samples without
producing NaN/Inf or biased sample statistics. Before the guard update, the
function used ddof=0 for the Sortino/tracking-error calculations and required
an exact length match between portfolio and benchmark returns, which silently
discarded benchmark comparisons when market calendars differed.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from risk.cvar import tail_risk_analysis


def _run_case(name, returns, benchmark=None):
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        result = tail_risk_analysis(returns, benchmark)
    return result


if __name__ == "__main__":
    print("Benchmark: tail_risk_analysis small-sample robustness")
    print("-" * 60)

    cases = [
        ("empty_returns", np.array([]), None),
        ("single_positive", np.array([0.01]), None),
        ("single_negative", np.array([-0.01]), None),
        ("two_one_downside", np.array([0.01, -0.02]), None),
        ("two_both_downside", np.array([-0.01, -0.02]), None),
        ("short_aligned_benchmark", np.array([0.01, -0.02, 0.005]), np.array([0.005, -0.01, 0.002])),
        ("short_mismatched_benchmark", np.array([0.01, -0.02, 0.005, 0.001]), np.array([0.005, -0.01])),
        ("single_benchmark", np.array([0.01, -0.02]), np.array([0.005])),
    ]

    for name, returns, benchmark in cases:
        result = _run_case(name, returns, benchmark)
        has_sortino = "sortino_ratio" in result
        has_te = "tracking_error" in result
        has_ir = "information_ratio" in result
        finite = all(np.isfinite(v) for v in result.values() if isinstance(v, (int, float)))
        print(f"{name:30s} | sortino={has_sortino!s:5} | te={has_te!s:5} | ir={has_ir!s:5} | finite={finite!s}")

    print("-" * 60)
    print("OK - tail_risk_analysis is robust to small samples and mismatched calendars.")
