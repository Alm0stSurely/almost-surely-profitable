"""
Micro-benchmark for the CPCV fold-score statistics in backtest.cpcv.

`calculate_purged_cv_score` aggregates per-fold metric scores; the dispersion
statistic switched from the population std (ddof=0, numpy default) to the
sample std (ddof=1) to match the repo-wide estimator convention. The delta
must be ns-scale — the estimator change is a convention fix, not a perf
change, and must not move the aggregation cost profile.
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from backtest.cpcv import calculate_purged_cv_score


def _bench(func, n=20_000):
    for _ in range(200):
        func()

    start = time.perf_counter()
    for _ in range(n):
        func()
    elapsed = time.perf_counter() - start
    return elapsed / n * 1e6  # microseconds per call


def _make_results(n_folds):
    """Deterministic per-fold MSEs spread across [0, 1]."""
    rows = {"actual": [], "predicted": [], "fold": [], "combination_id": []}
    for f in range(n_folds):
        for i in range(10):
            rows["actual"].append(float(i))
            rows["predicted"].append(float(i) + (f + 1) / n_folds)
            rows["fold"].append(f)
            rows["combination_id"].append(f)
    return pd.DataFrame(rows)


def mse(y_true, y_pred):
    return float(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2))


def main():
    print("CPCV fold-score statistics benchmark")
    print("=" * 50)

    for n_folds in (5, 45):  # 5-fold K-Fold vs C(10, 2) CPCV
        results = _make_results(n_folds)

        def full():
            return calculate_purged_cv_score(results, mse)

        scores = full()
        us = _bench(full)

        def ddof0():
            return np.std(scores["scores"])

        def ddof1():
            return np.std(scores["scores"], ddof=1)

        us0 = _bench(ddof0)
        us1 = _bench(ddof1)

        print(f"n_folds={n_folds}")
        print(f"  full aggregation        {us:>10.3f} µs/call")
        print(f"  np.std ddof=0 (raw)     {us0:>10.3f} µs/call")
        print(f"  np.std ddof=1 (raw)     {us1:>10.3f} µs/call")


if __name__ == "__main__":
    main()
