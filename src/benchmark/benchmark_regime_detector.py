"""
Benchmark for src/analysis/regime_detector.py

Measures the wall-clock time of RegimeDetector.analyze() across realistic
history lengths and asset counts. Useful for spotting regressions after
adding input-validation and finite-value guards.
"""

import time

import numpy as np
import pandas as pd

from analysis.regime_detector import RegimeDetector


def _make_prices(n_days: int, n_assets: int, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
    returns = rng.normal(0.0, 0.015, size=(n_days, n_assets))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    columns = [f"ASSET_{i}" for i in range(n_assets)]
    return pd.DataFrame(prices, index=dates, columns=columns)


def benchmark_analyze(n_days: int, n_assets: int, n_runs: int = 10) -> dict:
    detector = RegimeDetector()
    prices = _make_prices(n_days, n_assets)

    # Warm-up run (exclude from timing)
    detector.analyze(prices)

    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        state = detector.analyze(prices)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    return {
        "n_days": n_days,
        "n_assets": n_assets,
        "n_runs": n_runs,
        "mean_s": float(np.mean(times)),
        "median_s": float(np.median(times)),
        "min_s": float(np.min(times)),
        "max_s": float(np.max(times)),
        "std_s": float(np.std(times)),
        "state": {
            "volatility_regime": state.volatility_regime,
            "trend_regime": state.trend_regime,
            "correlation_regime": state.correlation_regime,
        },
    }


def main() -> None:
    configs = [
        (120, 2),
        (252, 5),
        (504, 10),
        (1008, 20),
    ]

    print("RegimeDetector.analyze() benchmark")
    print("=" * 70)
    for n_days, n_assets in configs:
        result = benchmark_analyze(n_days, n_assets)
        print(
            f"{result['n_days']:4d} days x {result['n_assets']:2d} assets | "
            f"mean={result['mean_s']*1e3:6.2f} ms | "
            f"median={result['median_s']*1e3:6.2f} ms | "
            f"min={result['min_s']*1e3:6.2f} ms | "
            f"max={result['max_s']*1e3:6.2f} ms | "
            f"vol={result['state']['volatility_regime']} "
            f"trend={result['state']['trend_regime']} "
            f"corr={result['state']['correlation_regime']}"
        )


if __name__ == "__main__":
    main()
