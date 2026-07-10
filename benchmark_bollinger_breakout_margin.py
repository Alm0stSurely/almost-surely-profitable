"""Benchmark the Bollinger breakout margin threshold.

Compares the false-positive rate of the old threshold-less rule (min_pct=0.0)
against the new configurable rule (min_pct=1.0) on a synthetic sample of
breakout margins around the Bollinger upper band, and measures the runtime
overhead of the margin check.
"""

import sys
import time
from pathlib import Path
from random import Random

sys.path.insert(0, str(Path(__file__).parent / "src"))

from monitor import _breakout_margin, _is_significant_breakout


def main():
    rng = Random(42)
    band = 100.0
    n = 1_000

    # Synthetic margins from 0.0% to 2.0% beyond the band.
    margins = [rng.uniform(0.0, 2.0) for _ in range(n)]
    prices = [band * (1 + m / 100.0) for m in margins]

    old_alerts = sum(1 for p in prices if _is_significant_breakout(p, band, 0.0))
    new_alerts = sum(1 for p in prices if _is_significant_breakout(p, band, 1.0))

    start = time.perf_counter()
    for _ in range(100_000):
        _is_significant_breakout(101.5, band, 1.0)
    elapsed = time.perf_counter() - start

    print("Bollinger Breakout Margin Threshold Benchmark")
    print("=" * 50)
    print(f"Sample size:                {n}")
    print(f"Old rule (no min margin):   {old_alerts} alerts")
    print(f"New rule (min 1.0%):        {new_alerts} alerts")
    if old_alerts:
        reduction_pct = (1 - new_alerts / old_alerts) * 100
        print(f"False-positive reduction:   {old_alerts - new_alerts} alerts ({reduction_pct:.1f}%)")
    print(f"Decision time (100k evals): {elapsed * 1000:.3f} ms")
    print(f"Mean time per decision:     {elapsed / 100_000 * 1e6:.3f} µs")
    print(f"Mean absolute margin:       {sum(_breakout_margin(p, band) for p in prices) / n:.3f}%")


if __name__ == "__main__":
    main()
