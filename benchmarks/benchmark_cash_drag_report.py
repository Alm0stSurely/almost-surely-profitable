"""
Benchmark cash_drag_report analyze_cash_drag.

Measures the cost of loading many daily result files and classifying them,
including the new validation path for invalid rows.
"""

import json
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.cash_drag_report import analyze_cash_drag


def _write_results(tmp_path: Path, count: int, invalid_ratio: float = 0.0):
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    for i in range(count):
        date_str = f"2026-01-{i + 1:03d}"
        cash = 2000 + (i % 5) * 1000
        total = 10000
        if invalid_ratio and (i / count) < invalid_ratio:
            total = 0 if i % 2 == 0 else float("nan")
        result = {
            "date": date_str,
            "dry_run": False,
            "market_summary": {"assets_analyzed": 32},
            "decision": {"reasoning": "Normal market analysis."},
            "portfolio_after": {"cash": cash, "total_value": total},
            "executed_trades": [{"ticker": "SPY", "action": "buy"}] * (i % 3),
            "cooldown": {
                "status": {
                    "current_vol_regime": "normal",
                    "config": {"current_vol_regime": "normal"},
                }
            },
        }
        with open(results_dir / f"{date_str}.json", "w") as f:
            json.dump(result, f)


def _run(name: str, count: int, iterations: int = 100, invalid_ratio: float = 0.0):
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _write_results(tmp_path, count, invalid_ratio)
        start = time.perf_counter()
        for _ in range(iterations):
            analyze_cash_drag(tmp_path / "results" / "daily")
        elapsed = time.perf_counter() - start
    print(
        f"{name:40s} {count:5d} rows {iterations:5d} iter  "
        f"{elapsed:.3f}s  {elapsed / iterations * 1e3:.3f} ms/iter"
    )
    return elapsed


def main():
    print("=" * 70)
    print("Benchmark: cash_drag_report analyze_cash_drag")
    print("=" * 70)

    _run("100 rows, all valid", 100)
    _run("1_000 rows, all valid", 1_000)
    _run("1_000 rows, 10% invalid", 1_000, invalid_ratio=0.1)
    _run("10_000 rows, all valid", 10_000)


if __name__ == "__main__":
    main()
