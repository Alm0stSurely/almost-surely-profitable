#!/usr/bin/env python3
"""
Benchmark: DecisionMemory date-window precision and scaling.

The summary window in DecisionMemory.get_decision_summary() compares a full
datetime (datetime.now() - timedelta(days=N)) against record dates parsed as
midnight. This can exclude a record dated exactly N days ago when the current
clock has advanced past midnight, silently shrinking the reported window and
dropping keys such as best_trade / worst_trade.

This benchmark:
1. Verifies that the date-only comparison keeps boundary records in the window.
2. Measures summary latency as the decision memory grows.
"""

import json
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.decision_memory import DecisionMemory, DecisionRecord
from analysis import decision_memory as dm


def make_record(date: str, pnl_pct: float = 1.0) -> DecisionRecord:
    return DecisionRecord(
        date=date,
        ticker="AI.PA",
        action="buy",
        quantity=1.0,
        price=100.0,
        portfolio_value_before=10000.0,
        portfolio_value_after=10000.0,
        pnl_pct=pnl_pct,
        holding_period_days=1,
    )


def build_memory(path: Path, n_records: int) -> None:
    """Write a synthetic decision memory file with n_records spread over 60 days."""
    base = datetime(2026, 7, 15)
    records = []
    for i in range(n_records):
        day_offset = i % 60
        date = (base - timedelta(days=day_offset)).strftime("%Y-%m-%d")
        records.append(make_record(date, pnl_pct=1.0 + (i % 5) * 0.1))
    path.write_text(json.dumps([r.to_dict() for r in records]))


def benchmark_boundary_inclusion():
    print("=" * 70)
    print("BENCHMARK: 30-day window boundary inclusion")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "decision_memory.json"
        records = [
            make_record(date="2026-06-14", pnl_pct=3.0),   # 31 days ago -> excluded
            make_record(date="2026-06-15", pnl_pct=5.0),   # exactly 30 days ago -> included
            make_record(date="2026-07-14", pnl_pct=2.0),   # inside -> included
        ]
        path.write_text(json.dumps([r.to_dict() for r in records]))
        mem = DecisionMemory(memory_file=str(path))

        frozen_now = datetime(2026, 7, 15, 12, 0, 0)
        with patch.object(dm, "datetime", wraps=datetime) as mock_dt:
            mock_dt.now.return_value = frozen_now
            summary = mem.get_decision_summary(days=30)

        print(f"Frozen now:       {frozen_now}")
        print(f"30-day cutoff:    {(frozen_now - timedelta(days=30)).date()}")
        print(f"Total decisions:  {summary['total_decisions']}")
        print(f"Completed trades: {summary['completed_trades']}")
        print(f"Best trade:       {summary['best_trade']}")
        print(f"Worst trade:      {summary['worst_trade']}")
        assert summary["total_decisions"] == 2, "expected boundary + inside records"
        assert summary["best_trade"] == 5.0, "best trade should be the boundary record"
        print("\nPASS: boundary record dated 2026-06-15 is correctly included; older record excluded.")


def benchmark_scaling():
    print()
    print("=" * 70)
    print("BENCHMARK: get_decision_summary scaling")
    print("=" * 70)
    print(f"{'Records':>10}  {'Time (ms)':>12}  {'Decisions':>10}")
    print("-" * 38)

    for n in (100, 500, 1000, 5000, 10000):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "decision_memory.json"
            build_memory(path, n)
            mem = DecisionMemory(memory_file=str(path))

            start = time.perf_counter()
            for _ in range(100):
                mem.get_decision_summary(days=30)
            elapsed = (time.perf_counter() - start) / 100 * 1000

            summary = mem.get_decision_summary(days=30)
            print(f"{n:>10}  {elapsed:>12.3f}  {summary['total_decisions']:>10}")


if __name__ == "__main__":
    benchmark_boundary_inclusion()
    benchmark_scaling()
