"""
Micro-benchmark for the non-finite guards in decision_memory.py.

The benchmark measures the cost of filtering non-finite P&L values and the
safe formatting helpers used in the LLM context and lesson generation.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.decision_memory import (
    DecisionMemory,
    DecisionRecord,
    _safe_pct_str,
    _safe_value_str,
)


def _bench(func, value, n=100_000):
    # Warmup
    for _ in range(1000):
        func(value)

    start = time.perf_counter()
    for _ in range(n):
        func(value)
    elapsed = time.perf_counter() - start

    return elapsed / n * 1e6  # microseconds per call


def _build_records():
    today = datetime.now().strftime("%Y-%m-%d")
    return [
        DecisionRecord(
            date=today,
            ticker="AI.PA",
            action="buy",
            quantity=10.0,
            price=150.0,
            portfolio_value_before=10000.0,
            portfolio_value_after=9850.0,
            pnl_pct=5.0,
            holding_period_days=5,
        ),
        DecisionRecord(
            date=today,
            ticker="MC.PA",
            action="buy",
            quantity=5.0,
            price=800.0,
            portfolio_value_before=10000.0,
            portfolio_value_after=9900.0,
            pnl_pct=-2.0,
            holding_period_days=10,
        ),
    ]


def main():
    records = _build_records()

    # In-memory DecisionMemory without loading from disk
    mem = DecisionMemory(memory_file="/dev/null/nonexistent.json")
    mem.decisions = records

    start = time.perf_counter()
    n = 10_000
    for _ in range(n):
        mem.get_memory_context_for_llm()
    context_us = (time.perf_counter() - start) / n * 1e6

    start = time.perf_counter()
    for _ in range(n):
        mem.generate_lessons_learned()
    lessons_us = (time.perf_counter() - start) / n * 1e6

    results = [
        ("get_memory_context_for_llm", "finite 2 trades", context_us),
        ("generate_lessons_learned", "finite 2 trades", lessons_us),
        ("_safe_value_str", "finite", _bench(_safe_value_str, 1234.56)),
        ("_safe_value_str", "NaN", _bench(_safe_value_str, float("nan"))),
        ("_safe_pct_str", "finite +", _bench(lambda v: _safe_pct_str(v, symbol="+"), 5.0)),
        ("_safe_pct_str", "NaN +", _bench(lambda v: _safe_pct_str(v, symbol="+"), float("nan"))),
        ("_safe_pct_str", "finite ratio", _bench(lambda v: _safe_pct_str(v, as_ratio=True), 0.55)),
        ("_safe_pct_str", "NaN ratio", _bench(lambda v: _safe_pct_str(v, as_ratio=True), float("nan"))),
    ]

    print("| Helper | Input | µs/call |")
    print("|---|---|---|")
    for helper, inp, us in results:
        print(f"| {helper} | {inp} | {us:.3f} |")


if __name__ == "__main__":
    main()
