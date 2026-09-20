"""
Benchmark for corrupt-ledger quarantine in save_trade() / save_decision().

Resilience fix on a cold path (a handful of saves per day): measures the
healthy-path append cost and the quarantine path (one rename syscall) so we
can state honestly that preserving a corrupt ledger instead of truncating it
adds no measurable overhead to the daily pipeline.

Run:  .venv/bin/python benchmarks/benchmark_ledger_quarantine.py
"""
import json
import logging
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.getLogger("utils").setLevel(logging.CRITICAL)

from llm.trading_agent import TradingAgent
from portfolio.portfolio import Portfolio, Trade


def make_trade(i: int) -> Trade:
    return Trade(
        timestamp=datetime.now().isoformat(),
        ticker=f"T{i % 20}",
        action="buy",
        quantity=1.0,
        price=100.0,
        total_value=100.0,
        fees=0.0,
    )


def bench_save_trade(n_trades: int, n_iter: int, rounds: int = 3) -> float:
    """Best-of-*rounds* average wall time of one save_trade() on a healthy ledger."""
    best = float("inf")
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            pf = Portfolio(data_dir=tmp)
            for i in range(n_trades):
                pf.save_trade(make_trade(i))
            t0 = time.perf_counter()
            for i in range(n_iter):
                pf.save_trade(make_trade(i))
            dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def bench_quarantine_path(n_iter: int = 2000, rounds: int = 3) -> float:
    """Best-of-*rounds* average wall time of one save_trade() onto a corrupt ledger.

    Each iteration rewrites the corrupt payload and restores it after the
    call (the call quarantines/renames it), so every timed call performs the
    full quarantine path.
    """
    best = float("inf")
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            pf = Portfolio(data_dir=tmp)
            pf.trades_file.write_text("{corrupt")
            t0 = time.perf_counter()
            for i in range(n_iter):
                pf.trades_file.write_text("{corrupt")
                pf.save_trade(make_trade(i))
            dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def bench_save_decision(n_iter: int = 1000, rounds: int = 3) -> float:
    best = float("inf")
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            agent = TradingAgent(history_file=str(Path(tmp) / "decision_history.json"))
            for i in range(50):
                agent.save_decision({"timestamp": datetime.now().isoformat(),
                                     "actions": [], "reasoning": f"d{i}"})
            t0 = time.perf_counter()
            for i in range(n_iter):
                agent.save_decision({"timestamp": datetime.now().isoformat(),
                                     "actions": [], "reasoning": f"d{i}"})
            dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def main():
    t_100 = bench_save_trade(100, n_iter=1000)
    t_1000 = bench_save_trade(1000, n_iter=200)
    t_quarantine = bench_quarantine_path()
    t_decision = bench_save_decision()

    # Sanity: the quarantine path actually leaves evidence behind.
    with tempfile.TemporaryDirectory() as tmp:
        pf = Portfolio(data_dir=tmp)
        pf.trades_file.write_text("{corrupt")
        pf.save_trade(make_trade(0))
        assert len(list(Path(tmp).glob("*.corrupt-*"))) == 1

    print(f"save_trade healthy ledger (100 records) : {t_100 * 1e6:8.1f} us/call")
    print(f"save_trade healthy ledger (1000 records): {t_1000 * 1e6:8.1f} us/call")
    print(f"save_trade quarantine path (corrupt)    : {t_quarantine * 1e6:8.1f} us/call")
    print(f"save_decision healthy (50 records)      : {t_decision * 1e6:8.1f} us/call")
    print()
    print("All times best-of-3; quarantine path = corrupt payload restore "
          "+ save_trade (rename syscall). Daily pipeline does a handful of "
          "saves — negligible vs the 2-5 s fetch + LLM I/O.")


if __name__ == "__main__":
    main()
