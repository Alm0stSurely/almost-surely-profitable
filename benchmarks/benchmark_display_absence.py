"""Benchmark: cooldown/decision display block — same-cost lookup variant.

No perf claim: the change removes numeric .get defaults, so per-call cost is
identical modulo a few None checks. This benchmark exists because a benchmark
that never exercises the guarded path verifies nothing:

- healthy rows pin the unchanged render cost,
- degraded rows (absent keys) exercise the new n/a branches,
- the embedded functional gate FAILS on pre-fix code (a truncated cooldown
  dict used to launder into "Weekly trades used: 0/2").

Run from repo root:  python3 benchmarks/benchmark_display_absence.py
"""
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.trading_agent import TradingAgent  # noqa: E402

N_ITER = 2000
ROUNDS = 5

agent = TradingAgent(api_key="dummy", history_file=Path("/tmp/nonexistent_history.json"))
MARKET = {"assets": {}, "correlations": {}}
PORTFOLIO = {"cash": 100.0, "total_value": 10000.0}

HEALTHY = {
    "trades_this_week": 1,
    "weekly_cap": 3,
    "active_entries": {"MC.PA": {"entry_date": "2026-09-01T00:00:00", "hold_days": 12.5}},
    "recent_exits": {"TLT": {"exit_date": "2026-09-15T00:00:00", "days_since_exit": 4.0}},
    "config": {"min_hold_days": 5, "flip_cooldown_days": 10},
}
DEGRADED = {
    "active_entries": {"MC.PA": {"entry_date": "2026-09-01T00:00:00"}},
    "recent_exits": {"TLT": {"exit_date": "2026-09-15T00:00:00"}},
    "config": {},
}
RECENT = [{
    "timestamp": "2026-10-02T21:30:00",
    "reasoning": "x",
    "actions": [
        {"ticker": "AI.PA", "action": "buy", "pct": 17},
        {"ticker": "SPY", "action": "hold"},
    ],
}]

# --- embedded functional gate (fails on pre-fix code) ----------------------
_gate_prompt = agent.build_prompt(MARKET, PORTFOLIO, RECENT, DEGRADED)
assert "Weekly trades used: n/a/n/a" in _gate_prompt, (
    "SENTINEL COLLISION: truncated cooldown dict rendered a numeric budget "
    "reading — absence is being laundered into a default (pre-fix behavior)."
)
assert "- SPY: hold" in _gate_prompt and "- SPY: hold 0%" not in _gate_prompt, (
    "SENTINEL COLLISION: pct-less action rendered a fictitious 0% suffix."
)
print("functional gate: OK (absence renders n/a)")

# --- timing ----------------------------------------------------------------
def bench(label, recent, cooldown):
    samples = []
    for _ in range(ROUNDS):
        t0 = time.perf_counter()
        for _ in range(N_ITER):
            agent.build_prompt(MARKET, PORTFOLIO, recent, cooldown)
        samples.append((time.perf_counter() - t0) / N_ITER * 1e6)
    print(f"  {label:<38} best {min(samples):7.2f} µs  median {statistics.median(samples):7.2f} µs")


print(f"N_ITER={N_ITER}, best of {ROUNDS} rounds")
bench("healthy cooldown + history", RECENT, HEALTHY)
bench("degraded cooldown + history (n/a rows)", RECENT, DEGRADED)
