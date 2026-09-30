"""Benchmark: build_prompt portfolio-totals guard overhead (healthy + degraded paths).

A/B note: run with the fix in place, then `git stash push src/llm/trading_agent.py`
and rerun — the healthy-path numbers are comparable across versions; the
degraded sections exercise the new absent/None branch and only render n/a on
new code (on old code, absent keys launder into fictitious €0.00 readings).
"""
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")

from llm.trading_agent import TradingAgent

N_ITER = 2000
ROUNDS = 5

MARKET = {
    "assets": {
        "SPY": {"latest": {"price": 610.2, "sma_20": 605.1, "sma_50": 598.4,
                           "rsi_14": 55.2, "bb_position": 0.62,
                           "volatility_annual": 0.18, "drawdown": -0.04,
                           "daily_return": 0.003}},
    },
    "correlations": {},
    "regime": None,
}

PORT_HEALTHY = {
    "cash": 2725.69,
    "total_value": 10330.55,
    "total_return_pct": 3.31,
    "total_pnl": 330.55,
    "positions": [],
}

PORT_NONE = {
    "cash": 2725.69,
    "total_value": None,
    "total_return_pct": None,
    "total_pnl": None,
    "positions": [],
}

# cash kept finite: a fully absent block is indistinguishable from a
# partially-degraded one for the changed lines (per-element drop).
PORT_ABSENT = {
    "cash": 2725.69,
    "positions": [],
}


def make_agent():
    tmpdir = tempfile.TemporaryDirectory()
    return TradingAgent(api_key="test", history_file=str(Path(tmpdir.name) / "d.json"))


def time_build(portfolio, n_iter, agent):
    start = time.perf_counter()
    for _ in range(n_iter):
        agent.build_prompt(MARKET, portfolio)
    return (time.perf_counter() - start) / n_iter * 1e6  # µs/call


def bench(label, portfolio):
    agent = make_agent()
    # warmup
    time_build(portfolio, 200, agent)
    samples = [time_build(portfolio, N_ITER, agent) for _ in range(ROUNDS)]
    best = min(samples)
    print(f"  {label:<42} best {best:8.1f} µs/call  (median {statistics.median(samples):8.1f})")
    return best


print(f"build_prompt portfolio-totals benchmark — N_ITER={N_ITER}, ROUNDS={ROUNDS}")
print()

healthy = bench("healthy totals block (all finite)", PORT_HEALTHY)
t_none = bench("degraded: None totals", PORT_NONE)
t_absent = bench("degraded: absent total keys", PORT_ABSENT)

# Functional check that the degraded paths actually render through the
# changed code (a benchmark that never triggers the change verifies nothing).
agent = make_agent()
p_none = agent.build_prompt(MARKET, PORT_NONE)
p_absent = agent.build_prompt(MARKET, PORT_ABSENT)
assert "Total Value: €n/a" in p_none
assert "Total Return: n/a%" in p_none
assert "Total Value: €n/a" in p_absent
print()
print(f"  guard overhead vs healthy: None={t_none - healthy:+.1f} µs, absent={t_absent - healthy:+.1f} µs")
print()
print("Done.")
