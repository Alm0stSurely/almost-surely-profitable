"""Benchmark: build_prompt risk-metric guard overhead (healthy + degraded paths).

A/B note: run with the fix in place, then `git stash push src/llm/trading_agent.py`
and rerun — the healthy-path numbers are comparable across versions; the
degraded sections exercise the new `_safe_pct` branch and only run on new code.
"""
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, "src")

from llm.trading_agent import TradingAgent

try:
    from llm.trading_agent import _safe_pct
    NEW_CODE = True
except ImportError:
    NEW_CODE = False

N_ITER = 2000
ROUNDS = 5

MARKET = {
    "assets": {
        "SPY": {"latest": {"price": 610.2, "sma_20": 605.1, "sma_50": 598.4,
                           "rsi_14": 55.2, "bb_position": 0.62,
                           "volatility_annual": 0.18, "drawdown": -0.04,
                           "daily_return": 0.003}},
        "GLD": {"latest": {"price": 248.5, "sma_20": 246.0, "sma_50": 241.3,
                           "rsi_14": 48.1, "bb_position": 0.44,
                           "volatility_annual": 0.12, "drawdown": -0.02,
                           "daily_return": 0.001}},
    },
    "correlations": {},
    "regime": None,
}

BASE_PORT = {
    "cash": 2725.69,
    "total_value": 10330.55,
    "total_return_pct": 3.31,
    "total_pnl": 330.55,
    "positions": [
        {"ticker": "SPY", "quantity": 5.0, "avg_price": 590.0,
         "current_price": 610.2, "unrealized_pnl_pct": 3.42,
         "market_value": 3051.0},
    ],
}

RISK_HEALTHY = {
    "cvar_95": -0.025, "var_95": -0.02, "max_drawdown": -0.10,
    "sortino_ratio": 1.23, "skewness": -0.45, "kurtosis": 4.56,
}
RISK_NONE = {
    "cvar_95": None, "var_95": -0.02, "max_drawdown": -0.10,
    "sortino_ratio": None, "skewness": 0.10, "kurtosis": 3.0,
}
RISK_ABSENT = {"cvar_95": -0.025, "var_95": -0.02, "max_drawdown": -0.10}


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
    print(f"  {label:<38} best {best:8.1f} µs/call  (median {statistics.median(samples):8.1f})")
    return best


print(f"build_prompt guard benchmark — N_ITER={N_ITER}, ROUNDS={ROUNDS}, code={'NEW' if NEW_CODE else 'OLD'}")
print()

port_healthy = dict(BASE_PORT, risk_metrics=dict(RISK_HEALTHY))
healthy = bench("healthy risk block (all finite)", port_healthy)

if NEW_CODE:
    port_none = dict(BASE_PORT, risk_metrics=dict(RISK_NONE))
    port_absent = dict(BASE_PORT, risk_metrics=dict(RISK_ABSENT))
    t_none = bench("degraded: None ratios (new path)", port_none)
    t_absent = bench("degraded: absent keys (new path)", port_absent)
    # Functional check that the degraded paths actually render through the
    # changed code (a benchmark that never triggers the change verifies nothing).
    agent = make_agent()
    p_none = agent.build_prompt(MARKET, port_none)
    p_absent = agent.build_prompt(MARKET, port_absent)
    assert "CVaR 95% (Expected Shortfall): n/a%" in p_none
    assert "Sortino Ratio: n/a" in p_absent
    print()
    print(f"  guard overhead vs healthy: None={t_none - healthy:+.1f} µs, absent={t_absent - healthy:+.1f} µs")
print()
print("Done.")
