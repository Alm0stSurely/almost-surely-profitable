"""Benchmark: build_prompt asset-indicator guard overhead (healthy + degraded paths).

A/B note: run with the fix in place, then `git stash push src/llm/trading_agent.py`
and rerun — the healthy-path numbers are comparable across versions; the
degraded sections exercise the new absent/None branch and only render n/a on
new code (on old code, absent keys launder into fictitious €0.00 / RSI 50.0
readings and a stored None crashes the build with TypeError).

The benchmark also embeds functional asserts: on old code the degraded rows
either render the fictitious values or raise, so the run fails — an
in-benchmark fail-on-old.
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

LATEST_HEALTHY = {
    "price": 610.2, "sma_20": 605.1, "sma_50": 598.4,
    "rsi_14": 55.2, "bb_position": 0.62,
    "volatility_annual": 0.18, "drawdown": -0.04, "daily_return": 0.003,
}

MARKET_HEALTHY = {
    "assets": {"SPY": {"latest": dict(LATEST_HEALTHY)}},
    "correlations": {},
    "regime": None,
}

# All seven keys absent at once (reachable in production: get_latest_indicators
# returns {} for an empty indicator frame).
MARKET_ABSENT = {
    "assets": {"SPY": {"latest": {}}},
    "correlations": {},
    "regime": None,
}

# Stored None in the scaled ratios: crashes the old build (`None * 100`),
# renders n/a% on new code; finite siblings must survive (per-element drop).
LATEST_NONE = dict(LATEST_HEALTHY, volatility_annual=None, drawdown=None)
MARKET_NONE = {
    "assets": {"SPY": {"latest": LATEST_NONE}},
    "correlations": {},
    "regime": None,
}

PORTFOLIO = {"cash": 2725.69, "positions": []}


def make_agent():
    tmpdir = tempfile.TemporaryDirectory()
    return TradingAgent(api_key="test", history_file=str(Path(tmpdir.name) / "d.json"))


def bench(build_market, label):
    agent = make_agent()
    samples = []
    for _ in range(ROUNDS):
        start = time.perf_counter()
        for _ in range(N_ITER):
            agent.build_prompt(build_market(), PORTFOLIO)
        samples.append((time.perf_counter() - start) / N_ITER * 1e6)
    best = min(samples)
    print(f"{label:<34} {best:8.2f} us/op  (best of {ROUNDS})")
    return best


def check(market, expect_absent_na):
    agent = make_agent()
    prompt = agent.build_prompt(market, PORTFOLIO)
    if expect_absent_na:
        assert "Price: €n/a" in prompt, prompt
        assert "RSI(14): n/a" in prompt, prompt
        assert "Volatility (ann): n/a%" in prompt, prompt
        assert "Price: €0.00" not in prompt, prompt
        assert "RSI(14): 50.0" not in prompt, prompt
    else:
        assert "Volatility (ann): n/a%" in prompt, prompt
        assert "Price: €610.20" in prompt, prompt


if __name__ == "__main__":
    # Functional gate first: fails on old code (fictitious readings / TypeError).
    check(MARKET_ABSENT, expect_absent_na=True)
    check(MARKET_NONE, expect_absent_na=False)
    print("functional gate: OK (degraded rows render n/a, no fictitious values)")
    print()
    bench(lambda: MARKET_HEALTHY, "healthy (finite indicators)")
    bench(lambda: MARKET_ABSENT, "absent keys (all 7 missing)")
    bench(lambda: MARKET_NONE, "None scaled ratios (2 fields)")
