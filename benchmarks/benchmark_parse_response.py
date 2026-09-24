"""Benchmark: parse_response healthy vs malformed-mix vs nested-context.

The per-action validation and enclosing-brace extraction must not add
material cost to the healthy path — the failure direction fix is a
behavioral change, not a performance tax.
"""

import json
import os
import sys
import tempfile
import time
from contextlib import redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.trading_agent import TradingAgent

_BENCH_DIR = Path(tempfile.mkdtemp(prefix="parse_bench_"))


def _bench(label, response, rounds=5, iters=2000):
    agent = TradingAgent(api_key="bench", history_file=str(_BENCH_DIR / "bench.json"))
    # Warmup
    for _ in range(50):
        agent.parse_response(response)
    times = []
    for _ in range(rounds):
        start = time.perf_counter()
        for _ in range(iters):
            agent.parse_response(response)
        times.append((time.perf_counter() - start) / iters * 1e6)
    best = min(times)
    print(f"  {label:<38} {best:8.1f} µs/call (best of {rounds})")
    return best


def main():
    healthy = json.dumps({
        "actions": [{"ticker": t, "action": "hold"} for t in ("SPY", "GLD", "TLT", "FEZ", "QQQ")],
        "reasoning": "steady state, no changes warranted today",
    })
    case_mixed = json.dumps({
        "actions": [
            {"ticker": "SPY", "action": "HOLD"},
            {"ticker": "GLD", "action": "SELL", "pct": 25},
            {"ticker": "TLT", "action": "Buy", "pct": 10},
        ],
        "reasoning": "partial rotation",
    })
    malformed_mix = json.dumps({
        "actions": [
            {"ticker": "SPY", "action": "sell", "pct": 100},
            {"ticker": "GLD", "action": "byu", "pct": 10},
            {"action": "hold"},
            "SPY hold",
        ],
        "reasoning": "exit plus noise",
    })
    nested_context = (
        '{"reasoning": "rotate defensively", '
        '"context": {"regime": {"trend": "down", "depth": {"level": 2, "note": "x"}}}, '
        '"actions": [{"ticker": "SPY", "action": "sell", "pct": 50}, '
        '{"ticker": "GLD", "action": "buy", "pct": 10}], '
        '"confidence": 0.7}'
    )

    print("parse_response benchmark (2000 iters/round):")
    # Drop-path logging executes (and is timed) but is not printed.
    devnull = open(os.devnull, "w")
    with redirect_stderr(devnull):
        _bench("healthy (5 holds)", healthy)
        _bench("case-mixed (3 actions, normalized)", case_mixed)
        _bench("malformed-mix (2 valid, 2 dropped)", malformed_mix)
        _bench("nested-context (extraction walk)", nested_context)

        # Throughput sanity: decisions per second on healthy path
        agent = TradingAgent(api_key="bench", history_file=str(_BENCH_DIR / "bench.json"))
        start = time.perf_counter()
        n = 20000
        for _ in range(n):
            agent.parse_response(healthy)
    devnull.close()
    elapsed = time.perf_counter() - start
    print(f"\n  healthy throughput: {n / elapsed:,.0f} parses/s")


if __name__ == "__main__":
    main()
