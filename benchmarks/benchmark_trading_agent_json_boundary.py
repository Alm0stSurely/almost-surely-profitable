"""Benchmark for the TradingAgent decision-history JSON boundary.

Shows that ``TradingAgent.save_decision`` writes valid JSON even when the
decision contains non-finite floats, and that the safe-serialization path
adds only negligible overhead for normal decisions.
"""

import io
import json
import sys
import tempfile
import timeit
import warnings
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.trading_agent import TradingAgent


def _make_decision(pct_value):
    return {
        "timestamp": datetime.now().isoformat(),
        "actions": [
            {"ticker": "SPY", "action": "buy", "pct": pct_value},
            {"ticker": "TLT", "action": "hold"},
        ],
        "reasoning": "benchmark decision",
    }


def _benchmark(name: str, stmt: str, setup: str = "pass", number: int = 1000, globals_=None):
    full_setup = f"import warnings; warnings.simplefilter('error', RuntimeWarning); {setup}"
    elapsed = timeit.timeit(stmt, setup=full_setup, number=number, globals=globals_)
    mean_us = elapsed / number * 1e6
    print(f"{name:45s} | {number:6d} runs | {mean_us:8.2f} µs/run")
    return elapsed


if __name__ == "__main__":
    print("Benchmark: TradingAgent decision JSON boundary")
    print("-" * 80)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        clean_decision = _make_decision(15)
        dirty_decision = _make_decision(float("nan"))

        with warnings.catch_warnings():
            warnings.simplefilter("error", RuntimeWarning)

            # 1. Baseline: standard json.dumps of a clean decision.
            _benchmark(
                "json.dumps (clean decision)",
                "json.dumps(clean_decision, indent=2)",
                "import json",
                number=10000,
                globals_=globals(),
            )

            # 2. Save a clean decision through the agent.
            _benchmark(
                "TradingAgent.save_decision (clean)",
                "agent.save_decision(clean_decision)",
                number=1000,
                globals_=globals(),
            )

            # 3. Save a decision with non-finite values and verify strict JSON.
            agent.save_decision(dirty_decision)
            raw_text = history_file.read_text()
            assert "NaN" not in raw_text and "Infinity" not in raw_text
            parsed = json.loads(raw_text)
            assert parsed[-1]["actions"][0]["pct"] is None

            _benchmark(
                "TradingAgent.save_decision (non-finite)",
                "agent.save_decision(dirty_decision)",
                number=1000,
                globals_=globals(),
            )

            # 4. Round-trip load stays warning-free.
            loaded = agent.load_recent_decisions(days=1)
            assert loaded[-1]["actions"][0]["pct"] is None

    print("-" * 80)
    print("OK - TradingAgent decision boundary writes strict JSON.")
