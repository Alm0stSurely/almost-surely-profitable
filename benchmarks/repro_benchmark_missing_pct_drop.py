#!/usr/bin/env python3
"""Repro gate: missing pct on buy/sell must drop loudly, never reach the executor.

Pre-fix contract (the bug): ``_validate_actions`` accepted buy/sell actions
without ``pct``. The live executor then applied ``action.get('pct', 0)`` —
a silent 0% no-op buy, and via ``pct if pct > 0 else None`` a FULL position
exit on sell, recorded with the fabricated pct=0. Scope of failure (whole
position liquidated, cooldown armed, decision record lying about the size)
vastly exceeded scope of malformation (one omitted field).

Post-fix contract: missing pct on a non-hold action drops the action with a
per-element ERROR log naming the defect; valid siblings survive; hold stays
pct-optional; the all-dropped path still logs every defect before the
zero-survivor hold-all.

Gate exits 1 with named XX failures on pre-fix code, 0 on post-fix code.
The harness wraps every scenario so a pre-fix crash is recorded as a named
failure, not a traceback that hides later scenarios.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.llm.trading_agent import TradingAgent  # noqa: E402


class _RecordingHandler(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.records = []

    def emit(self, record):
        self.records.append(record)


def _run_scenario(name, response_actions, expect_valid, expect_dropped_reasons):
    """Parse *response_actions* and check the post-fix contract.

    Returns a list of failure strings; empty means the scenario passes.
    """
    failures = []
    handler = _RecordingHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    old_level = root.level
    root.setLevel(logging.DEBUG)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            agent = TradingAgent(
                api_key="gate-key",
                history_file=str(Path(tmp) / "decisions.json"),
            )
            payload = json.dumps({"actions": response_actions})
            try:
                result = agent.parse_response(payload)
            except Exception as exc:  # pre-fix code must fail, not crash
                return [f"{name}: parse_response raised {type(exc).__name__}: {exc}"]

        got_valid = result.get("actions")
        if got_valid != expect_valid:
            failures.append(
                f"{name}: expected valid actions {expect_valid!r}, got {got_valid!r}"
            )

        messages = [r.getMessage() for r in handler.records]
        for reason in expect_dropped_reasons:
            hits = [m for m in messages if "dropped:" in m and reason in m]
            if len(hits) != 1:
                failures.append(
                    f"{name}: expected exactly 1 per-element drop log naming "
                    f"{reason!r}, got {len(hits)}"
                )
    finally:
        root.removeHandler(handler)
        root.setLevel(old_level)
    return failures


def main():
    scenarios = [
        (
            "sell-missing-pct-drops",
            [{"ticker": "SPY", "action": "sell"},
             {"ticker": "GLD", "action": "hold"}],
            [{"ticker": "GLD", "action": "hold"}],
            ["missing pct on sell"],
        ),
        (
            "buy-missing-pct-drops-sibling-survives",
            [{"ticker": "MC.PA", "action": "buy"},
             {"ticker": "SPY", "action": "buy", "pct": 10}],
            [{"ticker": "SPY", "action": "buy", "pct": 10}],
            ["missing pct on buy"],
        ),
        (
            "hold-missing-pct-still-accepted",
            [{"ticker": "SPY", "action": "hold"}],
            [{"ticker": "SPY", "action": "hold"}],
            [],
        ),
        (
            "buy-with-pct-accepted",
            [{"ticker": "SPY", "action": "buy", "pct": 25}],
            [{"ticker": "SPY", "action": "buy", "pct": 25}],
            [],
        ),
        (
            "non-finite-pct-still-dropped",
            [{"ticker": "SPY", "action": "buy", "pct": 1e999},
             {"ticker": "GLD", "action": "hold"}],
            [{"ticker": "GLD", "action": "hold"}],
            ["pct present but not a finite number"],
        ),
        (
            "all-dropped-path-logs-every-defect",
            [{"ticker": "SPY", "action": "sell"},
             {"ticker": "MC.PA", "action": "buy"}],
            [],
            ["missing pct on sell", "missing pct on buy"],
        ),
    ]

    all_failures = []
    for name, actions, expect_valid, expect_reasons in scenarios:
        all_failures.extend(_run_scenario(name, actions, expect_valid, expect_reasons))

    if all_failures:
        print(f"GATE FAILED — {len(all_failures)} check(s) red:")
        for f in all_failures:
            print(f"  XX {f}")
        return 1
    print(f"GATE PASSED — {len(scenarios)} scenarios, all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
