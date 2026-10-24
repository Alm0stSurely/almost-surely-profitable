#!/usr/bin/env python3
"""Repro gate: degenerate pct (0 / negative / >100) on buy/sell must not
reach the executors' coercion conventions.

Background (PR #93): ``_validate_actions`` admits any *finite* pct. The
executors (daily_run.py:490, backtest.py:388, backtest.py:525) apply the
convention ``pct=pct if pct > 0 else None`` on sell:

- pct = 0   -> None -> Portfolio.sell(pct=None) -> sell_pct = 100.0
  -> FULL position exit. ``Portfolio.sell`` would have *rejected* 0
  (``_is_valid_percentage`` is (0, 100]), but the executor coercion
  rewrites the invalid order into the most extreme valid one before the
  portfolio layer ever sees it.
- pct = -5  -> None -> same full exit.
- pct = 150 -> passed verbatim -> ``_is_valid_percentage`` rejects loudly
  downstream (no fabrication, but the chokepoint should own the contract).

Buy side: pct = 0/-10 reaches ``Portfolio.buy`` which rejects loudly
("Invalid percentage") — no fabrication, but a wasted session slot and an
asymmetric convention: sell-0 fabricates, buy-0 merely fails.

Post-fix contract: the chokepoint drops pct outside (0, 100] on buy/sell
with a named per-element reason, siblings survive (PR #64 Lipschitz).

Gate protocol (PR #86/#87 lessons): every scenario runs inside try/except,
checks are defensive, pre-fix exit is 1 with the FULL named failure set
and zero tracebacks; healthy pins pass on BOTH sides.
"""

import io
import json
import logging
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.trading_agent import TradingAgent  # noqa: E402
from src.portfolio.portfolio import Portfolio  # noqa: E402


def _make_agent(tmp):
    return TradingAgent(api_key="test-key", history_file=str(Path(tmp) / "decisions.json"))


def _validated_actions(agent, response_json):
    """Run the decision through the real production parse path."""
    result = agent.parse_response(response_json)
    return result


def _executor_sell_outcome(pct, tmp):
    """Simulate daily_run.py:490's coercion + Portfolio.sell end state.

    Returns (position_exists_after, executed_full_exit, error).
    """
    try:
        portfolio = Portfolio(data_dir=str(Path(tmp) / "pf"))
        portfolio.cash = 0.0
        portfolio.positions["SPY"] = portfolio.__class__.__module__ and __import__(
            "src.portfolio.portfolio", fromlist=["Position"]
        ).Position(ticker="SPY", quantity=10.0, avg_price=100.0, current_price=120.0)
        # Executor convention under test:
        coerced = pct if pct > 0 else None
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            portfolio.sell("SPY", 120.0, pct=coerced)
        full_exit = "SPY" not in portfolio.positions
        return (not full_exit, full_exit, None)
    except Exception as e:  # gate must record, not crash (PR #87)
        return (None, None, f"{type(e).__name__}: {e}")


def _executor_buy_outcome(pct, tmp):
    """Simulate daily_run.py:462's pass-through + Portfolio.buy result."""
    try:
        portfolio = Portfolio(data_dir=str(Path(tmp) / "pf"))
        portfolio.cash = 1000.0
        buf_out, buf_err = io.StringIO(), io.StringIO()
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            ok = portfolio.buy("SPY", pct, 100.0)
        return (ok, None)
    except Exception as e:
        return (None, f"{type(e).__name__}: {e}")


SCENARIOS = []


def scenario(fn):
    SCENARIOS.append(fn)
    return fn


@scenario
def sell_zero_pct_full_exit(agent, tmp):
    """sell pct=0 must NOT survive validation (executor would fully exit)."""
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "sell", "pct": 0}]}
    ))
    survived = any(a.get("action") == "sell" for a in result.get("actions", []))
    checks = {}
    checks["sell-pct-0 dropped"] = (not survived, f"survived={survived}")
    if survived:
        _, full_exit, err = _executor_sell_outcome(0, tmp)
        checks["executor would fully exit"] = (full_exit is True, f"full_exit={full_exit} err={err}")
    return checks


@scenario
def sell_negative_pct_full_exit(agent, tmp):
    """sell pct=-5 must NOT survive validation (same coercion)."""
    result = _validated_actions(agent, json.dumps(
        {"actions": [
            {"ticker": "SPY", "action": "sell", "pct": -5},
            {"ticker": "GLD", "action": "hold"},
        ]}
    ))
    survived = any(a.get("action") == "sell" for a in result.get("actions", []))
    sibling = any(a.get("action") == "hold" and a.get("ticker") == "GLD" for a in result.get("actions", []))
    checks = {}
    checks["sell-pct-neg dropped"] = (not survived, f"survived={survived}")
    checks["hold sibling survives"] = (sibling, f"sibling={sibling}")
    if survived:
        _, full_exit, err = _executor_sell_outcome(-5, tmp)
        checks["executor would fully exit"] = (full_exit is True, f"full_exit={full_exit} err={err}")
    return checks


@scenario
def sell_overflow_pct_dropped(agent, tmp):
    """sell pct=150: downstream rejects loudly, but chokepoint must own it."""
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "sell", "pct": 150}]}
    ))
    survived = any(a.get("action") == "sell" for a in result.get("actions", []))
    return {"sell-pct-150 dropped": (not survived, f"survived={survived}")}


@scenario
def buy_zero_pct_dropped(agent, tmp):
    """buy pct=0: downstream loud-rejects; chokepoint should own the domain."""
    result = _validated_actions(agent, json.dumps(
        {"actions": [
            {"ticker": "SPY", "action": "buy", "pct": 0},
            {"ticker": "GLD", "action": "buy", "pct": 10},
        ]}
    ))
    zero_survived = any(a.get("action") == "buy" and a.get("pct") == 0 for a in result.get("actions", []))
    ten_survived = any(a.get("action") == "buy" and a.get("pct") == 10 for a in result.get("actions", []))
    return {
        "buy-pct-0 dropped": (not zero_survived, f"survived={zero_survived}"),
        "buy-pct-10 sibling survives": (ten_survived, f"sibling={ten_survived}"),
    }


@scenario
def buy_negative_pct_dropped(agent, tmp):
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "buy", "pct": -10}]}
    ))
    survived = any(a.get("action") == "buy" for a in result.get("actions", []))
    return {"buy-pct-neg dropped": (not survived, f"survived={survived}")}


@scenario
def buy_overflow_pct_dropped(agent, tmp):
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "buy", "pct": 250}]}
    ))
    survived = any(a.get("action") == "buy" for a in result.get("actions", []))
    return {"buy-pct-250 dropped": (not survived, f"survived={survived}")}


@scenario
def loud_reason_names_degenerate_pct(agent, tmp):
    """Mechanism pin (PR #83 doctrine): the log record names the defect."""
    logging.getLogger().setLevel(logging.ERROR)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        agent.parse_response(json.dumps(
            {"actions": [{"ticker": "SPY", "action": "sell", "pct": 0}]}
        ))
        log_text = stream.getvalue()
    finally:
        root.removeHandler(handler)
    named = "pct" in log_text and ("0" in log_text)
    return {"drop log names pct defect": (named, f"log={log_text.strip()[:160]!r}")}


# --- Healthy pins: must pass on BOTH sides of the fix ---

@scenario
def healthy_partial_sell_survives(agent, tmp):
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "sell", "pct": 50}]}
    ))
    ok = result.get("actions") == [{"ticker": "SPY", "action": "sell", "pct": 50}]
    return {"sell-pct-50 accepted": (ok, f"actions={result.get('actions')}")}


@scenario
def healthy_full_exit_sell_survives(agent, tmp):
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "sell", "pct": 100}]}
    ))
    ok = result.get("actions") == [{"ticker": "SPY", "action": "sell", "pct": 100}]
    return {"sell-pct-100 (full exit) accepted": (ok, f"actions={result.get('actions')}")}


@scenario
def healthy_buy_survives(agent, tmp):
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "buy", "pct": 15}]}
    ))
    ok = result.get("actions") == [{"ticker": "SPY", "action": "buy", "pct": 15}]
    return {"buy-pct-15 accepted": (ok, f"actions={result.get('actions')}")}


@scenario
def healthy_hold_without_pct_survives(agent, tmp):
    result = _validated_actions(agent, json.dumps(
        {"actions": [{"ticker": "SPY", "action": "hold"}]}
    ))
    ok = result.get("actions") == [{"ticker": "SPY", "action": "hold"}]
    return {"hold (no pct) accepted": (ok, f"actions={result.get('actions')}")}


def main():
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        for sc in SCENARIOS:
            name = sc.__name__
            try:
                agent = _make_agent(tmp)
                checks = sc(agent, tmp) or {}
            except Exception as e:
                failures.append(f"XX {name}: HARNESS ERROR {type(e).__name__}: {e}")
                continue
            for label, (ok, detail) in checks.items():
                if not ok:
                    failures.append(f"XX {name} :: {label} ({detail})")

    if failures:
        print(f"REPRO GATE: {len(failures)} failure(s), {len(SCENARIOS)} scenarios")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print(f"REPRO GATE: all {len(SCENARIOS)} scenarios green")
    sys.exit(0)


if __name__ == "__main__":
    main()
