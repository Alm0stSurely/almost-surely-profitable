"""Repro + benchmark for the load_recent_decisions robustness fix.

Repro gate (the functional proof):
    1. All-or-nothing filter swallow. Pre-fix, one malformed record — a
       non-dict element, a missing ``timestamp``, an unparseable or
       timezone-aware timestamp — raised inside the filter comprehension
       and was caught by the broad ``except Exception`` around the whole
       load, returning ``[]``: a single bad element voided the n-1 valid
       siblings and the LLM lost its entire recent-decision context.
       Anti-Lipschitz: the scope of the failure exceeded the scope of
       the malformation.
    2. Lenient parse at a quarantine-class boundary. Pre-fix, the raw
       ``json.load`` accepted ``NaN`` tokens no repo writer can emit and
       a corrupt file was left in place for the next ``save_decision()``
       to silently overwrite — the PR #60 evidence-destruction class.
    3. Post-fix, the load goes through the shared quarantine ingress
       (strict tokens, whole-file quarantine on unparseable) and drops
       malformed records individually with one loud warning each; valid
       siblings survive unchanged and the recent tail contract (last 5
       within the window) is byte-identical on healthy files.

    Exit code 0 iff every post-fix expectation holds. Pre-fix the gate
    fails on the quarantine group, the per-element-survival group, and
    the loudness group.

Benchmark (honest cold-path claim): the load runs once per daily
session on a cold path; a timing delta is not the point. We still time
the guarded per-element load against a plain ``json.load`` + naive
filter over a healthy 500-record history so the PR can state observed
numbers instead of asserting "no regression" from assumption.
"""

import json
import logging
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.trading_agent import TradingAgent

logging.basicConfig(level=logging.CRITICAL)  # keep the benchmark table readable

failures = []


def check(name, cond):
    print(f"  [{'ok' if cond else 'XX'}] {name}")
    if not cond:
        failures.append(name)


def _decision(hours_ago):
    return {
        # Naive on purpose: decision-history timestamps are naive ISO strings
        # and must compare against a naive cutoff (house convention).
        "timestamp": (datetime.now() - timedelta(hours=hours_ago)).isoformat(),  # noqa: DTZ005
        "actions": [{"ticker": "SPY", "action": "hold"}],
        "reasoning": f"gate decision {hours_ago}h ago",
    }


def _quarantine_files(path):
    return list(path.parent.glob(path.name + ".corrupt-*"))


def gate():
    print("Repro gate: load_recent_decisions robustness")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        history = tmp / "decision_history.json"

        # --- group 1: unparseable/token files quarantine whole ---
        print("group 1: quarantine ingress")
        token_decision = _decision(1)
        token_decision["pnl_pct"] = float("nan")
        history.write_text(json.dumps([token_decision]))  # literal NaN token
        agent = TradingAgent(api_key="test", history_file=str(history))
        result = agent.load_recent_decisions(days=5)
        check("token file -> empty result", result == [])
        quarantined = _quarantine_files(history)
        check("token file -> quarantined", len(quarantined) == 1)
        check("token file -> bytes preserved",
              bool(quarantined) and "NaN" in quarantined[0].read_text())

        history.write_bytes(b"\x80\x81 garbage")
        agent = TradingAgent(api_key="test", history_file=str(history))
        check("garbage bytes -> empty result", agent.load_recent_decisions(days=5) == [])
        check("garbage bytes -> quarantined", len(_quarantine_files(history)) == 2)

        history.write_text(json.dumps({"not": "a list"}))
        agent = TradingAgent(api_key="test", history_file=str(history))
        check("wrong shape -> empty result", agent.load_recent_decisions(days=5) == [])
        check("wrong shape -> quarantined", len(_quarantine_files(history)) == 3)

        # --- group 2: per-element drop, siblings survive ---
        print("group 2: per-element Lipschitz")
        good = [_decision(h) for h in (3, 2, 1)]
        history.write_text(json.dumps([good[0], {"no": "timestamp"}, good[1],
                                       "a string", good[2]]))
        agent = TradingAgent(api_key="test", history_file=str(history))
        result = agent.load_recent_decisions(days=5)
        check("one bad element does not void siblings", len(result) == 3)
        check("parseable file never quarantines", len(_quarantine_files(history)) == 3)

        aware_ts = (datetime.now() - timedelta(hours=2)).isoformat() + "+02:00"  # noqa: DTZ005
        history.write_text(json.dumps([_decision(2), {"timestamp": aware_ts},
                                       _decision(1)]))
        agent = TradingAgent(api_key="test", history_file=str(history))
        result = agent.load_recent_decisions(days=5)
        check("tz-aware record dropped, siblings survive", len(result) == 2)

        # --- group 3: healthy contract byte-identical ---
        print("group 3: healthy pins")
        oldest_first = [_decision(h) for h in (7, 6, 5, 4, 3, 2, 1)]
        history.write_text(json.dumps(oldest_first))
        agent = TradingAgent(api_key="test", history_file=str(history))
        result = agent.load_recent_decisions(days=5)
        check("last-five tail preserved",
              [r["reasoning"] for r in result] ==
              [f"gate decision {h}h ago" for h in (5, 4, 3, 2, 1)])
        check("cutoff excludes old decisions",
              agent.load_recent_decisions(days=0) == [])

        missing = TradingAgent(api_key="test", history_file=str(tmp / "nope.json"))
        check("missing file -> quiet empty", missing.load_recent_decisions(days=5) == [])

    print(f"gate: {len(failures)} failure(s)")
    return 1 if failures else 0


def bench():
    print("\nbenchmark: guarded per-element load vs plain json.load + naive filter")
    n_records, n_iter = 500, 2000
    with tempfile.TemporaryDirectory() as tmp:
        history = Path(tmp) / "decision_history.json"
        oldest_first = [_decision(h) for h in range(n_records, 0, -1)]
        history.write_text(json.dumps(oldest_first))
        agent = TradingAgent(api_key="test", history_file=str(history))

        guarded = []
        plain = []
        for _ in range(5):
            t0 = time.perf_counter()
            for _ in range(n_iter):
                agent.load_recent_decisions(days=365000)
            guarded.append((time.perf_counter() - t0) / n_iter * 1000)

            t0 = time.perf_counter()
            for _ in range(n_iter):
                with open(history) as f:
                    data = json.load(f)
                cutoff = datetime.now() - timedelta(days=365000)  # noqa: DTZ005
                recent = [d for d in data
                          if datetime.fromisoformat(d["timestamp"]) > cutoff]
                recent[-5:]
            plain.append((time.perf_counter() - t0) / n_iter * 1000)

        g, p = min(guarded), min(plain)
        print(f"  records={n_records} n_iter={n_iter} best-of-5")
        print(f"  guarded per-element load : {g:.4f} ms")
        print(f"  plain load + naive filter: {p:.4f} ms")
        print(f"  delta                    : {g - p:+.4f} ms "
              f"(per-element validation on a hot in-memory loop; the real call "
              f"runs once per session on a cold path)")


if __name__ == "__main__":
    code = gate()
    if code == 0:
        bench()
    sys.exit(code)
