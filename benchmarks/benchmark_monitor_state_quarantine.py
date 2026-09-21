"""
Benchmark for corrupt-state quarantine in the intraday monitor loaders.

Resilience fix on a cold path (6 monitor runs per day): measures the healthy
load cost and the quarantine path (one rename syscall) for
``load_alert_history`` and ``load_previous_close`` so we can state honestly
that preserving a corrupt state file instead of silently overwriting it adds
no measurable overhead to the monitor loop.

Run:  .venv/bin/python benchmarks/benchmark_monitor_state_quarantine.py
"""
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.getLogger("utils").setLevel(logging.CRITICAL)
logging.getLogger("monitor").setLevel(logging.CRITICAL)

import monitor
from monitor import load_alert_history, load_previous_close

HEALTHY_HISTORY = {
    "alerts": [
        {"ticker": f"T{i}", "type": "position_movement", "movement_pct": -2.5,
         "severity": "high", "timestamp": "2026-09-21T09:00:00"}
        for i in range(50)
    ],
    "last_reset": "2026-09-21T08:00:00",
}

HEALTHY_STATE = {
    "timestamp": "2026-09-21T08:00:00",
    "previous_close": {f"T{i}": 100.0 + i for i in range(20)},
}


def _patch_paths(tmp: str, name: str) -> Path:
    """Point both module-level state paths into *tmp*; return the target file."""
    target = Path(tmp) / name
    monitor.ALERT_HISTORY_PATH = target
    monitor.MARKET_STATE_PATH = target
    return target


def bench_load(name: str, payload: str, loader, n_iter: int, rounds: int = 3) -> float:
    """Best-of-*rounds* average wall time of one healthy load."""
    best = float("inf")
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            path = _patch_paths(tmp, name)
            path.write_text(payload)
            t0 = time.perf_counter()
            for _ in range(n_iter):
                loader()
            dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def bench_quarantine(name: str, loader, n_iter: int, rounds: int = 3) -> float:
    """Best-of-*rounds* average wall time of one load onto a corrupt state.

    Each iteration rewrites the corrupt payload before the call (the call
    quarantines/renames it), so every timed call performs the full quarantine
    path.
    """
    best = float("inf")
    for _ in range(rounds):
        with tempfile.TemporaryDirectory() as tmp:
            path = _patch_paths(tmp, name)
            path.write_text("{corrupt")
            t0 = time.perf_counter()
            for _ in range(n_iter):
                path.write_text("{corrupt")
                loader()
            dt = (time.perf_counter() - t0) / n_iter
        best = min(best, dt)
    return best


def main():
    n = 2000
    prev_close = lambda: load_previous_close(None)  # noqa: E731
    h_hist = bench_load("alert_history.json", json.dumps(HEALTHY_HISTORY),
                        load_alert_history, n)
    q_hist = bench_quarantine("alert_history.json", load_alert_history, n)
    h_state = bench_load("market_state.json", json.dumps(HEALTHY_STATE),
                         prev_close, n)
    q_state = bench_quarantine("market_state.json", prev_close, n)

    print(f"load_alert_history  healthy   : {h_hist * 1e6:8.1f} us/call")
    print(f"load_alert_history  quarantine: {q_hist * 1e6:8.1f} us/call")
    print(f"load_previous_close healthy   : {h_state * 1e6:8.1f} us/call")
    print(f"load_previous_close quarantine: {q_state * 1e6:8.1f} us/call")
    print()
    print("All times best-of-3; quarantine path = corrupt payload restore + load")
    print("(rename syscall). The monitor runs 6x/day on files of a few KB — the")
    print("quarantine is a cold path, negligible next to the price fetch.")

    # Sanity: the quarantine path actually leaves evidence behind.
    with tempfile.TemporaryDirectory() as tmp:
        path = _patch_paths(tmp, "alert_history.json")
        path.write_text("{corrupt")
        load_alert_history()
        assert len(list(Path(tmp).glob("*.corrupt-*"))) == 1


if __name__ == "__main__":
    main()
