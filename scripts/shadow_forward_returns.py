#!/usr/bin/env python3
"""Shadow-arm forward-return tracker (H4 Phase 2).

Computes 5-trading-day forward returns for shadow-arm BUY actions in
``data/shadow_decisions.json`` and persists them back onto each action as
``forward_return_5d`` / ``forward_return_computed_at``.

Convention (fixed 2026-11-10, supersedes the ad-hoc 2026-11-09 manual pass
whose entry basis was inconsistent across tickers — see
memory/research-2026-11-10.md):

    entry = close on the decision date            (live arm executes at the
                                                   decision-date close, so the
                                                   shadow arm mirrors it)
    exit  = close 5 trading sessions after entry  (entry day counts as day 0)

An action becomes observable once 6 daily bars exist from the entry bar
(entry + 5 forward sessions). Until then it is reported as pending with the
number of elapsed sessions.

Usage:
    .venv/bin/python scripts/shadow_forward_returns.py            # compute + update
    .venv/bin/python scripts/shadow_forward_returns.py --check    # report only, no write

Idempotent: actions already carrying ``forward_return_5d`` are skipped unless
``--force`` is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import yfinance as yf

REPO_ROOT = Path(__file__).resolve().parent.parent
SHADOW_PATH = REPO_ROOT / "data" / "shadow_decisions.json"
FORWARD_SESSIONS = 5


def load_records() -> list[dict]:
    with SHADOW_PATH.open() as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    for key in ("records", "decisions"):
        if key in data:
            return data[key]
    raise ValueError(f"Unrecognized shadow_decisions.json schema: {list(data)}")


def fetch_closes(ticker: str, start: date, end: date) -> dict[date, float]:
    """Return {session_date: close} for *ticker* over [start, end]."""
    df = yf.download(
        ticker,
        start=start.isoformat(),
        end=(end + __import__("datetime").timedelta(days=1)).isoformat(),
        progress=False,
        auto_adjust=False,
    )
    if df.empty:
        return {}
    closes = df["Close"]
    if hasattr(closes, "iloc"):  # yfinance >= 0.2 multi-index safety
        closes = closes.iloc[:, 0] if getattr(closes, "ndim", 1) == 2 else closes
    out = {}
    for idx, val in closes.items():
        d = idx.date() if hasattr(idx, "date") else idx
        try:
            out[d] = float(val)
        except (TypeError, ValueError):
            continue
    return out


def compute_forward_return(closes: dict[date, float], decision_date: date) -> tuple[float | None, int, str]:
    """Return (forward_return, elapsed_sessions, status).

    status is one of: "computed", "pending", "no_entry_bar".
    """
    sessions = sorted(d for d in closes if d >= decision_date)
    if not sessions or sessions[0] != decision_date:
        return None, 0, "no_entry_bar"
    entry_price = closes[decision_date]
    if entry_price <= 0:
        return None, 0, "no_entry_bar"
    elapsed = len(sessions) - 1
    if elapsed < FORWARD_SESSIONS:
        return None, elapsed, "pending"
    exit_date = sessions[FORWARD_SESSIONS]
    exit_price = closes[exit_date]
    if exit_price <= 0:
        return None, elapsed, "no_entry_bar"
    return exit_price / entry_price - 1.0, FORWARD_SESSIONS, "computed"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="report only, do not write")
    parser.add_argument("--force", action="store_true", help="recompute even if already present")
    args = parser.parse_args()

    records = load_records()
    today = datetime.now(timezone.utc).date()

    # Collect the distinct (ticker, decision_date) pairs needing computation.
    pending_jobs: list[tuple[str, date, dict]] = []  # (ticker, date, action dict)
    for rec in records:
        rec_date = date.fromisoformat(rec["date"])
        for action in rec.get("actions", []):
            if action.get("action") != "buy":
                continue
            if action.get("forward_return_5d") is not None and not args.force:
                continue
            pending_jobs.append((action["ticker"], rec_date, action))

    # Fetch each ticker once over the union window.
    closes_by_ticker: dict[str, dict[date, float]] = {}
    tickers = sorted({t for t, _, _ in pending_jobs})
    for ticker in tickers:
        earliest = min(d for t, d, _ in pending_jobs if t == ticker)
        closes_by_ticker[ticker] = fetch_closes(ticker, earliest, today)

    computed = 0
    pending = 0
    missing = 0
    for ticker, rec_date, action in pending_jobs:
        closes = closes_by_ticker.get(ticker, {})
        fwd, elapsed, status = compute_forward_return(closes, rec_date)
        if status == "computed":
            action["forward_return_5d"] = round(fwd, 6)
            action["forward_return_computed_at"] = today.isoformat()
            computed += 1
        elif status == "pending":
            pending += 1
        else:
            missing += 1

    # Report table.
    print(f"{'date':<12} {'ticker':<10} {'fwd_5d':>9} {'computed_at':<12} status")
    n_obs = 0
    returns: list[float] = []
    for rec in records:
        for action in rec.get("actions", []):
            if action.get("action") != "buy":
                continue
            fwd = action.get("forward_return_5d")
            comp = action.get("forward_return_computed_at") or "-"
            if fwd is None:
                status = "pending"
            else:
                status = "done"
                returns.append(fwd)
                n_obs += 1
            fwd_s = f"{fwd * 100:+.2f}%" if fwd is not None else "-"
            print(f"{rec['date']:<12} {action['ticker']:<10} {fwd_s:>9} {comp:<12} {status}")

    if returns:
        mean_r = sum(returns) / len(returns)
        pos = sum(1 for r in returns if r > 0)
        print(
            f"\nPhase 2: n={n_obs}, mean {mean_r * 100:+.2f}%, "
            f"positive rate {pos}/{n_obs} ({pos / n_obs:.0%})"
        )
    print(f"\nnewly computed: {computed}, pending: {pending}, no_entry_bar: {missing}")

    if not args.check and computed:
        with SHADOW_PATH.open("w") as f:
            json.dump(records, f, indent=2)
        print(f"updated {SHADOW_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
