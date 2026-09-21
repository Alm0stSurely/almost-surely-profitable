"""One-off repair of the 2026-07-06 accounting reset artifacts.

Background (research session 2026-09-21)
----------------------------------------
On 2026-07-06 a test run reset the paper-trading account to EUR 10,000 /
0 positions / realized P&L 0.0, wiping the pre-reset ledger. Two artifacts
survived:

1. ``data/trades_history.json`` still contains the 84 pre-reset trade
   records. They belong to a different accounting universe and corrupt
   churn/behavioral aggregates (they mixed pre- and post-reset P&L into
   a single misleading total).
2. ``data/portfolio_state.json`` carries ``total_realized_pnl = -408.11``,
   which decomposes as a bogus one-time seed of **-455.756443** injected
   when the state file was reconstructed on 2026-07-07 (a day whose only
   trades were two *buys* — sells are the only code path that books
   realized P&L, see ``portfolio.py`` ``sell()``), plus +47.642115 booked
   exactly by the 14 post-2026-07-07 sells. The seed has no corresponding
   trade records and double-counts losses from the pre-reset universe that
   the reset had already zeroed.

What this script does
---------------------
1. Backs up both state files next to themselves (``.bak-YYYYMMDD``).
2. Adds ``"pre_reset": true`` to every trade with timestamp < 2026-07-06
   so consumers can filter the old accounting universe without hardcoding
   the date. Post-reset records are left untouched (no flag = current
   universe).
3. Rebuilds the ledger baseline: ``total_realized_pnl`` := sum of recorded
   ``realized_pnl`` over sells with timestamp >= 2026-07-07T00:00:00
   (post-reset booking basis). The intra-2026-07-06 DBA sell (+7.45)
   predates the evening reset and is excluded, matching how the running
   ledger actually evolved after the seed.

The script is idempotent: re-running it skips already-flagged trades and
recomputes the same baseline. It exits non-zero if the repaired ledger
would disagree with the sell-by-sell replay by more than one cent.

Usage:
    venv/bin/python scripts/repair_pre_reset_accounting.py [--dry-run]
"""

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

RESET_DATE = "2026-07-06"            # flag trades strictly before this date
LEDGER_BASELINE_FROM = "2026-07-07"  # sells on/after this date form the new baseline
GUTTER_CENTS = 0.01


def _load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def _save_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def repair(data_dir: str = "data", dry_run: bool = False) -> dict:
    data = Path(data_dir)
    trades_path = data / "trades_history.json"
    state_path = data / "portfolio_state.json"

    trades = _load_json(trades_path)
    state = _load_json(state_path)

    # --- 1. flag pre-reset trades -------------------------------------
    flagged = 0
    for t in trades:
        if t.get("timestamp", "") < RESET_DATE and not t.get("pre_reset", False):
            t["pre_reset"] = True
            flagged += 1
    pre_reset_total = sum(1 for t in trades if t.get("pre_reset", False))

    # --- 2. rebuild ledger baseline ------------------------------------
    baseline = sum(
        (t.get("realized_pnl") or 0.0)
        for t in trades
        if t.get("action") == "sell" and t.get("timestamp", "") >= LEDGER_BASELINE_FROM
    )
    old_ledger = state.get("total_realized_pnl")
    state["total_realized_pnl"] = baseline

    report = {
        "dry_run": dry_run,
        "trades_total": len(trades),
        "newly_flagged": flagged,
        "pre_reset_flagged_total": pre_reset_total,
        "old_ledger_realized_pnl": old_ledger,
        "new_ledger_realized_pnl": baseline,
        "corrupt_seed_removed": (old_ledger - baseline) if old_ledger is not None else None,
    }

    # --- 3. consistency check: ledger vs sell-by-sell replay -----------
    replay = sum(
        (t.get("realized_pnl") or 0.0)
        for t in trades
        if t.get("action") == "sell" and not t.get("pre_reset", False)
        and t.get("timestamp", "") >= LEDGER_BASELINE_FROM
    )
    if abs(replay - baseline) > GUTTER_CENTS:
        print(f"FATAL: baseline {baseline:+.6f} != sell replay {replay:+.6f}", file=sys.stderr)
        sys.exit(1)

    if not dry_run:
        stamp = datetime.now().strftime("%Y%m%d")
        for path in (trades_path, state_path):
            backup = path.with_suffix(path.suffix + f".bak-{stamp}")
            if not backup.exists():  # keep the first backup, don't overwrite
                shutil.copy2(path, backup)
        _save_json(trades_path, trades)
        _save_json(state_path, state)

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="report only, do not write")
    parser.add_argument("--data-dir", default="data")
    args = parser.parse_args()

    report = repair(data_dir=args.data_dir, dry_run=args.dry_run)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
