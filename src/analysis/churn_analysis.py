"""
Portfolio churn analysis module.

Diagnoses overtrading by analyzing round-trip profitability,
holding periods, and action flip frequency.

Usage:
    python src/analysis/churn_analysis.py

Outputs key metrics:
    - Round-trip win rate
    - Average holding period
    - Action flip count
    - P&L by holding period bucket
"""

import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from utils import _is_finite_number


# Accounting-reset boundary. On 2026-07-06 a test run reset the account to
# EUR 10,000 / 0 positions / 0 realized P&L. Trades before this date belong
# to the pre-reset accounting universe and are flagged ``pre_reset`` in
# trades_history.json (see scripts/repair_pre_reset_accounting.py). Sells on
# or after this datetime form the post-reset ledger baseline.
RESET_BOUNDARY = datetime(2026, 7, 7)
_LEDGER_RECONCILE_TOLERANCE = 1.00  # euros


@dataclass
class RoundTrip:
    ticker: str
    buy_date: datetime
    sell_date: datetime
    hold_days: float
    pnl: float
    buy_price: float
    sell_price: float


def _is_valid_round_trip(rt: RoundTrip) -> bool:
    """Return True only when every numeric field is a finite scalar."""
    return (
        _is_finite_number(rt.pnl)
        and _is_finite_number(rt.hold_days)
        and _is_finite_number(rt.buy_price)
        and _is_finite_number(rt.sell_price)
    )


def _filter_valid_round_trips(round_trips: List[RoundTrip]) -> List[RoundTrip]:
    """Drop round trips whose numeric fields are not finite."""
    return [rt for rt in round_trips if _is_valid_round_trip(rt)]


def _safe_value_str(value, symbol: str = "", fmt: str = ".2f", default: str = "n/a") -> str:
    """Format a finite scalar, falling back to *default*."""
    if _is_finite_number(value):
        return f"{symbol}{value:{fmt}}"
    return default


def _safe_pct_str(value, fmt: str = ".1f", default: str = "n/a") -> str:
    """Format a finite scalar as a percentage, falling back to *default*."""
    if _is_finite_number(value):
        return f"{value:{fmt}}%"
    return default


def load_trades(data_dir: str = "data") -> List[Dict]:
    """Load trade history from JSON."""
    path = Path(data_dir) / "trades_history.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_decisions(data_dir: str = "data") -> List[Dict]:
    """Load decision history from JSON."""
    path = Path(data_dir) / "decision_history.json"
    if not path.exists():
        return []
    with open(path) as f:
        return json.load(f)


def load_ledger_realized_pnl(data_dir: str = "data") -> float:
    """Load the portfolio ledger's cumulative realized P&L.

    Returns NaN when the ledger file or the value is missing/non-finite so
    callers can render ``n/a`` via the standard safe-formatting helpers.
    """
    path = Path(data_dir) / "portfolio_state.json"
    if not path.exists():
        return float("nan")
    with open(path) as f:
        state = json.load(f)
    value = state.get("total_realized_pnl")
    return float(value) if _is_finite_number(value) else float("nan")


def _parse_trade_timestamp(t: Dict) -> datetime:
    """Parse trade timestamp, falling back to the ISO date string prefix."""
    raw_ts = t.get("timestamp", "")
    try:
        return datetime.fromisoformat(raw_ts)
    except (ValueError, TypeError):
        try:
            # Fallback: some trade records may only have a date string.
            return datetime.fromisoformat(raw_ts[:10]) if raw_ts else datetime.min
        except (ValueError, TypeError):
            return datetime.min


def match_round_trips(trades: List[Dict]) -> List[RoundTrip]:
    """Match buys and sells into round trips using FIFO.

    Trades are sorted by timestamp within each ticker so the FIFO order is
    deterministic even if trades_history.json is not perfectly ordered. A sell
    is only matched with a buy that occurred before it; orphan sells without a
    preceding buy are skipped.
    """
    from collections import deque

    ticker_trades = defaultdict(list)
    for t in trades:
        if not _is_finite_number(t.get("price")):
            continue
        ticker_trades[t["ticker"]].append(t)

    round_trips = []
    for tk, tl in ticker_trades.items():
        buys = deque(sorted([t for t in tl if t["action"] == "buy"], key=_parse_trade_timestamp))
        sells = sorted([t for t in tl if t["action"] == "sell"], key=_parse_trade_timestamp)

        for sell in sells:
            sell_dt = _parse_trade_timestamp(sell)
            # Discard buys that are strictly after this sell (they cannot be matched).
            while buys and _parse_trade_timestamp(buys[0]) > sell_dt:
                buys.popleft()
            if not buys:
                continue
            buy = buys.popleft()
            buy_dt = _parse_trade_timestamp(buy)
            hold_days = (sell_dt - buy_dt).total_seconds() / 86400

            pnl = sell.get("realized_pnl", 0)
            if not _is_finite_number(pnl):
                pnl = float("nan")

            round_trips.append(RoundTrip(
                ticker=tk,
                buy_date=buy_dt,
                sell_date=sell_dt,
                hold_days=hold_days,
                pnl=pnl,
                buy_price=buy["price"],
                sell_price=sell["price"],
            ))

    return _filter_valid_round_trips(round_trips)


def _bucket_metrics(round_trips: List[RoundTrip]) -> Dict:
    """Compute bucketed churn metrics from a list of round trips.

    Non-finite round trips are excluded from every aggregate so that a single
    corrupt record cannot poison totals, win rates, or holding-period averages.
    """
    valid_round_trips = _filter_valid_round_trips(round_trips)
    winning = [rt for rt in valid_round_trips if rt.pnl > 0]
    short = [rt for rt in valid_round_trips if rt.hold_days <= 3]
    medium = [rt for rt in valid_round_trips if 3 < rt.hold_days <= 14]
    long = [rt for rt in valid_round_trips if rt.hold_days > 14]
    total = len(valid_round_trips)
    return {
        "total_round_trips": total,
        "winning_round_trips": len(winning),
        "win_rate_pct": (len(winning) / max(total, 1)) * 100,
        "total_realized_pnl": sum(rt.pnl for rt in valid_round_trips),
        "avg_hold_days": sum(rt.hold_days for rt in valid_round_trips) / max(total, 1),
        "short_term_count": len(short),
        "short_term_win_rate": (len([r for r in short if r.pnl > 0]) / max(len(short), 1)) * 100,
        "short_term_pnl": sum(r.pnl for r in short),
        "medium_term_count": len(medium),
        "medium_term_win_rate": (len([r for r in medium if r.pnl > 0]) / max(len(medium), 1)) * 100,
        "medium_term_pnl": sum(r.pnl for r in medium),
        "long_term_count": len(long),
        "long_term_win_rate": (len([r for r in long if r.pnl > 0]) / max(len(long), 1)) * 100,
        "long_term_pnl": sum(r.pnl for r in long),
    }


def analyze_cohort(trades: List[Dict], cutoff: datetime) -> Tuple[Dict, Dict]:
    """Compute pre/post cutoff churn metrics to isolate regime changes.

    A round trip is attributed to a cohort based on its *entry* (buy) date.
    This avoids the artefact where a post-cutoff sell is matched against a
    pre-cutoff buy, which would otherwise produce negative holding periods.
    """
    all_round_trips = match_round_trips(trades)
    pre_rts = [rt for rt in all_round_trips if rt.buy_date < cutoff]
    post_rts = [rt for rt in all_round_trips if rt.buy_date >= cutoff]
    pre_metrics = _bucket_metrics(pre_rts)
    post_metrics = _bucket_metrics(post_rts)
    # Add activity context
    first_dt = _parse_trade_timestamp(trades[0]) if trades else datetime.min
    pre_days = max((cutoff - first_dt).days, 1)
    post_days = max((datetime.now() - cutoff).days, 1)
    pre_metrics["trades_per_year"] = len([t for t in trades if _parse_trade_timestamp(t) < cutoff]) * 365 / pre_days
    post_metrics["trades_per_year"] = len([t for t in trades if _parse_trade_timestamp(t) >= cutoff]) * 365 / post_days
    return pre_metrics, post_metrics


def analyze_churn(round_trips: List[RoundTrip], trades: List[Dict], decisions: List[Dict]) -> Dict:
    """Compute churn metrics."""
    metrics = _bucket_metrics(round_trips)

    # Action flips
    ticker_decisions = defaultdict(list)
    for d in decisions:
        date = d["timestamp"][:10]
        for a in d.get("actions", []):
            if a["action"] != "hold":
                ticker_decisions[a["ticker"]].append((date, a["action"]))

    flips = 0
    for acts in ticker_decisions.values():
        for i in range(1, len(acts)):
            if acts[i][1] != acts[i - 1][1]:
                flips += 1

    # Date range
    if decisions:
        first = datetime.fromisoformat(decisions[0]["timestamp"])
        last = datetime.fromisoformat(decisions[-1]["timestamp"])
        days_active = (last - first).days
    else:
        days_active = 1

    metrics.update({
        "losing_round_trips": metrics["total_round_trips"] - metrics["winning_round_trips"],
        "action_flips": flips,
        "days_active": days_active,
        "trades_per_week": len(trades) / max(days_active / 7, 1),
        "annualized_turnover": len(trades) * 365 / max(days_active, 1),
    })
    return metrics


def print_report(metrics: Dict):
    """Print formatted churn report.

    Every formatted numeric field is guarded so that non-finite values are
    rendered as ``n/a`` rather than leaking ``nan``/``inf`` tokens into the
    report output.
    """
    print("=" * 60)
    print("PORTFOLIO CHURN ANALYSIS")
    print("=" * 60)
    win_rate = metrics.get("win_rate_pct")
    lose_rate = 100 - win_rate if _is_finite_number(win_rate) else float("nan")
    print(f"\nRound Trips:          {metrics.get('total_round_trips', 'n/a')}")
    print(f"Winning:              {metrics.get('winning_round_trips', 'n/a')} ({_safe_pct_str(win_rate)})")
    print(f"Losing:               {metrics.get('losing_round_trips', 'n/a')} ({_safe_pct_str(lose_rate)})")
    print(f"Total Realized P&L:   {_safe_value_str(metrics.get('total_realized_pnl'), symbol='€', fmt='+.2f')}")
    print(f"Avg Holding Period:   {_safe_value_str(metrics.get('avg_hold_days'), fmt='.1f', default='n/a')} days")
    print(f"\n--- Holding Period Breakdown ---")
    print(f"Short (≤3d):          {metrics.get('short_term_count', 'n/a')} trips, "
          f"win rate {_safe_pct_str(metrics.get('short_term_win_rate'))}, "
          f"P&L {_safe_value_str(metrics.get('short_term_pnl'), symbol='€', fmt='+.2f')}")
    print(f"Medium (4-14d):       {metrics.get('medium_term_count', 'n/a')} trips, "
          f"win rate {_safe_pct_str(metrics.get('medium_term_win_rate'))}, "
          f"P&L {_safe_value_str(metrics.get('medium_term_pnl'), symbol='€', fmt='+.2f')}")
    print(f"Long (>14d):          {metrics.get('long_term_count', 'n/a')} trips, "
          f"win rate {_safe_pct_str(metrics.get('long_term_win_rate'))}, "
          f"P&L {_safe_value_str(metrics.get('long_term_pnl'), symbol='€', fmt='+.2f')}")
    print(f"\n--- Activity Metrics ---")
    print(f"Action Flips:         {metrics.get('action_flips', 'n/a')}")
    print(f"Trades/Week:          {_safe_value_str(metrics.get('trades_per_week'), fmt='.1f')}")
    print(f"Annualized Turnover:  {_safe_value_str(metrics.get('annualized_turnover'), fmt='.0f', default='n/a')} trades/year")
    print("=" * 60)


def reconcile_ledger_since_reset(trades: List[Dict], ledger_realized: float) -> Dict:
    """Reconcile the portfolio ledger against the post-reset sell replay.

    The ledger books realized P&L per sell (``portfolio.py:sell()``), so the
    post-reset ledger total must equal the sum of recorded ``realized_pnl``
    over sells at or after :data:`RESET_BOUNDARY`. The pre-reset universe is
    a known historical artifact (flagged trades, reset without compensating
    records) and is reported separately, never reconciled.

    Returns a dict with the booked sum, the gap, and the pre-reset flag
    counts; ``gap_is_warning`` is True when the post-reset disagreement
    exceeds ``_LEDGER_RECONCILE_TOLERANCE``.
    """
    booked = 0.0
    booked_count = 0
    pre_reset_sells = 0
    for t in trades:
        if t.get("action") != "sell":
            continue
        pnl = t.get("realized_pnl")
        if not _is_finite_number(pnl):
            continue
        if _parse_trade_timestamp(t) >= RESET_BOUNDARY:
            booked += pnl
            booked_count += 1
        elif t.get("pre_reset", False):
            pre_reset_sells += 1

    gap = ledger_realized - booked if _is_finite_number(ledger_realized) else float("nan")
    return {
        "booked_since_reset": booked,
        "booked_sells": booked_count,
        "ledger_realized": ledger_realized,
        "gap": gap,
        "gap_is_warning": _is_finite_number(gap) and abs(gap) > _LEDGER_RECONCILE_TOLERANCE,
        "pre_reset_flagged_sells": pre_reset_sells,
        "pre_reset_flagged_trades": sum(1 for t in trades if t.get("pre_reset", False)),
    }


def main():
    trades = load_trades()
    decisions = load_decisions()
    round_trips = match_round_trips(trades)
    metrics = analyze_churn(round_trips, trades, decisions)
    print_report(metrics)

    # Cohort analysis: pre/post cooldown integration (2026-06-18)
    cutoff = datetime(2026, 6, 18)
    pre, post = analyze_cohort(trades, cutoff)
    print(f"\n--- Pre/Post {cutoff.date()} Cohort ---")
    print(f"Pre:  {pre.get('total_round_trips', 'n/a')} RT, "
          f"win {_safe_pct_str(pre.get('win_rate_pct'))}, "
          f"avg hold {_safe_value_str(pre.get('avg_hold_days'), fmt='.1f')}d, "
          f"{_safe_value_str(pre.get('trades_per_year'), fmt='.0f')} trades/yr")
    print(f"Post: {post.get('total_round_trips', 'n/a')} RT, "
          f"win {_safe_pct_str(post.get('win_rate_pct'))}, "
          f"avg hold {_safe_value_str(post.get('avg_hold_days'), fmt='.1f')}d, "
          f"{_safe_value_str(post.get('trades_per_year'), fmt='.0f')} trades/yr")

    # Accounting-reset cohort: round trips attributed by ENTRY date relative
    # to the 2026-07-06/07 reset. The post-reset cohort is the clean,
    # authoritative universe; the pre-reset cohort is a historical artifact
    # whose P&L fields mix two accounting universes (indicative only).
    reset_pre, reset_post = analyze_cohort(trades, RESET_BOUNDARY)
    print(f"\n--- Accounting Reset {RESET_BOUNDARY.date()} Cohort ---")
    print(f"Pre-reset (artifact): {reset_pre.get('total_round_trips', 'n/a')} RT, "
          f"win {_safe_pct_str(reset_pre.get('win_rate_pct'))}, "
          f"P&L {_safe_value_str(reset_pre.get('total_realized_pnl'), symbol='€', fmt='+.2f')} "
          f"(indicative only)")
    print(f"Post-reset (clean):   {reset_post.get('total_round_trips', 'n/a')} RT, "
          f"win {_safe_pct_str(reset_post.get('win_rate_pct'))}, "
          f"avg hold {_safe_value_str(reset_post.get('avg_hold_days'), fmt='.1f')}d, "
          f"P&L {_safe_value_str(reset_post.get('total_realized_pnl'), symbol='€', fmt='+.2f')}")

    # Ledger reconciliation on the post-reset booking basis: the ledger must
    # equal the sum of recorded realized_pnl over sells since the boundary.
    # Pre-reset records are flagged and reported, never reconciled.
    ledger_realized = load_ledger_realized_pnl()
    recon = reconcile_ledger_since_reset(trades, ledger_realized)
    print(f"\n--- Ledger Reconciliation (since {RESET_BOUNDARY.date()}) ---")
    print(f"Booked by sells since reset ({recon['booked_sells']} sells): "
          f"{_safe_value_str(recon['booked_since_reset'], symbol='€', fmt='+.2f')}")
    print(f"Portfolio ledger realized P&L:                "
          f"{_safe_value_str(recon['ledger_realized'], symbol='€', fmt='+.2f')}")
    if recon["gap_is_warning"]:
        print(f"⚠ Gap {_safe_value_str(recon['gap'], symbol='€', fmt='+.2f')}: "
              f"post-reset sell replay and ledger disagree — investigate state corruption.")
    else:
        print(f"✓ Ledger consistent with sell-by-sell replay "
              f"(gap {_safe_value_str(recon['gap'], symbol='€', fmt='+.2f')}).")
    if recon["pre_reset_flagged_trades"]:
        print(f"Pre-reset flagged trades: {recon['pre_reset_flagged_trades']} "
              f"({recon['pre_reset_flagged_sells']} sells) — excluded from reconciliation "
              f"(2026-07-06 reset artifact).")
    print("=" * 60)


if __name__ == "__main__":
    main()
