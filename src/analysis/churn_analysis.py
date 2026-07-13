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
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple


@dataclass
class RoundTrip:
    ticker: str
    buy_date: datetime
    sell_date: datetime
    hold_days: float
    pnl: float
    buy_price: float
    sell_price: float


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

            round_trips.append(RoundTrip(
                ticker=tk,
                buy_date=buy_dt,
                sell_date=sell_dt,
                hold_days=hold_days,
                pnl=sell.get("realized_pnl", 0),
                buy_price=buy["price"],
                sell_price=sell["price"],
            ))

    return round_trips


def _bucket_metrics(round_trips: List[RoundTrip]) -> Dict:
    """Compute bucketed churn metrics from a list of round trips."""
    winning = [rt for rt in round_trips if rt.pnl > 0]
    short = [rt for rt in round_trips if rt.hold_days <= 3]
    medium = [rt for rt in round_trips if 3 < rt.hold_days <= 14]
    long = [rt for rt in round_trips if rt.hold_days > 14]
    return {
        "total_round_trips": len(round_trips),
        "winning_round_trips": len(winning),
        "win_rate_pct": (len(winning) / max(len(round_trips), 1)) * 100,
        "total_realized_pnl": sum(rt.pnl for rt in round_trips),
        "avg_hold_days": sum(rt.hold_days for rt in round_trips) / max(len(round_trips), 1),
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
    """Print formatted churn report."""
    print("=" * 60)
    print("PORTFOLIO CHURN ANALYSIS")
    print("=" * 60)
    print(f"\nRound Trips:          {metrics['total_round_trips']}")
    print(f"Winning:              {metrics['winning_round_trips']} ({metrics['win_rate_pct']:.1f}%)")
    print(f"Losing:               {metrics['losing_round_trips']} ({100 - metrics['win_rate_pct']:.1f}%)")
    print(f"Total Realized P&L:   €{metrics['total_realized_pnl']:+.2f}")
    print(f"Avg Holding Period:   {metrics['avg_hold_days']:.1f} days")
    print(f"\n--- Holding Period Breakdown ---")
    print(f"Short (≤3d):          {metrics['short_term_count']} trips, "
          f"win rate {metrics['short_term_win_rate']:.1f}%, P&L €{metrics['short_term_pnl']:+.2f}")
    print(f"Medium (4-14d):       {metrics['medium_term_count']} trips, "
          f"win rate {metrics['medium_term_win_rate']:.1f}%, P&L €{metrics['medium_term_pnl']:+.2f}")
    print(f"Long (>14d):          {metrics['long_term_count']} trips, "
          f"win rate {metrics['long_term_win_rate']:.1f}%, P&L €{metrics['long_term_pnl']:+.2f}")
    print(f"\n--- Activity Metrics ---")
    print(f"Action Flips:         {metrics['action_flips']}")
    print(f"Trades/Week:          {metrics['trades_per_week']:.1f}")
    print(f"Annualized Turnover:  {metrics['annualized_turnover']:.0f} trades/year")
    print("=" * 60)


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
    print(f"Pre:  {pre['total_round_trips']} RT, win {pre['win_rate_pct']:.1f}%, "
          f"avg hold {pre['avg_hold_days']:.1f}d, {pre['trades_per_year']:.0f} trades/yr")
    print(f"Post: {post['total_round_trips']} RT, win {post['win_rate_pct']:.1f}%, "
          f"avg hold {post['avg_hold_days']:.1f}d, {post['trades_per_year']:.0f} trades/yr")
    print("=" * 60)


if __name__ == "__main__":
    main()
