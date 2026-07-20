"""Behavioral analysis of LLM decision history for trading research."""
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_valid_daily_results

DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"
RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "results"
OUTPUT_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "analysis"


# Keyword concepts used for behavioral analysis. Multi-variant matching is required
# because the LLM paraphrases guardrails (e.g. "weekly trade cap", "trade limit",
# "trades used" all refer to the same cooldown constraint).
KEYWORD_CONCEPTS = {
    "loss aversion": ["loss aversion", "loss-aversion"],
    "CVaR": ["cvar", "conditional value at risk", "expected shortfall"],
    "cash buffer": ["cash buffer", "cash target", "cash position", "cash level", "cash allocation"],
    "mean reversion": ["mean reversion", "mean-reversion", "revert to mean", "reverting"],
    "overbought": ["overbought", "over-bought"],
    "oversold": ["oversold", "over-sold"],
    "tail risk": ["tail risk", "tail-risk"],
    "correlation": ["correlation", "correlated", "correlations"],
    "diversification": ["diversification", "diversified"],
    "prospect theory": ["prospect theory"],
    "regime": ["regime", "volatility regime", "market regime"],
    "momentum": ["momentum", "trend", "trending"],
    "drawdown": ["drawdown", "max drawdown"],
    "let winners run": ["let winners run", "winners run", "ride winners", "run winners"],
    "trade cap": ["trade cap", "weekly trade", "weekly cap", "trade limit", "trades used", "trades remaining", "weekly trade cap"],
    "stop-loss": ["stop-loss", "stop loss", "stoploss"],
    "cooldown": ["cooldown", "cool-down", "holding period", "mandatory hold", "flip cooldown"],
}


def count_keyword_concepts(decisions, keyword_concepts=None):
    """Count how many decisions mention each behavioral keyword concept.

    Each decision is counted at most once per concept. Matching is
    case-insensitive and supports multiple variants per concept so that
    paraphrases are captured (e.g. "weekly trade cap" and "trade limit").
    """
    if keyword_concepts is None:
        keyword_concepts = KEYWORD_CONCEPTS
    keyword_counts = {concept: 0 for concept in keyword_concepts}
    for d in decisions:
        r = d.get("reasoning", "").lower()
        for concept, variants in keyword_concepts.items():
            if any(variant in r for variant in variants):
                keyword_counts[concept] += 1
    return keyword_counts

def main():
    with open(DATA_DIR / "decision_history.json") as f:
        decisions = json.load(f)
    with open(DATA_DIR / "trades_history.json") as f:
        trades = json.load(f)

    valid = [d for d in decisions if not d.get("error", False)]
    errors = [d for d in decisions if d.get("error", False)]

    lines = [
        "=" * 60,
        "BEHAVIORAL ANALYSIS OF LLM DECISIONS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
        "SAMPLE OVERVIEW",
        f"Total decisions: {len(decisions)}",
        f"Valid decisions: {len(valid)}",
        f"Errors: {len(errors)} ({len(errors)/max(len(decisions),1)*100:.1f}%)",
        "",
    ]

    # Error rate by month
    monthly = defaultdict(lambda: {"total": 0, "errors": 0})
    for d in decisions:
        month = d["timestamp"][:7]
        monthly[month]["total"] += 1
        monthly[month]["errors"] += 1 if d.get("error", False) else 0

    lines.append("ERROR RATE EVOLUTION")
    for month in sorted(monthly.keys()):
        stats = monthly[month]
        err_rate = stats["errors"] / stats["total"] * 100
        lines.append(f"{month}: {stats['errors']:3d}/{stats['total']:3d} errors ({err_rate:5.1f}%)")
    lines.append("")

    # Error day-of-week pattern
    dow_counts = defaultdict(lambda: {"total": 0, "errors": 0})
    for d in decisions:
        dt = datetime.fromisoformat(d["timestamp"])
        dow = dt.strftime("%A")
        dow_counts[dow]["total"] += 1
        dow_counts[dow]["errors"] += 1 if d.get("error", False) else 0
    lines.append("ERROR RATE BY DAY OF WEEK")
    for dow in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]:
        if dow in dow_counts:
            stats = dow_counts[dow]
            err_rate = stats["errors"] / stats["total"] * 100
            lines.append(f"{dow:10s}: {stats['errors']:3d}/{stats['total']:3d} errors ({err_rate:5.1f}%)")
    lines.append("")

    # Action distribution
    actions_count = Counter()
    for d in valid:
        for a in d.get("actions", []):
            actions_count[a.get("action", "unknown")] += 1
    total_actions = sum(actions_count.values())
    lines.append("ACTION DISTRIBUTION")
    lines.append(f"Total actions: {total_actions} ({total_actions/max(len(valid),1):.1f} per decision)")
    for action, count in actions_count.most_common():
        pct = count / max(total_actions, 1) * 100
        lines.append(f"{action:6s}: {count:4d} ({pct:5.1f}% of actions)")
    lines.append("")

    # Keyword frequency
    keyword_concepts = KEYWORD_CONCEPTS
    keyword_counts = count_keyword_concepts(valid, keyword_concepts)

    lines.append("BEHAVIORAL KEYWORD FREQUENCY")
    for concept, count in sorted(keyword_counts.items(), key=lambda x: -x[1]):
        pct = count / max(len(valid), 1) * 100
        lines.append(f"{concept:20s}: {count:4d} ({pct:5.1f}%)")
    lines.append("")

    # Cash creep from daily results (skip dry-run/test artifacts)
    daily_results = load_valid_daily_results(str(RESULTS_DIR / "daily"))[-20:]
    lines.append("CASH LEVELS (recent 20 daily results)")
    lines.append(f"{'Date':12s} {'Cash %':>8s} {'Positions':>10s} {'Total Return':>14s} {'Trades':>7s}")
    for data in daily_results:
        pa = data.get("portfolio_after", {})
        cash = pa.get("cash", 0)
        total = pa.get("total_value", 1)
        num_pos = pa.get("num_positions", 0)
        ret = pa.get("total_return_pct", 0)
        trades_count = len(data.get("executed_trades", []))
        lines.append(f"{data['date']:12s} {cash/total*100:8.1f} {num_pos:10d} {ret:14.2f} {trades_count:7d}")
    lines.append("")

    # Round-trip churn analysis
    ticker_trades = defaultdict(list)
    for t in trades:
        ticker_trades[t["ticker"]].append(t)

    round_trips = []
    for tk, tl in ticker_trades.items():
        buys = [t for t in tl if t["action"] == "buy"]
        sells = [t for t in tl if t["action"] == "sell"]
        buy_idx = 0
        for sell in sells:
            if buy_idx < len(buys):
                buy_dt = datetime.fromisoformat(buys[buy_idx]["timestamp"])
                sell_dt = datetime.fromisoformat(sell["timestamp"])
                hold_days = (sell_dt - buy_dt).total_seconds() / 86400
                round_trips.append({
                    "ticker": tk,
                    "hold_days": hold_days,
                    "pnl": sell.get("realized_pnl", 0),
                })
                buy_idx += 1

    winning = [r for r in round_trips if r["pnl"] > 0]
    short = [r for r in round_trips if r["hold_days"] <= 3]
    medium = [r for r in round_trips if 3 < r["hold_days"] <= 14]
    long = [r for r in round_trips if r["hold_days"] > 14]

    first = datetime.fromisoformat(decisions[0]["timestamp"])
    last = datetime.fromisoformat(decisions[-1]["timestamp"])
    days = max((last - first).days, 1)

    lines.append("CHURN / ROUND-TRIP ANALYSIS")
    lines.append(f"Round trips: {len(round_trips)}")
    lines.append(f"Win rate: {len(winning)/max(len(round_trips),1)*100:.1f}%")
    lines.append(f"Avg hold period: {sum(r['hold_days'] for r in round_trips)/max(len(round_trips),1):.1f} days")
    lines.append(f"Short holds (≤3d): {len(short)}, win rate {len([r for r in short if r['pnl']>0])/max(len(short),1)*100:.1f}%")
    lines.append(f"Medium holds (4-14d): {len(medium)}, win rate {len([r for r in medium if r['pnl']>0])/max(len(medium),1)*100:.1f}%")
    lines.append(f"Long holds (>14d): {len(long)}, win rate {len([r for r in long if r['pnl']>0])/max(len(long),1)*100:.1f}%")
    lines.append(f"Annualized turnover: {len(trades) * 365 / days:.0f} trades/year")
    lines.append("")
    lines.append("=" * 60)

    report = "\n".join(lines)
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"behavioral_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")

if __name__ == "__main__":
    main()
