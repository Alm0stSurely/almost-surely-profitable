"""Behavioral analysis of LLM decision history for trading research."""
import math
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.churn_analysis import match_round_trips
from utils import load_json_list_or_quarantine, load_valid_daily_results


def _safe_cash_pct(cash, total):
    """Return cash as a percentage of total, or None when the ratio is undefined.

    Guards against zero/negative totals, missing values, and non-finite
    inputs that can leak from degenerate daily result files. Booleans are
    rejected (``bool`` is a subclass of ``int``): a JSON ``true`` must not
    be read as a cash or total value — same convention as
    ``utils._is_finite_number``.
    """
    if (
        not isinstance(cash, (int, float))
        or isinstance(cash, bool)
        or not isinstance(total, (int, float))
        or isinstance(total, bool)
    ):
        return None
    if not math.isfinite(cash) or not math.isfinite(total) or total <= 0:
        return None
    return cash / total * 100


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
    "deflated sharpe": ["deflated sharpe", "dsr", "sharpe ratio"],
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

def get_round_trips(trades):
    """Return round-trip dicts using the same FIFO matcher as churn_analysis.py."""
    return [
        {"ticker": rt.ticker, "hold_days": rt.hold_days, "pnl": rt.pnl}
        for rt in match_round_trips(trades)
    ]


def _format_cash_levels(daily_results):
    """Format the cash-levels table, guarding against degenerate portfolio values."""
    lines = [
        "CASH LEVELS (recent 20 daily results)",
        f"{'Date':12s} {'Cash %':>8s} {'Positions':>10s} {'Total Return':>14s} {'Trades':>7s}",
    ]
    for data in daily_results:
        pa = data.get("portfolio_after", {})
        cash = pa.get("cash", 0)
        total = pa.get("total_value", 1)
        num_pos = pa.get("num_positions", 0)
        ret = pa.get("total_return_pct", 0)
        trades_count = len(data.get("executed_trades", []))
        cash_pct = _safe_cash_pct(cash, total)
        cash_str = f"{cash_pct:8.1f}" if cash_pct is not None else "     n/a"
        lines.append(f"{data['date']:12s} {cash_str} {num_pos:10d} {ret:14.2f} {trades_count:7d}")
    lines.append("")
    return lines


def _load_analysis_histories():
    """Load decision history + trade ledger through the shared quarantine ingress.

    Both ledgers are append-then-overwrite files whose writers already load
    through the same helper (``save_decision`` / ``save_trade``), so the
    analysis reader shares one trust model with them: a corrupt file is
    quarantined (bytes preserved, one ERROR log) instead of crashing the run,
    being silently ingested as non-standard tokens no writer can emit (PR #84
    strict hook), or sitting in place for the next save to overwrite. Records
    that parse but violate the element contract drop loudly one-per-record —
    one malformed record must not void n-1 valid siblings (anti-Lipschitz,
    PR #64 doctrine).

    Returns ``(decisions, trades, warnings)``; *warnings* are degradation
    notes the caller must embed in the report artifact — the report is what a
    human reads, and a silently degraded report is the failure mode this
    hardening exists to prevent (PR #87 doctrine).
    """
    warnings = []

    decisions_path = DATA_DIR / "decision_history.json"
    decisions = []
    if decisions_path.exists():
        history = load_json_list_or_quarantine(
            decisions_path, context="decision history")
        if not decisions_path.exists():
            # The helper quarantined the file (rename to .corrupt-*, bytes
            # preserved, ERROR logged).
            warnings.append(
                f"⚠ Decision history {decisions_path.name} was corrupt and has "
                "been quarantined; decision-based sections reflect an empty history."
            )
        dropped = 0
        for idx, d in enumerate(history):
            # Element guard wraps the whole element evaluation (PR #86 rule):
            # every downstream section needs mapping access plus a naive,
            # ISO-parseable timestamp — the monthly histogram slices it, the
            # day-of-week section parses it, the churn header subtracts two
            # of them (a tz-aware record would parse then detonate at the
            # subtraction, one operation downstream of a successful parse).
            if not isinstance(d, dict):
                print(f"⚠ Dropping malformed decision record at index {idx} in "
                      f"{decisions_path.name} (expected dict, got "
                      f"{type(d).__name__})")
                dropped += 1
                continue
            ts = d.get("timestamp")
            if not isinstance(ts, str):
                print(f"⚠ Dropping malformed decision record at index {idx} in "
                      f"{decisions_path.name} (missing/non-string timestamp: "
                      f"{ts!r})")
                dropped += 1
                continue
            try:
                parsed = datetime.fromisoformat(ts)
            except ValueError:
                print(f"⚠ Dropping malformed decision record at index {idx} in "
                      f"{decisions_path.name} (unparseable timestamp: {ts!r})")
                dropped += 1
                continue
            if parsed.tzinfo is not None:
                print(f"⚠ Dropping malformed decision record at index {idx} in "
                      f"{decisions_path.name} (timezone-aware timestamp in "
                      f"naive-timestamp ledger: {ts!r})")
                dropped += 1
                continue
            decisions.append(d)
        if dropped:
            warnings.append(
                f"⚠ Dropped {dropped} malformed record(s) from decision history; "
                "surviving records used."
            )

    trades_path = DATA_DIR / "trades_history.json"
    trades = []
    if trades_path.exists():
        ledger = load_json_list_or_quarantine(
            trades_path, context="trade ledger")
        if not trades_path.exists():
            warnings.append(
                f"⚠ Trade ledger {trades_path.name} was corrupt and has been "
                "quarantined; round-trip/churn sections reflect an empty ledger."
            )
        dropped = 0
        for idx, t in enumerate(ledger):
            # Contract required by match_round_trips: mapping access, a string
            # ticker (used as a dict key), and an action key. Timestamps that
            # are missing or unparseable take churn's documented fallback
            # (datetime.min); timestamps that parse but carry tzinfo are
            # dropped loudly — the FIFO matcher compares/subtracts against
            # naive datetimes, so an aware record detonates one operation
            # downstream and voids every round trip (PR #64 doctrine; the
            # churn_analysis ingress validates the same contract).
            if not isinstance(t, dict):
                print(f"⚠ Dropping malformed trade record at index {idx} in "
                      f"{trades_path.name} (expected dict, got "
                      f"{type(t).__name__})")
                dropped += 1
                continue
            if not isinstance(t.get("ticker"), str) or "action" not in t:
                print(f"⚠ Dropping malformed trade record at index {idx} in "
                      f"{trades_path.name} (missing string 'ticker' or "
                      f"'action' key: {t!r})")
                dropped += 1
                continue
            raw_ts = t.get("timestamp")
            if isinstance(raw_ts, str) and raw_ts:
                try:
                    parsed_ts = datetime.fromisoformat(raw_ts)
                except ValueError:
                    parsed_ts = None
                if parsed_ts is not None and parsed_ts.tzinfo is not None:
                    print(f"⚠ Dropping malformed trade record at index {idx} in "
                          f"{trades_path.name} (timezone-aware timestamp in "
                          f"naive-timestamp ledger: {raw_ts!r})")
                    dropped += 1
                    continue
            trades.append(t)
        if dropped:
            warnings.append(
                f"⚠ Dropped {dropped} malformed record(s) from the trade "
                "ledger; surviving records used."
            )

    return decisions, trades, warnings


def main():
    decisions, trades, warnings = _load_analysis_histories()

    valid = [d for d in decisions if not d.get("error", False)]
    errors = [d for d in decisions if d.get("error", False)]

    lines = [
        "=" * 60,
        "BEHAVIORAL ANALYSIS OF LLM DECISIONS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 60,
        "",
    ]
    if warnings:
        # The report is the artifact a human reads; degradation must be
        # visible in it, not only on stderr (PR #87 doctrine).
        lines.extend(warnings)
        lines.append("")
    lines.extend([
        "SAMPLE OVERVIEW",
        f"Total decisions: {len(decisions)}",
        f"Valid decisions: {len(valid)}",
        f"Errors: {len(errors)} ({len(errors)/max(len(decisions),1)*100:.1f}%)",
        "",
    ])

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
    lines.extend(_format_cash_levels(daily_results))

    # Round-trip churn analysis (use the same FIFO matching logic as churn_analysis.py
    # so the two reports stay consistent and avoid negative/mismatched round trips).
    round_trips = get_round_trips(trades)

    winning = [r for r in round_trips if r["pnl"] > 0]
    short = [r for r in round_trips if r["hold_days"] <= 3]
    medium = [r for r in round_trips if 3 < r["hold_days"] <= 14]
    long = [r for r in round_trips if r["hold_days"] > 14]

    if decisions:
        first = datetime.fromisoformat(decisions[0]["timestamp"])
        last = datetime.fromisoformat(decisions[-1]["timestamp"])
        days = max((last - first).days, 1)
    else:
        # Empty history is a legitimate state (fresh account); churn_analysis
        # uses the same days_active = 1 convention for it. After element
        # filtering the list can be empty even when the file held records.
        days = 1

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
