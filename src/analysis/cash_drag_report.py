"""Cash-drag report: compare daily cash levels to regime targets.

This report helps distinguish two common pathologies:
1. Cash drag - cash is above the regime upper bound, suggesting under-investment.
2. Cap binding - cash is above target but the weekly trade cap is already hit,
   so the constraint is the cap, not the prompt.
"""
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import load_valid_daily_results

REGIME_BOUNDS = {
    "high": (0.30, 0.50),
    "normal": (0.15, 0.30),
    "low": (0.10, 0.20),
}

DEFAULT_REGIME = "normal"
OUTPUT_DIR = ROOT / "results" / "analysis"


def _get_regime(result):
    """Extract volatility regime from a daily result, defaulting to normal."""
    status = result.get("cooldown", {}).get("status", {})
    regime = status.get("current_vol_regime") or status.get("config", {}).get("current_vol_regime")
    return (regime or DEFAULT_REGIME).lower()


def _cap_info(result):
    """Return (trades_this_week, weekly_cap) if available, else (None, None)."""
    status = result.get("cooldown", {}).get("status", {})
    return status.get("trades_this_week"), status.get("weekly_cap")


def analyze_cash_drag(results_dir, output_path=None):
    """Generate a cash-drag report from valid daily results."""
    results = load_valid_daily_results(str(results_dir))

    rows = []
    status_counter = Counter()
    for result in results:
        portfolio = result.get("portfolio_after", {})
        cash = portfolio.get("cash", 0.0)
        total = portfolio.get("total_value") or portfolio.get("total_value", 1.0)
        total = total or 1.0
        cash_pct = cash / total
        regime = _get_regime(result)
        lower, upper = REGIME_BOUNDS.get(regime, REGIME_BOUNDS[DEFAULT_REGIME])

        if cash_pct < lower:
            status = "below"
        elif cash_pct > upper:
            status = "above"
        else:
            status = "ok"
        status_counter[status] += 1

        trades = len(result.get("executed_trades", []))
        trades_this_week, weekly_cap = _cap_info(result)

        rows.append({
            "date": result.get("date", "unknown"),
            "cash_pct": cash_pct,
            "regime": regime,
            "lower": lower,
            "upper": upper,
            "status": status,
            "trades": trades,
            "trades_this_week": trades_this_week,
            "weekly_cap": weekly_cap,
        })

    if output_path is None:
        output_path = OUTPUT_DIR / f"cash_drag_{datetime.now().strftime('%Y%m%d')}.txt"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "=" * 70,
        "CASH DRAG REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "=" * 70,
        "",
        f"Days analyzed: {len(rows)}",
        f"  Within target  : {status_counter['ok']} ({status_counter['ok']/max(len(rows),1)*100:.1f}%)",
        f"  Above target   : {status_counter['above']} ({status_counter['above']/max(len(rows),1)*100:.1f}%)",
        f"  Below target   : {status_counter['below']} ({status_counter['below']/max(len(rows),1)*100:.1f}%)",
        "",
        "Regime targets: HIGH 30-50%, NORMAL 15-30%, LOW 10-20%",
        "",
        "Per-day detail (most recent last):",
        "-" * 70,
        f"{'Date':<12} {'Regime':<8} {'Cash %':>8} {'Target':<14} {'Status':<8} {'Trades':>6} {'Cap':>8}",
        "-" * 70,
    ]

    for row in rows:
        target = f"{row['lower']*100:.0f}-{row['upper']*100:.0f}%"
        cap = "-"
        if row["trades_this_week"] is not None and row["weekly_cap"] is not None:
            cap = f"{row['trades_this_week']}/{row['weekly_cap']}"
        lines.append(
            f"{row['date']:<12} {row['regime']:<8} {row['cash_pct']*100:>7.1f}% "
            f"{target:<14} {row['status']:<8} {row['trades']:>6} {cap:>8}"
        )

    # Flag days where cash was above target and the cap was not yet hit:
    # these are pure cash-drag days (the prompt should have acted).
    drag_days = [
        r for r in rows
        if r["status"] == "above"
        and (r["weekly_cap"] is None or (r["trades_this_week"] is not None and r["trades_this_week"] < r["weekly_cap"]))
    ]
    cap_bound_days = [
        r for r in rows
        if r["status"] == "above"
        and r["weekly_cap"] is not None
        and r["trades_this_week"] is not None
        and r["trades_this_week"] >= r["weekly_cap"]
    ]

    lines.extend([
        "",
        "=" * 70,
        "DIAGNOSIS",
        "=" * 70,
        f"Cash-drag days (above target with cap headroom): {len(drag_days)}",
        f"Cap-binding days (above target but cap reached): {len(cap_bound_days)}",
        "",
        "Interpretation:",
        "- drag days    → prompt is not deploying cash aggressively enough",
        "- cap days     → the weekly trade cap is the binding constraint",
        "",
    ])

    text = "\n".join(lines)
    output_path.write_text(text)
    return text, rows


def main():
    text, _ = analyze_cash_drag(ROOT / "results" / "daily")
    print(text)


if __name__ == "__main__":
    main()
