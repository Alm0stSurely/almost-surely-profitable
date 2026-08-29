"""Keyword mention-rate trend analysis for LLM decision history.

Tracks how often each behavioral concept appears in the LLM's reasoning over
ISO-calendar weeks. A 4-week rolling average smooths noise and a simple linear
trend slope highlights concepts that are becoming more (or less) internalized.

Usage:
    python src/analysis/keyword_trends.py

Output:
    Printed report and results/analysis/keyword_trends_YYYYMMDD.txt
"""
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from analysis.behavioral_analysis import KEYWORD_CONCEPTS, count_keyword_concepts
from utils import _is_finite_number

DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "results" / "analysis"

# Concepts that are most diagnostic of prompt alignment.
HIGHLIGHT_CONCEPTS = [
    "loss aversion",
    "CVaR",
    "tail risk",
    "mean reversion",
    "momentum",
    "cash buffer",
    "drawdown",
    "stop-loss",
    "trade cap",
    "cooldown",
    "let winners run",
    "deflated sharpe",
]


def load_decisions(path=None):
    """Load decisions from JSON."""
    if path is None:
        path = DATA_DIR / "decision_history.json"
    with open(path) as f:
        return json.load(f)


def group_by_iso_week(decisions):
    """Group valid decisions by ISO calendar week."""
    weeks = defaultdict(list)
    for d in decisions:
        if d.get("error", False):
            continue
        dt = datetime.fromisoformat(d["timestamp"])
        cal = dt.isocalendar()
        week = f"{cal.year}-W{cal.week:02d}"
        weeks[week].append(d)
    return weeks


def compute_weekly_rates(weeks, keyword_concepts=None):
    """Compute per-week mention rates for each keyword concept."""
    if keyword_concepts is None:
        keyword_concepts = KEYWORD_CONCEPTS
    weekly_rates = {}
    for week in sorted(weeks):
        decisions = weeks[week]
        counts = count_keyword_concepts(decisions, keyword_concepts)
        n = max(len(decisions), 1)
        weekly_rates[week] = {concept: count / n * 100 for concept, count in counts.items()}
        weekly_rates[week]["_n"] = len(decisions)
    return weekly_rates


def rolling_average(values, window=4):
    """Return a list of rolling averages with partial windows at the start.

    Any window that contains a non-finite value returns NaN so that a single
    corrupt observation does not silently skew the smoothed trend line.
    """
    out = []
    for i in range(len(values)):
        start = max(0, i - window + 1)
        window_vals = values[start : i + 1]
        if not all(_is_finite_number(v) for v in window_vals):
            out.append(float("nan"))
        else:
            out.append(sum(window_vals) / len(window_vals))
    return out


def linear_slope(values):
    """Least-squares slope in index units (percentage points per week).

    Returns 0.0 (flat) when the series is too short or contains any non-finite
    value, preventing NaN/inf slopes from reaching the trend report.
    """
    n = len(values)
    if n < 2 or not all(_is_finite_number(v) for v in values):
        return 0.0
    x_mean = (n - 1) / 2.0
    y_mean = sum(values) / n
    numerator = sum((i - x_mean) * (v - y_mean) for i, v in enumerate(values))
    denominator = sum((i - x_mean) ** 2 for i in range(n))
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _safe_pct_str(value, width=7, precision=1):
    """Format *value* as a percentage string, or right-aligned ``n/a``.

    Non-finite inputs (including ``NaN`` and ``inf``) render as ``n/a`` so
    the report never emits misleading tokens such as ``nan%``.
    """
    if not _is_finite_number(value):
        return f"{'n/a':>{width + 1}}"
    return f"{value:>{width}.{precision}f}%"


def _safe_signed_str(value, width=9, precision=2):
    """Format a signed numeric *value*, or right-aligned ``n/a``.

    Keeps the trend column aligned even when the slope computation falls
    back to a non-finite sentinel.
    """
    if not _is_finite_number(value):
        return f"{'n/a':>{width}}"
    return f"{value:>+{width}.{precision}f}"


def format_report(weekly_rates, highlight_concepts=None, window=4):
    """Render the keyword trend report as a string."""
    if not weekly_rates:
        return "No weekly data available."

    sorted_weeks = sorted(weekly_rates)
    if highlight_concepts is None:
        highlight_concepts = [
            c for c in weekly_rates[sorted_weeks[0]].keys() if not c.startswith("_")
        ]

    # Rolling averages for each highlighted concept.
    rolling = {
        concept: rolling_average(
            [weekly_rates[w].get(concept, 0.0) for w in sorted_weeks], window=window
        )
        for concept in highlight_concepts
    }

    # Linear trend slopes.
    trends = {
        concept: linear_slope([weekly_rates[w].get(concept, 0.0) for w in sorted_weeks])
        for concept in highlight_concepts
    }

    lines = [
        "=" * 80,
        "KEYWORD MENTION-RATE TRENDS",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Window: {window}-week rolling average",
        "=" * 80,
        "",
    ]

    # Weekly table.
    header_parts = [f"{'Week':<10}", f"{'N':>3}"]
    for concept in highlight_concepts:
        header_parts.append(f" {concept[:8]:>8}")
    lines.append("".join(header_parts))
    lines.append("-" * 80)

    for week in sorted_weeks:
        row_parts = [f"{week:<10}", f"{weekly_rates[week]['_n']:>3}"]
        for concept in highlight_concepts:
            rate = weekly_rates[week].get(concept, 0.0)
            row_parts.append(f" {_safe_pct_str(rate, width=7, precision=1)}")
        lines.append("".join(row_parts))

    lines.append("")
    lines.append(
        f"{'Concept':<20} {'Latest':>10} {f'{window}W Avg':>10} {'Trend':>10} {'Direction':>10}"
    )
    lines.append("-" * 80)

    latest_week = sorted_weeks[-1]
    for concept in highlight_concepts:
        latest = weekly_rates[latest_week].get(concept, 0.0)
        avg = rolling[concept][-1]
        slope = trends[concept]
        if not _is_finite_number(slope):
            direction = "n/a"
        elif slope > 0.5:
            direction = "rising"
        elif slope < -0.5:
            direction = "falling"
        else:
            direction = "flat"
        lines.append(
            f"{concept:<20} {_safe_pct_str(latest, width=9, precision=1)}"
            f" {_safe_pct_str(avg, width=9, precision=1)}"
            f" {_safe_signed_str(slope, width=9, precision=2)} {direction:>10}"
        )

    lines.append("=" * 80)
    return "\n".join(lines)


def main():
    decisions = load_decisions()
    weeks = group_by_iso_week(decisions)
    weekly_rates = compute_weekly_rates(weeks)
    report = format_report(weekly_rates, highlight_concepts=HIGHLIGHT_CONCEPTS, window=4)
    print(report)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"keyword_trends_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_path, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()
