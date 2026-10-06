"""Cash-level vs next-day alpha correlation report.

Quantifies the opportunity cost of cash drag: does the cash share of the
portfolio on day J predict the strategy's alpha (vs the live equal-weight
benchmark) on day J+1?

Hypothesis (research-2026-10-05): a high cash share on a market-up day
forfeits next-day upside, so corr(cash_pct_J, alpha_J+1) should be negative.
A positive correlation would instead suggest cash is acting as defensive
ballast (protecting on down days).

Guards follow the shared-utils convention: non-finite inputs are dropped,
and below MIN_PAIRED_SAMPLES paired observations the report refuses to
emit a correlation (small-sample guard).
"""

import math
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "src"))

from utils import _is_finite_number, load_valid_daily_results

OUTPUT_DIR = ROOT / "results" / "analysis"
MIN_PAIRED_SAMPLES = 10

# Cash bands used for the binned breakdown (fractions of portfolio value).
CASH_BANDS = [
    ("< 15%", None, 0.15),
    ("15-30%", 0.15, 0.30),
    ("30-50%", 0.30, 0.50),
    (">= 50%", 0.50, None),
]


def _cash_pct(result):
    """Cash share of portfolio_after, or None if undefined."""
    portfolio = result.get("portfolio_after", {})
    cash = portfolio.get("cash")
    total = portfolio.get("total_value")
    if not _is_finite_number(cash) or not _is_finite_number(total):
        return None
    if total <= 0:
        return None
    return cash / total


def _series(result):
    """(total_value, benchmark_total_value) from a daily result, guarded."""
    portfolio = result.get("portfolio_after", {})
    total = portfolio.get("total_value")
    bench = result.get("equalweight_benchmark", {})
    bench_total = bench.get("total_value")
    if not _is_finite_number(total) or total <= 0:
        return None
    if not _is_finite_number(bench_total) or bench_total <= 0:
        return None
    return total, bench_total


def _pearson(xs, ys):
    """Pearson r, or None if undefined (zero variance or too few points)."""
    n = len(xs)
    if n < 3:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _spearman(xs, ys):
    """Spearman rho via rank correlation, or None if undefined."""
    n = len(xs)
    if n < 3:
        return None

    def ranks(values):
        order = sorted(range(n), key=lambda i: values[i])
        result = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg_rank = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                result[order[k]] = avg_rank
            i = j + 1
        return result

    return _pearson(ranks(xs), ranks(ys))


def build_pairs(results):
    """Pair cash share on day J with next-day alpha (strategy vs EW benchmark).

    Returns a list of (date_J, cash_pct_J, alpha_J1_pct) tuples where
    alpha_J1_pct is in percentage points. Days without a valid successor
    (gaps, missing benchmark) are skipped.
    """
    pairs = []
    for idx in range(len(results) - 1):
        cur = results[idx]
        nxt = results[idx + 1]
        cash_pct = _cash_pct(cur)
        s_cur = _series(cur)
        s_next = _series(nxt)
        if cash_pct is None or s_cur is None or s_next is None:
            continue
        strat_ret = s_next[0] / s_cur[0] - 1.0
        bench_ret = s_next[1] / s_cur[1] - 1.0
        alpha = (strat_ret - bench_ret) * 100.0
        if not _is_finite_number(alpha):
            continue
        pairs.append((cur.get("date", "unknown"), cash_pct, alpha))
    return pairs


def analyze_cash_alpha_correlation(results_dir, output_path=None):
    """Generate the cash-vs-next-day-alpha report from valid daily results."""
    results = load_valid_daily_results(str(results_dir))
    pairs = build_pairs(results)

    lines = [
        "=" * 70,
        "CASH LEVEL vs NEXT-DAY ALPHA CORRELATION",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",  # noqa: DTZ005
        "=" * 70,
        "",
        f"Valid daily results: {len(results)}",
        f"Paired observations (day J -> alpha J+1): {len(pairs)}",
        "",
        "Alpha is the daily strategy return minus the live equal-weight",
        "benchmark return, in percentage points.",
        "",
    ]

    if len(pairs) < MIN_PAIRED_SAMPLES:
        lines.extend(
            [
                f"INSUFFICIENT DATA: {len(pairs)} paired observations",
                f"(minimum {MIN_PAIRED_SAMPLES} required for a meaningful",
                "correlation). Re-run once more history accumulates.",
                "",
            ]
        )
        text = "\n".join(lines)
        if output_path is None:
            if Path(results_dir).resolve() != (ROOT / "results" / "daily").resolve():
                return text, pairs
            output_path = OUTPUT_DIR / (
                f"cash_alpha_correlation_{datetime.now().strftime('%Y%m%d')}.txt"  # noqa: DTZ005
            )
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text)
        return text, pairs

    xs = [p[1] for p in pairs]
    ys = [p[2] for p in pairs]
    mean_cash = sum(xs) / len(xs)
    mean_alpha = sum(ys) / len(ys)
    pearson = _pearson(xs, ys)
    spearman = _spearman(xs, ys)

    lines.extend(
        [
            f"Mean cash share (J)     : {mean_cash * 100:.1f}%",
            f"Mean next-day alpha     : {mean_alpha:+.2f} pp",
            f"Pearson r               : "
            + (f"{pearson:+.3f}" if pearson is not None else "undefined (zero variance)"),
            f"Spearman rho            : "
            + (f"{spearman:+.3f}" if spearman is not None else "undefined (zero variance)"),
            "",
            "-" * 70,
            "Binned by cash share on day J:",
            f"{'Band':<10} {'Days':>5} {'Mean alpha J+1':>16} {'Win vs bench':>14}",
            "-" * 70,
        ]
    )

    for label, lo, hi in CASH_BANDS:
        bucket = [
            y
            for x, y in zip(xs, ys)
            if (lo is None or x >= lo) and (hi is None or x < hi)
        ]
        if bucket:
            bmean = sum(bucket) / len(bucket)
            win = sum(1 for b in bucket if b > 0) / len(bucket) * 100
            lines.append(f"{label:<10} {len(bucket):>5} {bmean:>+15.2f} pp {win:>12.1f}%")
        else:
            lines.append(f"{label:<10} {0:>5} {'n/a':>16} {'n/a':>14}")

    lines.extend(
        [
            "",
            "Interpretation:",
            "- Negative correlation -> high cash forfeits next-day upside",
            "  (cash drag is costly). Tighten redeploy language in prompt.",
            "- Positive correlation -> cash cushions next-day drawdowns",
            "  (defensive ballast). Current cash target may be justified.",
            "- |r| < 0.1 -> cash level does not predict next-day alpha;",
            "  the drag shows up at longer horizons or only on rally days.",
            "",
        ]
    )

    text = "\n".join(lines)

    if output_path is None:
        # Same guard as cash_drag_report.py: only the canonical production
        # input writes the dated artefact; tests/benchmarks pass fixtures.
        if Path(results_dir).resolve() != (ROOT / "results" / "daily").resolve():
            return text, pairs
        output_path = OUTPUT_DIR / (
            f"cash_alpha_correlation_{datetime.now().strftime('%Y%m%d')}.txt"  # noqa: DTZ005
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(text)
    return text, pairs


def main():
    text, _ = analyze_cash_alpha_correlation(ROOT / "results" / "daily")
    print(text)


if __name__ == "__main__":
    main()
