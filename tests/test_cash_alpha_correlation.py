"""Tests for cash_alpha_correlation.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.cash_alpha_correlation import (
    MIN_PAIRED_SAMPLES,
    _pearson,
    _spearman,
    analyze_cash_alpha_correlation,
    build_pairs,
)


def _write_result(tmp_path, date, cash, total, bench_total):
    """Helper to build a minimal valid daily result with a benchmark."""
    result = {
        "date": date,
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
        "portfolio_after": {"cash": cash, "total_value": total},
        "equalweight_benchmark": {"total_value": bench_total},
        "executed_trades": [],
    }
    with open(tmp_path / f"{date}.json", "w") as f:
        json.dump(result, f)


def _series_with_n(tmp_path, n, cash_frac=0.25, strat_growth=1.0, bench_growth=1.0):
    """Write n consecutive results with controlled growth and cash share."""
    strat, bench = 10000.0, 10000.0
    for i in range(n):
        cash = strat * cash_frac
        _write_result(tmp_path, f"2026-08-{10 + i:02d}", cash, strat, bench)
        strat *= strat_growth
        bench *= bench_growth


def test_pearson_perfect_positive_and_negative():
    assert _pearson([1, 2, 3, 4], [2, 4, 6, 8]) == 1.0
    assert _pearson([1, 2, 3, 4], [8, 6, 4, 2]) == -1.0


def test_pearson_zero_variance_returns_none():
    assert _pearson([1, 1, 1], [1, 2, 3]) is None
    assert _pearson([1, 2, 3], [1, 1, 1]) is None
    assert _pearson([1, 2], [1, 2]) is None  # too few points


def test_spearman_rank_correlation():
    assert _spearman([1, 2, 3, 4], [10, 20, 30, 40]) == 1.0
    assert _spearman([1, 2, 3, 4], [40, 30, 20, 10]) == -1.0
    # Ties get average ranks; monotone either way
    rho = _spearman([1, 1, 2], [5, 5, 9])
    assert rho is not None and rho > 0


def test_build_pairs_pairs_cash_with_next_day_alpha(tmp_path):
    # Day 1: cash 25%, strat 10000 -> 10100 (+1.0%), bench 10000 -> 10050 (+0.5%)
    # Day 2 provides day 1's successor prices.
    _write_result(tmp_path, "2026-08-10", 2500, 10000, 10000)
    _write_result(tmp_path, "2026-08-11", 2500, 10100, 10050)
    results_dir = tmp_path
    from utils import load_valid_daily_results

    results = load_valid_daily_results(str(results_dir))
    pairs = build_pairs(results)
    assert len(pairs) == 1
    date_j, cash_pct, alpha = pairs[0]
    assert date_j == "2026-08-10"
    assert cash_pct == 0.25
    assert abs(alpha - 0.5) < 1e-9  # +1.0% - +0.5% = +0.5 pp


def test_build_pairs_skips_non_finite_inputs(tmp_path):
    _write_result(tmp_path, "2026-08-10", 2500, 10000, 10000)
    # Successor with NaN benchmark value must be dropped.
    bad = {
        "date": "2026-08-11",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
        "portfolio_after": {"cash": 2500, "total_value": 10100},
        "equalweight_benchmark": {"total_value": float("nan")},
        "executed_trades": [],
    }
    with open(tmp_path / "2026-08-11.json", "w") as f:
        json.dump(bad, f)

    from utils import load_valid_daily_results

    results = load_valid_daily_results(str(tmp_path))
    pairs = build_pairs(results)
    assert pairs == []


def test_small_sample_guard_emits_insufficient_data(tmp_path):
    _series_with_n(tmp_path, MIN_PAIRED_SAMPLES - 1)
    from utils import load_valid_daily_results

    results = load_valid_daily_results(str(tmp_path))
    assert len(build_pairs(results)) < MIN_PAIRED_SAMPLES

    text, _ = analyze_cash_alpha_correlation(tmp_path)
    assert "INSUFFICIENT DATA" in text
    assert "Pearson" not in text.split("INSUFFICIENT DATA")[1]


def test_fixture_dir_does_not_write_dated_artifact(tmp_path):
    """Regression guard: fixture runs must not add new files to results/analysis."""
    _series_with_n(tmp_path, MIN_PAIRED_SAMPLES + 2)
    analysis_dir = Path("results/analysis")
    before = set(analysis_dir.glob("cash_alpha_correlation_*.txt"))
    analyze_cash_alpha_correlation(tmp_path)
    after = set(analysis_dir.glob("cash_alpha_correlation_*.txt"))
    assert after == before


def test_explicit_output_path_writes_report(tmp_path):
    _series_with_n(tmp_path, MIN_PAIRED_SAMPLES + 2)
    out = tmp_path / "report.txt"
    text, pairs = analyze_cash_alpha_correlation(tmp_path, output_path=out)
    assert out.exists()
    assert out.read_text() == text
    assert len(pairs) >= MIN_PAIRED_SAMPLES
    assert "Pearson r" in text


def test_flat_markets_report_zero_mean_alpha(tmp_path):
    # Strategy and benchmark grow identically -> all alphas zero -> Pearson
    # undefined (zero variance), report must not crash.
    _series_with_n(
        tmp_path, MIN_PAIRED_SAMPLES + 3, cash_frac=0.25, strat_growth=1.001, bench_growth=1.001
    )
    text, _ = analyze_cash_alpha_correlation(tmp_path)
    assert "undefined (zero variance)" in text
    assert "Mean next-day alpha" in text
