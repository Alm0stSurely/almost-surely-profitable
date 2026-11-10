"""Since-inception relative metrics (beta/alpha vs SPY) in run_daily_pipeline.

The nightly fetch window is 30 calendar days (~22 trading sessions), while
calculate_beta_alpha / calculate_information_ratio hard-require n>=30. The
6.5 step therefore persisted beta/alpha/treynor/information_ratio/
tracking_error as null in every daily JSON since inception — a dead feature
guarded by an unsatisfiable precondition (same class as the 2026-11-06
prompt-enrichment wiring bug).

Fix: absolute metrics stay on the nightly window; the relative metrics move
to the since-inception strategy-value series persisted in
results/daily/<date>.json (portfolio_after.total_value, ~190 sessions).
These tests pin the loader, the estimator wiring on deterministic data, and
the end-to-end JSON output.

Discriminator doctrine (PR #84): tests drive the production wiring through
run_daily_pipeline and assert on computed values, so they fail on the old
code with a visible symptom instead of erroring at import time.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

REPO_SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(REPO_SRC))

from daily_run import (  # noqa: E402
    _load_strategy_value_series,
    _relative_metrics_since_inception,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _write_result(results_dir, date_str, total_value, timestamp=None, dry_run=False):
    name = f"{date_str}{'_dry_run' if dry_run else ''}.json"
    record = {
        "date": date_str,
        "timestamp": timestamp or f"{date_str}T21:00:00",
        "portfolio_after": {"total_value": total_value},
    }
    (results_dir / name).write_text(json.dumps(record))


def _business_days(start, periods):
    return pd.date_range(start, periods=periods, freq="B")


def _spy_close_frame(dates, closes):
    return pd.DataFrame({"Close": closes}, index=pd.DatetimeIndex(dates))


# ---------------------------------------------------------------------------
# Loader unit tests
# ---------------------------------------------------------------------------

class TestLoadStrategyValueSeries:
    def test_builds_sorted_series_from_daily_jsons(self, tmp_path):
        rd = tmp_path / "daily"
        rd.mkdir()
        _write_result(rd, "2026-10-01", 10000.0)
        _write_result(rd, "2026-10-02", 10100.0)
        _write_result(rd, "2026-10-03", 10050.0)

        series = _load_strategy_value_series(rd)
        assert series is not None
        assert list(series.index) == list(pd.to_datetime(
            ["2026-10-01", "2026-10-02", "2026-10-03"]
        ))
        np.testing.assert_allclose(series.to_numpy(), [10000.0, 10100.0, 10050.0])

    def test_excludes_dry_run_files(self, tmp_path):
        rd = tmp_path / "daily"
        rd.mkdir()
        _write_result(rd, "2026-10-01", 10000.0)
        _write_result(rd, "2026-10-02", 10100.0)
        _write_result(rd, "2026-10-02", 999999.0, dry_run=True)

        series = _load_strategy_value_series(rd)
        assert series is not None
        assert len(series) == 2
        assert series.loc[pd.Timestamp("2026-10-02")] == 10100.0  # dry run ignored

    def test_skips_corrupt_files_loudly_and_keeps_bytes(self, tmp_path, capsys):
        rd = tmp_path / "daily"
        rd.mkdir()
        _write_result(rd, "2026-10-01", 10000.0)
        _write_result(rd, "2026-10-03", 10100.0)
        bad = rd / "2026-10-02.json"
        bad.write_text("{not json")
        before = bad.read_bytes()

        series = _load_strategy_value_series(rd)
        assert series is not None
        assert len(series) == 2
        assert bad.read_bytes() == before  # read-only: evidence preserved
        assert "skipped 1 file(s)" in capsys.readouterr().out

    def test_drops_non_finite_or_nonpositive_values(self, tmp_path):
        rd = tmp_path / "daily"
        rd.mkdir()
        _write_result(rd, "2026-10-01", 10000.0)
        (rd / "2026-10-02.json").write_text(json.dumps({
            "date": "2026-10-02",
            "portfolio_after": {"total_value": float("nan")},
        }))
        (rd / "2026-10-03.json").write_text(json.dumps({
            "date": "2026-10-03",
            "portfolio_after": {"total_value": -5.0},
        }))
        _write_result(rd, "2026-10-04", 10100.0)

        series = _load_strategy_value_series(rd)
        assert series is not None
        assert len(series) == 2

    def test_duplicate_date_keeps_later_timestamp(self, tmp_path):
        rd = tmp_path / "daily"
        rd.mkdir()
        _write_result(rd, "2026-10-01", 10000.0, timestamp="2026-10-01T21:00:00")
        _write_result(rd, "2026-10-01", 10050.0, timestamp="2026-10-01T22:30:00")
        _write_result(rd, "2026-10-02", 10100.0)

        series = _load_strategy_value_series(rd)
        assert series is not None
        assert len(series) == 2
        assert series.loc[pd.Timestamp("2026-10-01")] == 10050.0

    def test_returns_none_with_fewer_than_two_marks(self, tmp_path):
        rd = tmp_path / "daily"
        rd.mkdir()
        _write_result(rd, "2026-10-01", 10000.0)
        assert _load_strategy_value_series(rd) is None
        assert _load_strategy_value_series(tmp_path / "nonexistent") is None


# ---------------------------------------------------------------------------
# Relative-metrics estimator wiring — deterministic data
# ---------------------------------------------------------------------------

class TestRelativeMetricsSinceInception:
    def test_beta_two_x_spy_on_deterministic_data(self, tmp_path, monkeypatch):
        # 45 sessions -> 44 returns (>= 30 guard satisfied).
        days = _business_days("2026-08-20", 45)
        rng = np.linspace(-0.02, 0.025, 45)
        spy_closes = 100.0 * np.cumprod(1.0 + rng)

        rd = tmp_path / "daily"
        rd.mkdir()
        # Strategy marks engineered so strategy returns == 2 * SPY returns:
        # value_{t} = value_{t-1} * (1 + 2*r_t). Start after the first SPY
        # close so the SPY pct_change return lines up with the strategy marks.
        value = 10000.0
        _write_result(rd, days[1].strftime("%Y-%m-%d"), value)
        for i in range(2, 45):
            value = value * (1.0 + 2.0 * rng[i])
            _write_result(rd, days[i].strftime("%Y-%m-%d"), value)

        spy_frame = _spy_close_frame(days, spy_closes)
        monkeypatch.setattr(
            "daily_run.fetch_historical_data",
            MagicMock(return_value={"SPY": spy_frame}),
        )

        series = _load_strategy_value_series(rd)
        result = _relative_metrics_since_inception(series)

        # Marks on days[1..44] give 43 strategy returns; SPY pct_change gives
        # 44 returns indexed days[1..44]; the joint window is days[2..44].
        assert result["relative_metrics_window_days"] == 43
        assert result["beta"] == pytest.approx(2.0, abs=1e-9)
        # alpha_daily = mean_p - (rf_d + beta*(mean_b - rf_d)); mean_p = 2*mean_b
        #   -> alpha_daily = rf_d = 0.02/252; annualized = 0.02.
        assert result["alpha"] == pytest.approx(0.02, abs=1e-9)
        # active = strategy - spy = spy returns on the joint window
        # days[2..44] (43 values); the joint aligns SPY pct_change (indexed
        # days[1..44]) with strategy pct_change (indexed days[2..44]).
        spy_joint_closes = spy_closes[1:]
        active = np.diff(spy_joint_closes) / spy_joint_closes[:-1]
        expected_ir = (np.mean(active) * 252) / (np.std(active, ddof=1) * np.sqrt(252))
        assert result["information_ratio"] == pytest.approx(expected_ir, abs=1e-9)
        assert result["tracking_error"] == pytest.approx(
            np.std(active, ddof=1) * np.sqrt(252), abs=1e-12
        )
        assert result["treynor_ratio"] is not None

    def test_short_history_returns_none_with_window_days(self, tmp_path, capsys):
        rd = tmp_path / "daily"
        rd.mkdir()
        days = _business_days("2026-10-20", 25)
        value = 10000.0
        for d in days:
            _write_result(rd, d.strftime("%Y-%m-%d"), value)
            value *= 1.001

        series = _load_strategy_value_series(rd)
        result = _relative_metrics_since_inception(series)
        assert result["beta"] is None
        assert result["relative_metrics_window_days"] == 24
        assert "< 30" in capsys.readouterr().out

    def test_fetch_failure_returns_none_contract(self, tmp_path, monkeypatch, capsys):
        rd = tmp_path / "daily"
        rd.mkdir()
        days = _business_days("2026-08-20", 45)
        for i, d in enumerate(days):
            _write_result(rd, d.strftime("%Y-%m-%d"), 10000.0 * (1.001 ** i))

        monkeypatch.setattr(
            "daily_run.fetch_historical_data",
            MagicMock(side_effect=RuntimeError("network down")),
        )
        result = _relative_metrics_since_inception(_load_strategy_value_series(rd))
        assert all(result[k] is None for k in
                   ("beta", "alpha", "treynor_ratio", "information_ratio",
                    "tracking_error", "relative_metrics_window_days"))
        assert "fetch failed" in capsys.readouterr().out

    def test_spy_frame_missing_close_returns_none(self, tmp_path, monkeypatch, capsys):
        rd = tmp_path / "daily"
        rd.mkdir()
        days = _business_days("2026-08-20", 45)
        for i, d in enumerate(days):
            _write_result(rd, d.strftime("%Y-%m-%d"), 10000.0 * (1.001 ** i))

        monkeypatch.setattr(
            "daily_run.fetch_historical_data",
            MagicMock(return_value={"SPY": pd.DataFrame({"Open": [1.0]})}),
        )
        result = _relative_metrics_since_inception(_load_strategy_value_series(rd))
        assert result["beta"] is None
        assert "missing Close" in capsys.readouterr().out

    def test_joint_window_below_guard_after_alignment(self, tmp_path, monkeypatch, capsys):
        # SPY covers only the last 25 strategy sessions -> 24 joint returns,
        # strictly below the 30-return guard -> None contract + loud report.
        rd = tmp_path / "daily"
        rd.mkdir()
        days = _business_days("2026-08-20", 45)
        for i, d in enumerate(days):
            _write_result(rd, d.strftime("%Y-%m-%d"), 10000.0 * (1.001 ** i))
        spy_days = days[-25:]
        spy_closes = 100.0 * np.cumprod(1.0 + np.linspace(-0.01, 0.012, 25))

        monkeypatch.setattr(
            "daily_run.fetch_historical_data",
            MagicMock(return_value={"SPY": _spy_close_frame(spy_days, spy_closes)}),
        )
        result = _relative_metrics_since_inception(_load_strategy_value_series(rd))
        assert result["beta"] is None
        assert result["relative_metrics_window_days"] == 24
        assert "joint SPY/strategy window" in capsys.readouterr().out

    def test_none_series_returns_none_contract(self):
        result = _relative_metrics_since_inception(None)
        assert all(v is None for v in result.values())


# ---------------------------------------------------------------------------
# End-to-end wiring: relative metrics reach the persisted daily JSON.
# Reuses the proven _run_pipeline scaffolding from
# test_daily_run_date_aligned_risk.py (PR #84 discriminator doctrine): real
# estimator functions run through run_daily_pipeline on deterministic data,
# and the persisted JSON is asserted on computed values.
# ---------------------------------------------------------------------------

def _run_pipeline_with_history(tmp_path, monkeypatch, n_sessions=45):
    """Fabricate a since-inception value history plus a deterministic SPY
    Close frame, run run_daily_pipeline, return the persisted result dict."""
    from test_daily_run_date_aligned_risk import _asset, _run_pipeline

    # 45 business sessions ending 2026-07-31 — before the fixture's fixed
    # run date (2026-08-12), so no mark coincides with the new daily JSON.
    days = _business_days("2026-06-01", n_sessions)
    rng = np.linspace(-0.015, 0.02, n_sessions)
    spy_closes = 100.0 * np.cumprod(1.0 + rng)

    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True)
    # Strategy marks engineered so strategy returns == 2 * SPY returns.
    value = 10000.0
    _write_result(results_dir, days[1].strftime("%Y-%m-%d"), value)
    for i in range(2, n_sessions):
        value *= (1.0 + 2.0 * rng[i])
        _write_result(results_dir, days[i].strftime("%Y-%m-%d"), value)

    spy_frame = pd.DataFrame({"Close": spy_closes}, index=pd.DatetimeIndex(days))
    daily_frame = pd.DataFrame({"Daily_Return": rng}, index=pd.DatetimeIndex(days))
    market_analysis = {
        "assets": {"SPY": _asset(daily_frame)},
        "analysis_date": "2026-08-12",
    }
    positions = [
        {"ticker": "SPY", "quantity": 80.0, "avg_price": 100.0,
         "current_price": float(spy_closes[-1]), "market_value": 8000.0,
         "unrealized_pnl_pct": 0.0},
    ]

    _run_pipeline(tmp_path, monkeypatch, market_analysis, {"SPY": spy_frame}, positions)

    with open(results_dir / "2026-08-12.json") as f:
        return json.load(f)


class TestRelativeMetricsReachDailyJson:
    def test_beta_persisted_in_daily_result(self, tmp_path, monkeypatch):
        result = _run_pipeline_with_history(tmp_path, monkeypatch)
        perf = result["performance_metrics"]
        assert perf["beta"] == pytest.approx(2.0, abs=1e-6)
        assert perf["alpha"] == pytest.approx(0.02, abs=1e-6)
        assert perf["relative_metrics_window_days"] == 43
        assert perf["information_ratio"] is not None
        assert perf["tracking_error"] is not None

    def test_absolute_metrics_untouched_by_relative_switch(self, tmp_path, monkeypatch):
        # Sharpe/volatility come from the nightly window regardless of the
        # since-inception series — regime continuity of the absolute block.
        result = _run_pipeline_with_history(tmp_path, monkeypatch)
        perf = result["performance_metrics"]
        assert np.isfinite(perf["sharpe_ratio"])
        assert np.isfinite(perf["volatility"])
        assert np.isfinite(perf["max_drawdown"])

    def test_relative_metrics_none_without_history(self, tmp_path, monkeypatch):
        # No fabricated history -> loader yields None -> the five relative
        # keys stay null but the window-day count reports zero sessions.
        from test_daily_run_date_aligned_risk import _asset, _run_pipeline

        results_dir = tmp_path / "results" / "daily"
        results_dir.mkdir(parents=True)
        days = _business_days("2026-06-01", 45)
        spy_closes = 100.0 * np.cumprod(np.full(45, 1.0005))
        spy_frame = pd.DataFrame({"Close": spy_closes}, index=pd.DatetimeIndex(days))
        daily_frame = pd.DataFrame(
            {"Daily_Return": np.full(45, 0.0005)}, index=pd.DatetimeIndex(days)
        )
        market_analysis = {
            "assets": {"SPY": _asset(daily_frame)},
            "analysis_date": "2026-08-12",
        }
        positions = [
            {"ticker": "SPY", "quantity": 80.0, "avg_price": 100.0,
             "current_price": float(spy_closes[-1]), "market_value": 8000.0,
             "unrealized_pnl_pct": 0.0},
        ]

        _run_pipeline(
            tmp_path, monkeypatch, market_analysis, {"SPY": spy_frame}, positions
        )

        with open(results_dir / "2026-08-12.json") as f:
            perf = json.load(f)["performance_metrics"]
        assert perf["beta"] is None
        assert perf["alpha"] is None
        assert perf["information_ratio"] is None
        assert perf["tracking_error"] is None
        assert perf["treynor_ratio"] is None
        # None-history contract: the window-day count is itself undefined.
        assert perf["relative_metrics_window_days"] is None
