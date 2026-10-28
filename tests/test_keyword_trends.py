"""Tests for keyword_trends.py formatting guards."""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.keyword_trends import (
    HIGHLIGHT_CONCEPTS,
    _safe_pct_str,
    _safe_signed_str,
    format_report,
    group_by_iso_week,
    linear_slope,
    load_decisions,
    rolling_average,
)
from utils import _is_finite_number


class TestHighlightConcepts:
    def test_cash_band_is_highlighted(self):
        """Rule-5 internalization concept must appear in the trend report.

        The dashboard exists to surface guardrail internalization early;
        "cash buffer" undercounts band/drag language (see 2026-10-26
        research session), so "cash band" is a first-class highlight.
        """
        assert "cash band" in HIGHLIGHT_CONCEPTS
        assert "cash buffer" in HIGHLIGHT_CONCEPTS


class TestSafeFormatHelpers:
    def test_safe_pct_str_finite(self):
        assert _safe_pct_str(12.34) == "   12.3%"

    def test_safe_pct_str_nan(self):
        assert _safe_pct_str(float("nan")) == "     n/a"

    def test_safe_pct_str_inf(self):
        assert _safe_pct_str(float("inf")) == "     n/a"

    def test_safe_pct_str_none(self):
        assert _safe_pct_str(None) == "     n/a"

    def test_safe_signed_str_finite_positive(self):
        assert _safe_signed_str(0.42) == "    +0.42"

    def test_safe_signed_str_finite_negative(self):
        assert _safe_signed_str(-1.23) == "    -1.23"

    def test_safe_signed_str_nan(self):
        assert _safe_signed_str(float("nan")) == "      n/a"

    def test_safe_signed_str_inf(self):
        assert _safe_signed_str(float("inf")) == "      n/a"


class TestRollingAverage:
    def test_window_with_nan_returns_nan(self):
        values = [10.0, 20.0, float("nan"), 40.0]
        out = rolling_average(values, window=4)
        assert any(v != v for v in out[2:])
        assert all(_is_finite_number(v) for v in out[:2])

    def test_all_finite_returns_expected(self):
        values = [10.0, 20.0, 30.0]
        out = rolling_average(values, window=2)
        assert out == [10.0, 15.0, 25.0]


class TestLinearSlope:
    def test_non_finite_input_returns_zero(self):
        assert linear_slope([1.0, float("nan"), 3.0]) == 0.0
        assert linear_slope([1.0, float("inf"), 3.0]) == 0.0

    def test_short_series_returns_zero(self):
        assert linear_slope([5.0]) == 0.0

    def test_rising_trend_positive_slope(self):
        slope = linear_slope([0.0, 1.0, 2.0, 3.0])
        assert slope > 0.9


class TestFormatReportGuards:
    def test_non_finite_rolling_average_renders_na(self):
        weekly_rates = {
            "2026-W01": {"_n": 1, "loss aversion": 10.0},
            "2026-W02": {"_n": 1, "loss aversion": float("nan")},
            "2026-W03": {"_n": 1, "loss aversion": 30.0},
            "2026-W04": {"_n": 1, "loss aversion": 40.0},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        # The rolling average for the last week spans a window containing NaN.
        lines = report.splitlines()
        summary_line = [l for l in lines if l.strip().startswith("loss aversion")][0]
        assert "     n/a" in summary_line

    def test_non_finite_latest_renders_na(self):
        weekly_rates = {
            "2026-W01": {"_n": 1, "loss aversion": float("nan")},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        assert "     n/a" in report

    def test_finite_values_render_normally(self):
        weekly_rates = {
            "2026-W01": {"_n": 2, "loss aversion": 50.0},
            "2026-W02": {"_n": 2, "loss aversion": 50.0},
            "2026-W03": {"_n": 2, "loss aversion": 50.0},
            "2026-W04": {"_n": 2, "loss aversion": 50.0},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        assert "50.0%" in report
        assert "nan%" not in report
        assert "inf%" not in report

    def test_non_finite_slope_renders_direction_na(self):
        weekly_rates = {
            "2026-W01": {"_n": 1, "loss aversion": 0.0},
            "2026-W02": {"_n": 1, "loss aversion": float("nan")},
        }
        report = format_report(weekly_rates, highlight_concepts=["loss aversion"], window=4)
        assert "       n/a" in report

    def test_empty_weekly_rates(self):
        assert format_report({}) == "No weekly data available."


class TestLoadDecisionsIngress:
    """File-level contract: corrupt decision history is quarantined, not
    ingested and not silently crashed on (PR #84 strict-reader doctrine)."""

    def _write(self, tmp_path, payload):
        path = tmp_path / "decision_history.json"
        if isinstance(payload, bytes):
            path.write_bytes(payload)
        else:
            path.write_text(payload)
        return path

    def test_nan_token_file_quarantined(self, tmp_path, caplog):
        import logging

        path = self._write(
            tmp_path,
            '[{"timestamp": "2026-10-20T09:00:00", "reasoning": NaN}]',
        )
        with caplog.at_level(logging.ERROR):
            result = load_decisions(path)
        assert result == []
        quarantines = list(tmp_path.glob("decision_history.json.corrupt-*"))
        assert len(quarantines) == 1
        assert b"NaN" in quarantines[0].read_bytes()
        assert any("decision history" in r.getMessage() for r in caplog.records
                   if r.levelno >= logging.ERROR)

    def test_infinity_token_file_quarantined(self, tmp_path):
        path = self._write(tmp_path, '[{"timestamp": "2026-10-20T09:00:00", "score": Infinity}]')
        assert load_decisions(path) == []
        assert list(tmp_path.glob("decision_history.json.corrupt-*"))

    def test_garbage_bytes_quarantined(self, tmp_path):
        path = self._write(tmp_path, b"\xff\xfe{not json")
        assert load_decisions(path) == []
        assert list(tmp_path.glob("decision_history.json.corrupt-*"))

    def test_wrong_shape_dict_quarantined(self, tmp_path):
        path = self._write(tmp_path, '{"timestamp": "2026-10-20T09:00:00"}')
        assert load_decisions(path) == []
        assert list(tmp_path.glob("decision_history.json.corrupt-*"))

    def test_wrong_shape_scalar_quarantined(self, tmp_path):
        path = self._write(tmp_path, "42")
        assert load_decisions(path) == []
        assert list(tmp_path.glob("decision_history.json.corrupt-*"))

    def test_healthy_file_returns_records(self, tmp_path):
        records = [
            {"timestamp": "2026-10-20T09:00:00", "reasoning": "hold"},
            {"timestamp": "2026-10-21T09:00:00", "reasoning": "buy SPY"},
        ]
        path = self._write(tmp_path, json.dumps(records))
        assert load_decisions(path) == records
        assert not list(tmp_path.glob("decision_history.json.corrupt-*"))

    def test_missing_file_returns_empty(self, tmp_path):
        assert load_decisions(tmp_path / "decision_history.json") == []


class TestGroupByIsoWeekIngress:
    """Element-level contract: one malformed element drops loudly, valid
    siblings survive (anti-Lipschitz); expected degenerate inputs stay quiet."""

    def test_non_dict_element_dropped_loudly_siblings_survive(self, capsys):
        decisions = [
            {"timestamp": "2026-10-19T09:00:00", "reasoning": "hold"},
            "not-a-dict",
            {"timestamp": "2026-10-21T09:00:00", "reasoning": "buy"},
        ]
        weeks = group_by_iso_week(decisions)
        out = capsys.readouterr().out
        assert "index 1" in out and "expected dict, got str" in out
        assert sum(len(v) for v in weeks.values()) == 2
        assert "2026-W43" in weeks and "2026-W43" in weeks  # Oct 19 & 21 are both W43

    def test_missing_timestamp_dropped_loudly(self, capsys):
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00"},
            {"reasoning": "no timestamp here"},
        ])
        out = capsys.readouterr().out
        assert "index 1" in out and "missing/empty timestamp" in out
        assert sum(len(v) for v in weeks.values()) == 1

    def test_non_string_timestamp_dropped_loudly(self, capsys):
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00"},
            {"timestamp": 1234567890},
        ])
        out = capsys.readouterr().out
        assert "index 1" in out and "1234567890" in out
        assert sum(len(v) for v in weeks.values()) == 1

    def test_unparseable_timestamp_dropped_loudly(self, capsys):
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00"},
            {"timestamp": "not-a-date"},
        ])
        out = capsys.readouterr().out
        assert "index 1" in out and "unparseable timestamp" in out
        assert sum(len(v) for v in weeks.values()) == 1

    def test_empty_timestamp_dropped_loudly(self, capsys):
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00"},
            {"timestamp": ""},
        ])
        assert "index 1" in capsys.readouterr().out
        assert sum(len(v) for v in weeks.values()) == 1

    def test_error_flagged_record_without_timestamp_skips_quietly(self, capsys):
        """Expected degenerate input: an error record carries no analysis
        payload, so the documented quiet path must not become noisy."""
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00", "reasoning": "hold"},
            {"error": True, "reasoning": "LLM call failed"},
        ])
        assert capsys.readouterr().out == ""
        assert sum(len(v) for v in weeks.values()) == 1

    def test_tz_aware_timestamp_accepted(self, capsys):
        """Codomain check: this module only calls isocalendar() per record —
        no naive/aware arithmetic — so tz-aware parses and groups like any
        other record (contrast churn_analysis, which drops them)."""
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00+02:00"},
        ])
        assert capsys.readouterr().out == ""
        assert sum(len(v) for v in weeks.values()) == 1

    def test_healthy_records_group_by_iso_week(self, capsys):
        weeks = group_by_iso_week([
            {"timestamp": "2026-10-19T09:00:00"},
            {"timestamp": "2026-10-20T09:00:00"},
            {"timestamp": "2026-10-26T09:00:00"},
        ])
        assert capsys.readouterr().out == ""
        assert len(weeks["2026-W43"]) == 2
        assert len(weeks["2026-W44"]) == 1
