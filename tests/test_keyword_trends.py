"""Tests for keyword_trends.py."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.keyword_trends import (
    compute_weekly_rates,
    format_report,
    group_by_iso_week,
    linear_slope,
    load_decisions,
    rolling_average,
)


def make_decision(timestamp, reasoning="", error=False):
    return {"timestamp": timestamp, "reasoning": reasoning, "error": error, "actions": []}


class TestLoadDecisions:
    def test_loads_from_default_path(self, tmp_path, monkeypatch):
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        decisions = [
            make_decision("2026-01-05T21:00:00", "loss aversion"),
        ]
        with open(data_dir / "decision_history.json", "w") as f:
            import json

            json.dump(decisions, f)
        monkeypatch.setattr(
            "analysis.keyword_trends.DATA_DIR", data_dir
        )
        loaded = load_decisions()
        assert loaded == decisions


class TestGroupByIsoWeek:
    def test_groups_by_iso_week(self):
        decisions = [
            make_decision("2026-01-05T21:00:00", "loss aversion"),  # Monday W02
            make_decision("2026-01-06T21:00:00", "cash buffer"),  # Tuesday W02
            make_decision("2026-01-12T21:00:00", "loss aversion"),  # Monday W03
        ]
        weeks = group_by_iso_week(decisions)
        assert set(weeks.keys()) == {"2026-W02", "2026-W03"}
        assert len(weeks["2026-W02"]) == 2
        assert len(weeks["2026-W03"]) == 1

    def test_skips_error_decisions(self):
        decisions = [make_decision("2026-01-05T21:00:00", "loss aversion", error=True)]
        assert group_by_iso_week(decisions) == {}


class TestComputeWeeklyRates:
    def test_rates_per_week(self):
        concepts = {"loss aversion": ["loss aversion"]}
        decisions = [
            make_decision("2026-01-05T21:00:00", "loss aversion"),
            make_decision("2026-01-06T21:00:00", "cash buffer"),
        ]
        weeks = group_by_iso_week(decisions)
        rates = compute_weekly_rates(weeks, concepts)
        assert rates["2026-W02"]["loss aversion"] == 50.0
        assert rates["2026-W02"]["_n"] == 2


class TestRollingAverage:
    def test_rolling_average(self):
        assert rolling_average([1, 2, 3, 4], window=2) == [1.0, 1.5, 2.5, 3.5]

    def test_partial_window_at_start(self):
        assert rolling_average([10, 20, 30], window=4) == [10.0, 15.0, 20.0]


class TestLinearSlope:
    def test_positive_slope(self):
        values = [0, 2, 4, 6]
        assert linear_slope(values) == pytest.approx(2.0)

    def test_flat_slope(self):
        assert linear_slope([5, 5, 5]) == 0.0

    def test_single_value_returns_zero(self):
        assert linear_slope([42]) == 0.0


class TestFormatReport:
    def test_report_contains_latest_and_trend(self):
        concepts = {"loss aversion": ["loss aversion"]}
        decisions = [
            make_decision("2026-01-05T21:00:00", "loss aversion"),
            make_decision("2026-01-06T21:00:00", ""),
        ]
        weeks = group_by_iso_week(decisions)
        rates = compute_weekly_rates(weeks, concepts)
        report = format_report(
            rates, highlight_concepts=["loss aversion"], window=4
        )
        assert "KEYWORD MENTION-RATE TRENDS" in report
        assert "loss aversion" in report
        assert "50.0%" in report

    def test_empty_data_returns_message(self):
        assert format_report({}) == "No weekly data available."
