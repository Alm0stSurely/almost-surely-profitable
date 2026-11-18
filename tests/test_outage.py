"""
Tests for the LLM outage escalation tracker (utils/outage.py).

Focus: the consecutive-error-day streak that decides between OK / DEGRADED
/ ALERT, and the in-memory ``today_error`` mode used by the daily pipeline
before it persists today's result.
"""

import json
import sys
from datetime import date
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils.outage import (
    DEFAULT_ALERT_DAYS,
    assess_outage,
    count_consecutive_error_days,
    iter_daily_results,
    write_outage_status,
)


def _write_daily(results_dir: Path, day: str, error: bool, decision_block: bool = True):
    """Write a minimal daily result file for ``day`` ('YYYY-MM-DD')."""
    results_dir.mkdir(parents=True, exist_ok=True)
    payload = {"date": day, "decision": {"error": error, "actions": []}}
    if not decision_block:
        payload = {"date": day}
    (results_dir / f"{day}.json").write_text(json.dumps(payload))


class TestIterDailyResults:
    def test_empty_dir_returns_empty(self, tmp_path):
        assert iter_daily_results(tmp_path / "nope") == []

    def test_ignores_dry_run_and_misc_files(self, tmp_path):
        rd = tmp_path / "results" / "daily"
        _write_daily(rd, "2026-11-16", error=True)
        (rd / "2026-11-17_dry_run.json").write_text("{}")
        (rd / "notes.txt").write_text("junk")
        (rd / ".hidden").write_text("junk")
        entries = iter_daily_results(rd)
        assert len(entries) == 1
        assert entries[0]["date"] == date(2026, 11, 16)

    def test_corrupt_file_skipped_loudly_not_raised(self, tmp_path, caplog):
        rd = tmp_path / "daily"
        rd.mkdir(parents=True)
        (rd / "2026-11-16.json").write_text("{not json")
        (rd / "2026-11-17.json").write_text("[]")  # non-dict payload
        with caplog.at_level("WARNING"):
            entries = iter_daily_results(rd)
        assert entries == []
        assert any("skipping" in r.message.lower() for r in caplog.records)

    def test_missing_decision_block_counts_as_no_error(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-16", error=False, decision_block=False)
        entries = iter_daily_results(rd)
        assert entries[0]["error"] is False


class TestCountConsecutiveErrorDays:
    def test_no_files_zero_streak(self, tmp_path):
        out = count_consecutive_error_days(tmp_path, today=date(2026, 11, 18))
        assert out["consecutive_error_days"] == 0
        assert out["today_observed"] is False

    def test_single_error_today(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-18", error=True)
        out = count_consecutive_error_days(rd, today=date(2026, 11, 18))
        assert out["consecutive_error_days"] == 1
        assert out["today_is_error"] is True
        assert out["today_observed"] is True

    def test_streak_ignores_interleaved_healthy_day(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-16", error=True)
        _write_daily(rd, "2026-11-17", error=False)
        _write_daily(rd, "2026-11-18", error=True)
        out = count_consecutive_error_days(rd, today=date(2026, 11, 18))
        assert out["consecutive_error_days"] == 1
        assert out["streak_dates"] == ["2026-11-18"]

    def test_gap_breaks_streak(self, tmp_path):
        # 2026-11-14 was Saturday: no file. An unobserved day is not
        # evidence of health, so the streak must not extend across it.
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-13", error=True)
        _write_daily(rd, "2026-11-16", error=True)  # Monday
        out = count_consecutive_error_days(rd, today=date(2026, 11, 16))
        assert out["consecutive_error_days"] == 1

    def test_future_files_ignored(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-18", error=True)
        _write_daily(rd, "2026-11-19", error=True)  # future — must not extend
        out = count_consecutive_error_days(rd, today=date(2026, 11, 18))
        assert out["consecutive_error_days"] == 1


class TestAssessOutage:
    def test_default_threshold_is_two(self):
        assert DEFAULT_ALERT_DAYS == 2

    def test_alert_at_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTAGE_ALERT_DAYS", "2")
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-17", error=True)
        _write_daily(rd, "2026-11-18", error=True)
        out = assess_outage(rd, today=date(2026, 11, 18))
        assert out["state"] == "ALERT"
        assert out["alert"] is True
        assert out["consecutive_error_days"] == 2
        assert out["streak_dates"] == ["2026-11-18", "2026-11-17"]

    def test_degraded_below_threshold(self, tmp_path, monkeypatch):
        monkeypatch.setenv("OUTAGE_ALERT_DAYS", "3")
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-17", error=True)
        _write_daily(rd, "2026-11-18", error=True)
        out = assess_outage(rd, today=date(2026, 11, 18))
        assert out["state"] == "DEGRADED"
        assert out["alert"] is False

    def test_ok_when_today_healthy_resets(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-17", error=True)
        _write_daily(rd, "2026-11-18", error=False)
        out = assess_outage(rd, today=date(2026, 11, 18))
        assert out["state"] == "OK"
        assert out["consecutive_error_days"] == 0

    def test_in_memory_mode_matches_on_disk(self, tmp_path):
        # Passing today_error in memory must agree with reading today's
        # already-persisted file — the pipeline uses the in-memory mode.
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-17", error=True)
        mem = assess_outage(rd, today=date(2026, 11, 18), today_error=True)
        _write_daily(rd, "2026-11-18", error=True)
        disk = assess_outage(rd, today=date(2026, 11, 18))
        assert mem["consecutive_error_days"] == disk["consecutive_error_days"]
        assert mem["streak_dates"] == disk["streak_dates"]
        assert mem["state"] == disk["state"] == "ALERT"

    def test_in_memory_mode_healthy_today_resets(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-17", error=True)
        out = assess_outage(rd, today=date(2026, 11, 18), today_error=False)
        assert out["state"] == "OK"
        assert out["consecutive_error_days"] == 0
        assert out["streak_dates"] == []

    def test_in_memory_mode_ignores_unwritten_today_file(self, tmp_path):
        # A stale/paradoxical on-disk file for today must be ignored when
        # the caller provides today_error explicitly.
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-17", error=True)
        _write_daily(rd, "2026-11-18", error=False)  # stale — today is an error
        out = assess_outage(rd, today=date(2026, 11, 18), today_error=True)
        assert out["consecutive_error_days"] == 2

    @pytest.mark.parametrize("bad", ["0", "-1", "abc"])
    def test_invalid_env_threshold_falls_back(self, tmp_path, monkeypatch, bad):
        monkeypatch.setenv("OUTAGE_ALERT_DAYS", bad)
        out = assess_outage(tmp_path, today=date(2026, 11, 18))
        assert out["threshold"] == DEFAULT_ALERT_DAYS


class TestWriteOutageStatus:
    def test_round_trip(self, tmp_path):
        rd = tmp_path / "daily"
        _write_daily(rd, "2026-11-18", error=True)
        status = assess_outage(rd, today=date(2026, 11, 18))
        path = write_outage_status(status, rd.parent)
        assert path == tmp_path / "llm_outage_status.json"
        loaded = json.loads(path.read_text())
        assert loaded["state"] == "DEGRADED"  # 1 day, default threshold 2
        assert loaded["streak_dates"] == ["2026-11-18"]

    def test_creates_parent_dirs(self, tmp_path):
        status = assess_outage(tmp_path / "daily", today=date(2026, 11, 18))
        path = write_outage_status(status, tmp_path / "results")
        assert path.exists()
