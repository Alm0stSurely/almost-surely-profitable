"""Ingress tests for ``analysis.decision_memory`` (PR #96).

Pins the post-fix contract of ``DecisionMemory._load_memory``:

- file-level: unparseable bytes, non-standard numeric tokens (NaN /
  Infinity), and wrong-shaped top levels are quarantined to
  ``.corrupt-<timestamp>`` (bytes preserved, one ERROR log on the ``utils``
  logger) with an honest empty memory — the file makes no claim;
- element-level: one malformed record (non-dict, missing required field,
  unknown key, non-string date, date outside the ``%Y-%m-%d`` codomain its
  consumers compute in) drops loudly with its index and reason while valid
  siblings survive — never the pre-fix whole-memory wipe that the next
  ``save_memory()`` persisted;
- healthy paths stay byte-identical and quiet.

Discriminators are the sibling-survival checks (they fail on pre-fix code,
where any single malformed element voided every record via the blanket
``except``); the name/index loudness pins pass on both sides when the
failure is loud, and bound the post-fix log contract.
"""

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from analysis.decision_memory import DecisionMemory


def _valid_record(**overrides):
    rec = {
        "date": "2026-10-20",
        "ticker": "SPY",
        "action": "buy",
        "quantity": 10.0,
        "price": 100.0,
        "portfolio_value_before": 10000.0,
        "portfolio_value_after": 9900.0,
    }
    rec.update(overrides)
    return rec


class TestFileLevelQuarantine:
    def test_corrupt_bytes_quarantined_with_error_log(self, tmp_path, caplog):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_bytes(b"not json at all\x00\xff")
        with caplog.at_level(logging.ERROR, logger="utils"):
            mem = DecisionMemory(str(memory_file))
        assert mem.decisions == []
        quarantines = list(tmp_path.glob("decision_memory.json.corrupt-*"))
        assert len(quarantines) == 1
        assert b"not json" in quarantines[0].read_bytes()
        assert not memory_file.exists()
        errors = [r for r in caplog.records if r.levelno == logging.ERROR]
        assert len(errors) == 1
        assert "decision memory" in errors[0].getMessage()

    def test_nan_token_quarantined_by_strict_hook(self, tmp_path, caplog):
        memory_file = tmp_path / "decision_memory.json"
        payload = json.dumps([_valid_record()]).replace("100.0", "NaN", 1)
        memory_file.write_text(payload)
        with caplog.at_level(logging.ERROR, logger="utils"):
            mem = DecisionMemory(str(memory_file))
        assert mem.decisions == []
        assert list(tmp_path.glob("decision_memory.json.corrupt-*"))

    def test_top_level_dict_quarantined(self, tmp_path, caplog):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text(json.dumps({"decisions": [_valid_record()]}))
        with caplog.at_level(logging.ERROR, logger="utils"):
            mem = DecisionMemory(str(memory_file))
        assert mem.decisions == []
        assert list(tmp_path.glob("decision_memory.json.corrupt-*"))

    def test_top_level_null_quarantined(self, tmp_path, caplog):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text("null")
        with caplog.at_level(logging.ERROR, logger="utils"):
            mem = DecisionMemory(str(memory_file))
        assert mem.decisions == []
        assert list(tmp_path.glob("decision_memory.json.corrupt-*"))


class TestPerRecordDrops:
    def test_non_dict_element_dropped_loudly_siblings_survive(
        self, tmp_path, capsys
    ):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text(json.dumps(
            [_valid_record(), "a string record", _valid_record(ticker="GLD")]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 2
        assert "Dropping malformed decision record at index 1" in captured
        assert "expected dict, got str" in captured
        assert "Dropped 1 malformed record(s)" in captured

    def test_missing_required_field_dropped_sibling_survives(
        self, tmp_path, capsys
    ):
        memory_file = tmp_path / "decision_memory.json"
        bad = _valid_record()
        del bad["price"]
        memory_file.write_text(json.dumps([_valid_record(), bad]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 1
        assert mem.decisions[0].ticker == "SPY"
        assert "price" in captured

    def test_unknown_key_dropped_sibling_survives(self, tmp_path, capsys):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text(json.dumps(
            [_valid_record(), _valid_record(schema_version=2)]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 1
        assert "schema_version" in captured

    def test_wrong_codomain_date_dropped_sibling_survives(
        self, tmp_path, capsys
    ):
        # A full ISO datetime is a valid JSON string, but every consumer
        # computes with datetime.strptime(d.date, "%Y-%m-%d") outside any
        # try — parseable here, detonating there (PR #88 class).
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text(json.dumps(
            [_valid_record(), _valid_record(date="2026-10-20T21:30:00")]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 1
        assert "2026-10-20T21:30:00" in captured

    def test_non_string_date_dropped_sibling_survives(self, tmp_path, capsys):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text(json.dumps(
            [_valid_record(), _valid_record(date=20261020)]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 1
        assert "20261020" in captured

    def test_multiple_bad_records_each_named(self, tmp_path, capsys):
        memory_file = tmp_path / "decision_memory.json"
        bad_missing = _valid_record()
        del bad_missing["ticker"]
        memory_file.write_text(json.dumps(
            [_valid_record(), "junk", bad_missing, 42.0,
             _valid_record(ticker="GLD")]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 2
        assert "index 1" in captured
        assert "index 2" in captured
        assert "index 3" in captured
        assert "Dropped 3 malformed record(s)" in captured


class TestPersistenceAfterDrop:
    def test_save_after_drop_writes_only_valid_records(self, tmp_path, capsys):
        memory_file = tmp_path / "decision_memory.json"
        bad = _valid_record()
        del bad["price"]
        memory_file.write_text(json.dumps([_valid_record(), bad]))
        mem = DecisionMemory(str(memory_file))
        mem.save_memory()
        reloaded = DecisionMemory(str(memory_file))
        assert len(reloaded.decisions) == 1
        assert reloaded.decisions[0].ticker == "SPY"
        # Round-trip is strict and finite per the existing writer contract.
        data = json.loads(memory_file.read_text())
        assert len(data) == 1
        assert data[0]["ticker"] == "SPY"


class TestHealthyPathsUnchanged:
    def test_healthy_file_loads_quietly(self, tmp_path, capsys):
        memory_file = tmp_path / "decision_memory.json"
        memory_file.write_text(json.dumps(
            [_valid_record(), _valid_record(ticker="GLD", action="hold")]))
        mem = DecisionMemory(str(memory_file))
        captured = capsys.readouterr().out
        assert len(mem.decisions) == 2
        assert "Dropping malformed" not in captured
        assert "Loaded 2 decisions from memory" in captured
        assert not list(tmp_path.glob("decision_memory.json.corrupt-*"))

    def test_missing_file_empty_and_quiet(self, tmp_path, capsys):
        mem = DecisionMemory(str(tmp_path / "decision_memory.json"))
        captured = capsys.readouterr().out
        assert mem.decisions == []
        assert "Warning" not in captured
        assert "Dropping" not in captured

    def test_round_trip_bytes_stable(self, tmp_path):
        memory_file = tmp_path / "decision_memory.json"
        records = [_valid_record(), _valid_record(ticker="GLD", action="sell")]
        memory_file.write_text(json.dumps(records))
        mem = DecisionMemory(str(memory_file))
        mem.save_memory()
        reloaded = DecisionMemory(str(memory_file))
        assert [d.ticker for d in reloaded.decisions] == ["SPY", "GLD"]
        assert [d.action for d in reloaded.decisions] == ["buy", "sell"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
