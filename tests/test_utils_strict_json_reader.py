"""Strict-reader tests for the shared JSON ingress helpers.

Doctrine: strict-on-write is worthless without strict-on-read (PR #80
rule). Every writer in the repository serializes with ``allow_nan=False``,
so a file containing the literal tokens ``NaN`` / ``Infinity`` /
``-Infinity`` is data no repo code can emit — a hand-edited or externally
corrupted file. Python's ``json.load`` accepts those tokens by default
and maps them to non-finite floats, which then poison aggregates
downstream and crash the *next* strict write maximally far from the
cause.

The shared ingress helpers in ``utils`` now reject such tokens at the
read boundary:

- ``load_json_list_or_quarantine`` / ``load_json_dict_or_quarantine``
  classify a non-standard token as a parse failure: the file is
  quarantined (renamed aside, bytes preserved) and the caller gets the
  honest fresh-start container. Pre-fix the token sailed through and the
  non-finite floats reached the caller.
- ``load_valid_daily_results`` skips the file with its existing loud
  "Skipping <name>" warning (the bare ``except Exception`` already
  covers the new ``ValueError``).

Discriminators assert the quarantine/skip *behaviour* (mechanism:
parse_constant raising inside the helper's own try); healthy-path pins
assert ordinary files — including ones with legitimate ``null`` values
and non-finite floats serialized as ``null`` by ``dump_json_safe`` —
parse identically on both sides.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import (
    dump_json_safe,
    load_json_dict_or_quarantine,
    load_json_list_or_quarantine,
    load_valid_daily_results,
)

LOGGER_NAME = "utils"


# --------------------------------------------------------------------------
# The hook itself
# --------------------------------------------------------------------------

class TestRejectNonstandardConstant:
    # Imported lazily so the file still collects under fail-on-old (the
    # hook does not exist pre-fix; those tests fail, the behaviour
    # discriminators below still run).
    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_raises_value_error_naming_token(self, token):
        from utils import reject_nonstandard_constant

        with pytest.raises(ValueError, match="non-standard JSON constant"):
            reject_nonstandard_constant(token)

    def test_message_contains_the_token(self):
        from utils import reject_nonstandard_constant

        with pytest.raises(ValueError, match=r"'NaN'"):
            reject_nonstandard_constant("NaN")


# --------------------------------------------------------------------------
# Quarantine helpers: a non-standard token is a parse failure -> quarantine
# --------------------------------------------------------------------------

def _write_raw(path: Path, text: str) -> None:
    path.write_text(text)


class TestListOrQuarantineRejectsNonStandardTokens:
    @pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity"])
    def test_nan_token_file_quarantined(self, tmp_path, caplog, token):
        ledger = tmp_path / "trades_history.json"
        _write_raw(ledger, f'[{{"ticker": "SPY", "price": {token}}}]')

        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            result = load_json_list_or_quarantine(ledger, context="ledger")

        assert result == []
        # File preserved aside, not destroyed, not left in place to be
        # silently overwritten by the next save.
        assert not ledger.exists()
        quarantined = list(tmp_path.glob("trades_history.json.corrupt-*"))
        assert len(quarantined) == 1
        assert "unreadable" in caplog.text
        assert "non-standard JSON constant" in caplog.text

    def test_dict_variant_quarantines_nan_token(self, tmp_path, caplog):
        state = tmp_path / "market_state.json"
        _write_raw(state, '{"SPY": {"price": NaN}}')

        with caplog.at_level(logging.ERROR, logger=LOGGER_NAME):
            result = load_json_dict_or_quarantine(state, context="state")

        assert result == {}
        assert not state.exists()
        assert list(tmp_path.glob("market_state.json.corrupt-*"))
        assert "non-standard JSON constant" in caplog.text


# --------------------------------------------------------------------------
# Healthy-path pins: ordinary content parses identically on both sides
# --------------------------------------------------------------------------

class TestHealthyFilesUnchanged:
    def test_plain_list_round_trips(self, tmp_path):
        ledger = tmp_path / "trades_history.json"
        records = [{"ticker": "SPY", "price": 412.5, "qty": 3}]
        with open(ledger, "w") as f:
            json.dump(records, f, allow_nan=False)

        assert load_json_list_or_quarantine(ledger, context="ledger") == records

    def test_plain_dict_round_trips(self, tmp_path):
        state = tmp_path / "market_state.json"
        payload = {"SPY": {"price": 412.5}, "cached_at": "2026-10-15T09:00:00"}
        with open(state, "w") as f:
            json.dump(payload, f, allow_nan=False)

        assert load_json_dict_or_quarantine(state, context="state") == payload

    def test_dump_json_safe_output_with_nulls_still_loads(self, tmp_path):
        # The repo's own sanitizer writes non-finite floats as ``null``;
        # that output is legitimate and must keep loading.
        state = tmp_path / "state.json"
        with open(state, "w") as f:
            dump_json_safe({"cash": None, "note": "n/a"}, f)

        assert load_json_dict_or_quarantine(state, context="state") == {
            "cash": None,
            "note": "n/a",
        }

    def test_missing_file_returns_empty_containers(self, tmp_path):
        assert load_json_list_or_quarantine(tmp_path / "nope.json") == []
        assert load_json_dict_or_quarantine(tmp_path / "nope.json") == {}


# --------------------------------------------------------------------------
# load_valid_daily_results: non-standard token -> loud skip
# --------------------------------------------------------------------------

class TestValidDailyResultsRejectsNonStandardTokens:
    def _make_result(self, path: Path, text: str) -> None:
        path.write_text(text)

    def test_nan_token_file_skipped_loudly(self, tmp_path, caplog):
        results_dir = tmp_path / "daily"
        results_dir.mkdir()
        good = results_dir / "2026-10-14.json"
        bad = results_dir / "2026-10-15.json"
        good_payload = {
            "date": "2026-10-14",
            "portfolio_summary": {"cash": 100.0, "total_value": 1000.0},
            "llm_response": {},
            "actions": [],
        }
        with open(good, "w") as f:
            json.dump(good_payload, f, allow_nan=False)
        self._make_result(bad, '{"date": "2026-10-15", "cash": NaN}')

        with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
            results = load_valid_daily_results(str(results_dir))

        # Only the healthy file is aggregated; the token file is skipped
        # with a loud reason, not silently ingested.
        assert [r["date"] for r in results] == ["2026-10-14"]
        assert "Skipping 2026-10-15.json" in caplog.text
        assert "non-standard JSON constant" in caplog.text
