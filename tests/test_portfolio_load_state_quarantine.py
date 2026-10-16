"""Quarantine contract for corrupt portfolio state files (PR #85).

Companion to tests/test_portfolio_load_state_guard.py (per-position drops)
and tests/test_portfolio_load_state_monetary_guard.py (parseable monetary
refusal). This module pins the *file-level* boundary between them:

Layering (the PR #81 ingress tree at two resolutions, with the PR #84
strict reader in the parse layer):

- Unparseable file — garbage bytes, or NaN/Infinity tokens (data no repo
  writer can emit: every writer is strict ``allow_nan=False``) — makes no
  usable claim. ``load_json_dict_or_quarantine`` renames it aside as
  ``portfolio_state.json.corrupt-<timestamp>`` (bytes preserved for
  recovery), logs one loud error, and returns an empty dict. The
  constructor then takes the honest no-claim path: fresh portfolio, same
  as "no file". Pre-fix the corrupt file was LEFT IN PLACE and the next
  ``save_state()`` silently overwrote it — the PR #60 class: evidence of
  the corruption destroyed by the system's own write path.
- Valid JSON with a non-mapping top level (``null``, a list, a scalar) —
  same class: no claim about cash or positions, quarantined whole.
- A *parseable* dict claiming an invalid monetary base (string/bool/
  negative/absent cash) still REFUSES with ValueError — that contract is
  pinned in test_portfolio_load_state_monetary_guard.py and asserted
  here once as a layering cross-check: quarantine owns unparseable,
  refusal owns parseable lies, and neither swallows the other's class.
- An empty mapping ``{}`` is indistinguishable from the quarantine
  helper's output at this layer; per the PR #82 claim/no-claim doctrine
  it takes the honest fresh-start path (bounded pin below).

Discriminators assert the mechanism: quarantine rename happened, the
error log names the file, the original bytes survive under the new name,
the portfolio is fresh, and the original path is free for the next
``save_state()`` to write a clean file.
"""

import json
import logging
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio

STATE_NAME = "portfolio_state.json"

_HEALTHY = {
    "cash": 2295.17,
    "total_realized_pnl": -112.50,
    "positions": {
        "SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": 410.0}
    },
}


def _write_raw(tmpdir: str, text: str) -> Path:
    path = Path(tmpdir, STATE_NAME)
    path.write_text(text)
    return path


def _write_token_state(tmpdir: str, token_key: str, token: str) -> Path:
    """Write a state file with a literal NaN/Infinity token for *token_key*.

    ``json.dump(..., allow_nan=False)`` cannot produce the token, so the
    file is dumped with a placeholder and replaced — same trick as the
    PR #80/#84 repros: a NaN token is data no repo code can write but a
    lenient ``json.load`` accepts by default.
    """
    state = dict(_HEALTHY)
    state[token_key] = "__TOKEN_PLACEHOLDER__"
    text = json.dumps(state, indent=2)
    text = text.replace('"__TOKEN_PLACEHOLDER__"', token)
    return _write_raw(tmpdir, text)


def _quarantine_files(tmpdir: str):
    return sorted(Path(tmpdir).glob(f"{STATE_NAME}.corrupt-*"))


def _assert_quarantined(tmpdir: str, caplog, original_bytes: bytes, *, reason_frag: str):
    """Shared mechanism assertions for the whole-file quarantine path.

    Constructs the Portfolio first — the constructor's load_state is what
    triggers the quarantine — then asserts the mechanism.
    """
    portfolio = Portfolio(data_dir=tmpdir)
    # The corrupt file was renamed aside, not left in place to be
    # silently overwritten by the next save_state().
    assert not Path(tmpdir, STATE_NAME).exists()
    quarantines = _quarantine_files(tmpdir)
    assert len(quarantines) == 1
    # Bytes preserved for recovery.
    assert quarantines[0].read_bytes() == original_bytes
    # Loud: one error log naming the file and the quarantine target.
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(errors) == 1
    assert STATE_NAME in errors[0].getMessage()
    assert "quarantined" in errors[0].getMessage()
    assert reason_frag in errors[0].getMessage()
    # Fresh, honest portfolio — the no-claim path.
    assert portfolio.cash == Portfolio.INITIAL_CAPITAL
    assert portfolio.total_realized_pnl == 0.0
    assert portfolio.positions == {}
    return portfolio


class TestUnparseableFileQuarantined:
    """Garbage bytes / non-standard tokens → whole-file quarantine."""

    def test_garbage_bytes_quarantined(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        original = b"this is not json at all {{{"
        _write_raw(data_dir, original.decode())

        _assert_quarantined(data_dir, caplog, original, reason_frag="unreadable")

    def test_nan_cash_token_quarantined(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        path = _write_token_state(data_dir, "cash", "NaN")
        original = path.read_bytes()

        _assert_quarantined(data_dir, caplog, original, reason_frag="unreadable")

    def test_nan_nested_quantity_token_quarantined(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        state = {
            "cash": 5000.0,
            "total_realized_pnl": 0.0,
            "positions": {"SPY": {"quantity": "__TOKEN_PLACEHOLDER__", "avg_price": 400.0, "current_price": 410.0}},
        }
        text = json.dumps(state, indent=2).replace('"__TOKEN_PLACEHOLDER__"', "NaN")
        path = _write_raw(data_dir, text)
        original = path.read_bytes()

        _assert_quarantined(data_dir, caplog, original, reason_frag="unreadable")

    def test_infinity_realized_pnl_token_quarantined(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        path = _write_token_state(data_dir, "total_realized_pnl", "Infinity")
        original = path.read_bytes()

        _assert_quarantined(data_dir, caplog, original, reason_frag="unreadable")


class TestWrongShapeQuarantined:
    """Valid JSON that is not a mapping makes no claim → quarantine."""

    @pytest.mark.parametrize("payload", ["[1, 2, 3]", "null", '"just a string"', "42"])
    def test_non_mapping_top_level_quarantined(self, tmp_path, caplog, payload):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        path = _write_raw(data_dir, payload)
        original = path.read_bytes()

        _assert_quarantined(data_dir, caplog, original, reason_frag="wrong shape")


class TestFreshStartPaths:
    """Bounded pins — honest no-claim paths must stay quiet and clean."""

    def test_missing_file_is_fresh_start_no_quarantine(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        portfolio = Portfolio(data_dir=data_dir)
        assert portfolio.cash == Portfolio.INITIAL_CAPITAL
        assert portfolio.positions == {}
        assert _quarantine_files(data_dir) == []
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    def test_empty_mapping_is_fresh_start_no_quarantine(self, tmp_path, caplog):
        # Claim/no-claim doctrine (PR #82): `{}` is indistinguishable from
        # the quarantine helper's output at this layer, so it takes the
        # same honest fresh-start path. Absence-of-cash refusal applies
        # only to truthy claim-shaped dicts (see the layering test below).
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        _write_raw(data_dir, "{}")
        portfolio = Portfolio(data_dir=data_dir)
        assert portfolio.cash == Portfolio.INITIAL_CAPITAL
        assert portfolio.positions == {}
        assert _quarantine_files(data_dir) == []
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []

    def test_quarantined_file_does_not_block_clean_reload_cycle(self, tmp_path, caplog):
        # The quarantined bytes are preserved, the original path is free,
        # and the next save_state writes a clean file that loads normally.
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        path = _write_token_state(data_dir, "cash", "NaN")
        original = path.read_bytes()

        portfolio = _assert_quarantined(
            data_dir, caplog, original, reason_frag="unreadable"
        )
        portfolio.save_state()
        assert Path(data_dir, STATE_NAME).exists()
        reloaded = Portfolio(data_dir=data_dir)
        assert reloaded.cash == Portfolio.INITIAL_CAPITAL
        assert reloaded.positions == {}
        # One quarantine total — the reload of the clean file must not
        # trigger another.
        assert len(_quarantine_files(data_dir)) == 1


class TestHealthyContractUnchanged:
    """Healthy files must not be quarantined and must load exactly."""

    def test_healthy_state_loads_exactly_no_quarantine(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        with open(Path(data_dir, STATE_NAME), "w") as f:
            json.dump(_HEALTHY, f, indent=2, allow_nan=False)

        portfolio = Portfolio(data_dir=data_dir)
        assert portfolio.cash == 2295.17
        assert portfolio.total_realized_pnl == -112.50
        assert set(portfolio.positions) == {"SPY"}
        assert portfolio.positions["SPY"].quantity == 10.0
        assert _quarantine_files(data_dir) == []
        assert [r for r in caplog.records if r.levelno >= logging.ERROR] == []


class TestParseableRefusalLayering:
    """Cross-check: quarantine must not swallow the parseable-refusal class."""

    def test_string_cash_still_refuses_not_quarantines(self, tmp_path, caplog):
        caplog.set_level(logging.ERROR)
        data_dir = str(tmp_path)
        state = dict(_HEALTHY, cash="2295.17")
        path = Path(data_dir, STATE_NAME)
        with open(path, "w") as f:
            json.dump(state, f, indent=2, allow_nan=False)
        original = path.read_bytes()

        with pytest.raises(ValueError, match="'cash' is '2295.17'"):
            Portfolio(data_dir=data_dir)
        # Refusal, not quarantine: file untouched, still at the original
        # path, no rename, no quarantine log.
        assert path.read_bytes() == original
        assert _quarantine_files(data_dir) == []
        errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
        assert errors == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
