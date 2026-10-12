"""Ingress guards for the portfolio monetary base (cash + realized P&L).

Companion to tests/test_portfolio_load_state_guard.py (PR #80), which pins
the per-position drop doctrine. This module pins the *global refusal*
doctrine for the two session-global scalars:

- ``cash`` is the monetary base: every order size, portfolio weight, and
  LLM prompt field derives from it. A malformed cash (NaN, bool, string,
  negative, or absent) therefore poisons the entire session — and there is
  no non-fabricating substitute. Refusal (ValueError) beats fabrication.
- ``total_realized_pnl`` is a session-global aggregate feeding the LLM
  prompt and the strict ``save_state``: NaN loads silently but kills the
  NEXT save (allow_nan=False), i.e. the session crashes at the write
  site, maximally far from the cause. Same refusal doctrine.

Both keys are written unconditionally by ``save_state``, so a missing or
malformed value always means a corrupted or hand-edited state file.

Discriminators (must FAIL pre-fix, i.e. raise nothing) assert the refusal
mechanism itself: ``ValueError`` propagates out of ``Portfolio()`` and the
state file on disk is byte-untouched. Bounded pins (must pass on BOTH
sides of the fix) hold the healthy contract exactly: 0.0 cash / 0.0
realized P&L are legitimate states, not malformations.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from portfolio.portfolio import Portfolio


def _write_state(tmp_path, state: dict) -> str:
    """Write a state file, returning the data dir path."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    with open(data_dir / "portfolio_state.json", "w") as f:
        json.dump(state, f, indent=2, allow_nan=False)
    return str(data_dir)


def _write_state_with_nan(tmp_path, nan_key: str, state: dict) -> str:
    """Write a state file with a literal NaN token for *nan_key*.

    json.dump(..., allow_nan=False) cannot produce the token, so the
    fixture is dumped with a placeholder and replaced (same trick as the
    PR #80 repro): a NaN token is data no repo code can write but
    json.load accepts by default.
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    state = dict(state)
    state[nan_key] = "__NAN_PLACEHOLDER__"
    text = json.dumps(state, indent=2)
    text = text.replace('"__NAN_PLACEHOLDER__"', "NaN")
    with open(data_dir / "portfolio_state.json", "w") as f:
        f.write(text)
    return str(data_dir)


_HEALTHY = {
    "cash": 2295.17,
    "total_realized_pnl": -112.50,
    "positions": {
        "SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": 410.0}
    },
}


class TestCashRefusal:
    """Invalid cash must refuse the whole load — never fabricate."""

    def test_nan_cash_refused(self, tmp_path):
        data_dir = _write_state_with_nan(tmp_path, "cash", _HEALTHY)
        with pytest.raises(ValueError, match="'cash' is nan"):
            Portfolio(data_dir=data_dir)

    def test_bool_cash_refused(self, tmp_path):
        state = dict(_HEALTHY, cash=True)
        data_dir = _write_state(tmp_path, state)
        # Pre-fix: True launders into a €1.00 monetary base — every order is
        # sized on €1 and blocked as "too small": a silent, total block.
        with pytest.raises(ValueError, match="'cash' is True"):
            Portfolio(data_dir=data_dir)

    def test_string_cash_refused(self, tmp_path):
        state = dict(_HEALTHY, cash="2295.17")
        data_dir = _write_state(tmp_path, state)
        # Pre-fix: the string loads, then crashes the outer except at the
        # summary print — which RESET cash to INITIAL_CAPITAL and destroyed
        # the verified positions: the ghost-portfolio path.
        with pytest.raises(ValueError, match="'cash' is '2295.17'"):
            Portfolio(data_dir=data_dir)

    def test_negative_cash_refused(self, tmp_path):
        state = dict(_HEALTHY, cash=-500.0)
        data_dir = _write_state(tmp_path, state)
        # Unreachable on the healthy path: buy() rejects total_cost > cash
        # before subtracting, so persisted cash is always >= 0.
        with pytest.raises(ValueError, match="'cash' is -500.0"):
            Portfolio(data_dir=data_dir)

    def test_absent_cash_refused(self, tmp_path):
        state = {k: v for k, v in _HEALTHY.items() if k != "cash"}
        data_dir = _write_state(tmp_path, state)
        # Pre-fix: the .get default fabricated INITIAL_CAPITAL — a real
        # balance silently replaced by the €10,000 fresh-portfolio reading.
        with pytest.raises(ValueError, match="'cash' is None"):
            Portfolio(data_dir=data_dir)


class TestRealizedPnlRefusal:
    """Invalid total_realized_pnl must refuse — NaN kills the next save."""

    def test_nan_realized_pnl_refused(self, tmp_path):
        data_dir = _write_state_with_nan(tmp_path, "total_realized_pnl", _HEALTHY)
        with pytest.raises(ValueError, match="'total_realized_pnl' is nan"):
            Portfolio(data_dir=data_dir)

    def test_bool_realized_pnl_refused(self, tmp_path):
        state = dict(_HEALTHY, total_realized_pnl=True)
        data_dir = _write_state(tmp_path, state)
        # Pre-fix: True launders into a fabricated €1.00 realized P&L,
        # feeding a fiction into the LLM prompt's Total P&L line.
        with pytest.raises(ValueError, match="'total_realized_pnl' is True"):
            Portfolio(data_dir=data_dir)

    def test_string_realized_pnl_refused(self, tmp_path):
        state = dict(_HEALTHY, total_realized_pnl="-112.50")
        data_dir = _write_state(tmp_path, state)
        # Pre-fix: loads fine; total_pnl arithmetic TypeErrors mid-prompt-
        # build — one trading cycle away from the cause.
        with pytest.raises(ValueError, match="'total_realized_pnl' is '-112.50'"):
            Portfolio(data_dir=data_dir)

    def test_absent_realized_pnl_refused(self, tmp_path):
        state = {k: v for k, v in _HEALTHY.items() if k != "total_realized_pnl"}
        data_dir = _write_state(tmp_path, state)
        # Persisted unconditionally since the MVP commit — absence is a
        # corrupted or hand-edited file, not a legacy schema.
        with pytest.raises(ValueError, match="'total_realized_pnl' is None"):
            Portfolio(data_dir=data_dir)


class TestRefusalBlastRadius:
    """The refusal must not destroy, rewrite, or fabricate anything."""

    def test_refusal_leaves_file_untouched(self, tmp_path):
        data_dir = _write_state_with_nan(tmp_path, "cash", _HEALTHY)
        state_file = tmp_path / "data" / "portfolio_state.json"
        before = state_file.read_bytes()
        with pytest.raises(ValueError):
            Portfolio(data_dir=data_dir)
        assert state_file.read_bytes() == before

    def test_refusal_does_not_reset_healthy_positions(self, tmp_path):
        # Blast-radius pin: the positions in the file are healthy, but the
        # monetary base is not — the load refuses BEFORE the position
        # restore, so no in-memory ghost state is ever built.
        data_dir = _write_state_with_nan(tmp_path, "cash", _HEALTHY)
        with pytest.raises(ValueError):
            Portfolio(data_dir=data_dir)


class TestHealthyContract:
    """Bounded pins — must pass on BOTH sides of the fix."""

    def test_healthy_state_loads_exactly(self, tmp_path):
        data_dir = _write_state(tmp_path, _HEALTHY)
        p = Portfolio(data_dir=data_dir)
        assert p.cash == 2295.17
        assert p.total_realized_pnl == -112.50
        assert set(p.positions) == {"SPY"}
        assert p.positions["SPY"].quantity == 10.0

    def test_zero_cash_is_legitimate(self, tmp_path):
        # 0.0 cash = fully invested. The bound is INCLUSIVE — a guard that
        # rejected 0 would block every all-in portfolio.
        state = dict(_HEALTHY, cash=0.0)
        data_dir = _write_state(tmp_path, state)
        p = Portfolio(data_dir=data_dir)
        assert p.cash == 0.0

    def test_zero_realized_pnl_is_legitimate(self, tmp_path):
        # 0.0 realized P&L = "no closed trades yet". Sentinel collision
        # doctrine (PR #65): 0.0 is a real value, not a malformation.
        state = dict(_HEALTHY, total_realized_pnl=0.0)
        data_dir = _write_state(tmp_path, state)
        p = Portfolio(data_dir=data_dir)
        assert p.total_realized_pnl == 0.0

    def test_negative_realized_pnl_is_legitimate(self, tmp_path):
        # Realized losses are a normal state — no lower bound on this field.
        state = dict(_HEALTHY, total_realized_pnl=-987.65)
        data_dir = _write_state(tmp_path, state)
        p = Portfolio(data_dir=data_dir)
        assert p.total_realized_pnl == -987.65

    def test_int_cash_loads(self, tmp_path):
        # JSON has no float/int distinction for whole numbers: a hand edit
        # "cash": 5000 must load, not be refused for typing pedantry.
        state = dict(_HEALTHY, cash=5000)
        data_dir = _write_state(tmp_path, state)
        p = Portfolio(data_dir=data_dir)
        assert p.cash == 5000

    def test_missing_file_is_fresh_start(self, tmp_path):
        # No file at all = the legitimate fresh-portfolio state — this is
        # the ONLY path where INITIAL_CAPITAL applies, and it involves no
        # load at all.
        p = Portfolio(data_dir=str(tmp_path / "data"))
        assert p.cash == Portfolio.INITIAL_CAPITAL
        assert p.total_realized_pnl == 0.0
        assert p.positions == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
