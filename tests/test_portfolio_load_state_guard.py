"""Regression tests for the Portfolio.load_state ingress guard.

Contract: position fields (quantity, avg_price, current_price) read from the
state JSON must be validated at the ingress boundary, before they reach
Position arithmetic. Pre-fix they were passed straight through:

- ``bool`` is an int subclass: ``quantity=True`` laundered into a fabricated
  market value (``True * price == price``); the PR #79 note in
  test_bool_guard_rejection.py flagged exactly this ingress as the follow-up.
- ``json.load`` accepts ``NaN``/``Infinity`` tokens by default: a NaN
  quantity loaded silently and killed the *next* ``save_state`` (strict
  ``allow_nan=False``) — corruption in, crash out, far from the cause.
- a string ``avg_price`` loaded fine and crashed later, at first P&L access.
- a list-shaped position entry crashed the whole load via the outer
  ``except`` — which also reset cash to INITIAL_CAPITAL (blast radius far
  beyond the malformation).

Scope of failure must match scope of malformation (PR #64 doctrine): one
malformed position is dropped loudly, healthy siblings survive, and the
persisted cash is preserved. The outer ``except`` stays reserved for global
failure (unparseable file).

Discriminators assert the malformed entry is refused loudly; bounded
healthy-path pins assert real states round-trip byte-identically on both
sides of the fix.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio


def _write_state(tmpdir: str, positions, cash: float = 5000.0) -> None:
    """Write a state file by hand so non-standard tokens (NaN/Infinity) work.

    ``json.dump(..., allow_nan=False)`` — rightly — refuses these tokens,
    so the file is assembled manually; only tests need the raw forms.
    """
    state = {
        "cash": cash,
        "total_realized_pnl": 0.0,
        "positions": positions,
        "last_updated": "2026-10-11T00:00:00",
        "total_value": cash,
    }
    text = json.dumps(state, indent=2)
    for pos in (positions.values() if isinstance(positions, dict) else []):
        if not isinstance(pos, dict):
            continue
        for value in pos.values():
            if isinstance(value, float):
                if math.isnan(value):
                    text = text.replace(json.dumps(value), "NaN", 1)
                elif value == float("inf"):
                    text = text.replace(json.dumps(value), "Infinity", 1)
                elif value == float("-inf"):
                    text = text.replace(json.dumps(value), "-Infinity", 1)
    Path(tmpdir, "portfolio_state.json").write_text(text)


HEALTHY = {"quantity": 5.0, "avg_price": 200.0, "current_price": 210.0}


class TestLoadStateRejectsBoolFields:
    """bool quantity / avg_price / current_price must be dropped loudly."""

    def test_bool_quantity_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": True, "avg_price": 400.0, "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions
        out = capsys.readouterr().out
        assert "SPY" in out and "quantity" in out

    def test_bool_quantity_log_names_field(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": True, "avg_price": 400.0, "current_price": 410.0}})
            Portfolio(data_dir=tmpdir)
        out = capsys.readouterr().out
        assert "SPY" in out and "quantity" in out

    def test_bool_avg_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": True, "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions
        out = capsys.readouterr().out
        assert "SPY" in out and "avg_price" in out

    def test_bool_current_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": True}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions
        out = capsys.readouterr().out
        assert "current_price" in out


class TestLoadStateRejectsNonFiniteFields:
    """NaN/Infinity tokens (accepted by json.load) must not reach Position."""

    def test_nan_quantity_dropped_and_save_survives(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": float("nan"), "avg_price": 400.0, "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
            assert "SPY" not in portfolio.positions
            assert math.isfinite(portfolio.total_value)
            # The downstream kill is gone: saving the cleaned state must work.
            portfolio.save_state()
            reloaded = Portfolio(data_dir=tmpdir)
            assert reloaded.positions == {}

    def test_infinity_avg_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": float("inf"), "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions

    def test_negative_inf_current_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": float("-inf")}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions

    def test_nan_current_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": float("nan")}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions


class TestLoadStateRejectsWrongTypesAndShapes:
    """Strings, missing keys, and non-dict entries must be dropped loudly."""

    def test_string_avg_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": "100", "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" not in portfolio.positions

    def test_missing_quantity_dropped_sibling_survives(self, capsys):
        # Pre-fix this raised KeyError inside the outer except, which reset
        # the ENTIRE state (cash included) — the blast-radius assertions make
        # this a true discriminator, not an accidental pre-fix pass.
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {
                "SPY": {"avg_price": 400.0, "current_price": 410.0},
                "GLD": dict(HEALTHY),
            }, cash=5000.0)
            portfolio = Portfolio(data_dir=tmpdir)
        assert set(portfolio.positions) == {"GLD"}
        assert portfolio.cash == 5000.0

    def test_missing_avg_price_dropped_sibling_survives(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {
                "SPY": {"quantity": 10.0, "current_price": 410.0},
                "GLD": dict(HEALTHY),
            }, cash=5000.0)
            portfolio = Portfolio(data_dir=tmpdir)
        assert set(portfolio.positions) == {"GLD"}
        assert portfolio.cash == 5000.0

    def test_list_shaped_entry_dropped_cash_preserved(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": [10.0, 400.0]}, cash=5000.0)
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}
        # Blast-radius pin: pre-fix the outer except reset cash to
        # INITIAL_CAPITAL; the per-element guard must preserve it.
        assert portfolio.cash == 5000.0

    def test_scalar_shaped_entry_dropped_cash_preserved(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": 10.0}, cash=5000.0)
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}
        assert portfolio.cash == 5000.0

    def test_non_dict_positions_map_ignored_cash_preserved(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, ["SPY", "GLD"], cash=5000.0)
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}
        assert portfolio.cash == 5000.0

    def test_zero_quantity_dropped(self, capsys):
        # sell() removes positions at >= 99.99%, so a zero-quantity record is
        # degenerate — but the guard must not crash on it.
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 0.0, "avg_price": 400.0, "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}

    def test_negative_quantity_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": -5.0, "avg_price": 400.0, "current_price": 410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}

    def test_negative_current_price_dropped(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": -410.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}


class TestLoadStateHealthyPathsUnchanged:
    """Bounded pins: every healthy shape round-trips exactly as before."""

    def test_healthy_round_trip_exact(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": dict(HEALTHY), "GLD": {"quantity": 3, "avg_price": 180, "current_price": 190}})
            portfolio = Portfolio(data_dir=tmpdir)
            assert set(portfolio.positions) == {"SPY", "GLD"}
            spy = portfolio.positions["SPY"]
            assert spy.quantity == 5.0 and spy.avg_price == 200.0 and spy.current_price == 210.0
            # int fields (JSON has no float/int distinction for whole numbers
            # written by other tools) are accepted as-is.
            assert portfolio.positions["GLD"].quantity == 3
            assert portfolio.positions["GLD"].market_value == 3 * 190

    def test_current_price_absent_defaults_zero(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 5.0, "avg_price": 200.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" in portfolio.positions
        assert portfolio.positions["SPY"].current_price == 0.0

    def test_current_price_explicit_zero_accepted(self, capsys):
        # 0.0 is the documented "price unknown" default — the lower bound is
        # inclusive for current_price, unlike quantity/avg_price.
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": {"quantity": 5.0, "avg_price": 200.0, "current_price": 0.0}})
            portfolio = Portfolio(data_dir=tmpdir)
        assert "SPY" in portfolio.positions
        assert portfolio.positions["SPY"].current_price == 0.0

    def test_malformed_sibling_survives(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {
                "SPY": {"quantity": True, "avg_price": 400.0, "current_price": 410.0},
                "GLD": dict(HEALTHY),
            })
            portfolio = Portfolio(data_dir=tmpdir)
        assert set(portfolio.positions) == {"GLD"}
        assert portfolio.positions["GLD"].market_value == 5.0 * 210.0
        out = capsys.readouterr().out
        assert "SPY" in out  # the drop was loud

    def test_all_malformed_loads_empty_loudly(self, capsys):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {
                "SPY": {"quantity": float("nan"), "avg_price": 400.0, "current_price": 410.0},
                "GLD": {"quantity": 5.0, "avg_price": "200", "current_price": 210.0},
            })
            portfolio = Portfolio(data_dir=tmpdir)
        assert portfolio.positions == {}
        out = capsys.readouterr().out
        assert out.count("Dropping position") == 2

    def test_healthy_state_loads_quietly(self, capsys):
        # No drop noise on a fully healthy file (failure != absence).
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_state(tmpdir, {"SPY": dict(HEALTHY)})
            Portfolio(data_dir=tmpdir)  # constructor performs the load
        out = capsys.readouterr().out
        assert "Dropping" not in out
        assert "Ignoring" not in out
