"""Refusal discriminators for invalid benchmark cash at state ingress.

Doctrine: three-branch ingress decision tree (PR #64/#60/#81).
Benchmark cash is the monetary base of the comparison baseline — branch 3
(refuse), not branch 1 (per-element drop: no siblings) and not branch 2
(fallback: a parsed file claiming a false value, and every substitute
fabricates — initial_capital resurrects a ghost baseline that rebalance()
then persists via save_state()).

Pre-fix behaviour pinned by stash-verification: NaN/inf/bool/string/missing/
list-shaped cash all silently loaded as fabricated initial_capital (or, for
finite negatives, silently loaded as-is); the file was then overwritten on
the next rebalance, making the fabrication sticky.
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark

STATE_NAME = "equalweight_benchmark_state.json"


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "bench"


def _write_state(state_dir, **overrides):
    state = {
        "shares": {"SPY": 10.0, "GLD": 5.0},
        "cash": 2000.0,
        "start_date": "2026-07-31",
        "last_rebalanced": "2026-07-31T10:00:00",
    }
    state.update(overrides)
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / STATE_NAME
    with open(state_path, "w") as f:
        json.dump(state, f)
    return state_path


def _load(state_dir):
    return LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(state_dir))


# --- Refusal discriminators: every one of these FAILS pre-fix -----------


@pytest.mark.parametrize(
    "bad_cash",
    [
        float("nan"),
        float("inf"),
        float("-inf"),
        True,  # bool launders into €1 of monetary base pre-fix
        False,  # fabricates full initial_capital pre-fix
        "10000.0",  # string
        [10000.0],  # wrong shape
        {"cash": 10000.0},  # wrong shape
    ],
)
def test_refuses_invalid_cash_values(state_dir, bad_cash):
    state_path = _write_state(state_dir, cash=bad_cash)
    before = state_path.read_bytes()

    with pytest.raises(ValueError, match="cash"):
        _load(state_dir)

    assert state_path.read_bytes() == before


def test_refuses_missing_cash_key(state_dir):
    state_path = _write_state(state_dir)
    payload = json.loads(state_path.read_text())
    del payload["cash"]
    state_path.write_text(json.dumps(payload))
    before = state_path.read_bytes()

    with pytest.raises(ValueError, match="cash"):
        _load(state_dir)

    assert state_path.read_bytes() == before


def test_refuses_negative_cash(state_dir):
    # Pre-fix this was the worst silent case: finite, so it loaded AS-IS,
    # and total_value silently shrank by the negated amount.
    state_path = _write_state(state_dir, cash=-500.0)
    before = state_path.read_bytes()

    with pytest.raises(ValueError, match="cash"):
        _load(state_dir)

    assert state_path.read_bytes() == before


def test_error_message_names_field_value_and_file(state_dir):
    state_path = _write_state(state_dir, cash=float("nan"))
    with pytest.raises(ValueError) as exc_info:
        _load(state_dir)
    message = str(exc_info.value)
    assert "cash" in message
    assert "nan" in message
    assert str(state_path) in message


def test_no_mutation_occurs_before_refusal(state_dir):
    # Healthy shares must not even be assigned when cash fails: the raise
    # happens before any attribute mutation, so a caller catching the error
    # never observes a half-loaded object (construction failed outright).
    _write_state(state_dir, cash="garbage")
    with pytest.raises(ValueError):
        _load(state_dir)


# --- Healthy-path pins: pass on BOTH sides (bounded, no discrimination) ---


def test_loads_healthy_state(state_dir):
    _write_state(state_dir, cash=2000.0)
    bm = _load(state_dir)
    assert bm.cash == 2000.0
    assert bm.shares == {"SPY": 10.0, "GLD": 5.0}


def test_loads_zero_cash(state_dir):
    # Bounded boundary: 0.0 is a legitimate all-in state, not corruption.
    _write_state(state_dir, cash=0.0)
    bm = _load(state_dir)
    assert bm.cash == 0.0


def test_loads_integer_cash(state_dir):
    _write_state(state_dir, cash=3000)
    bm = _load(state_dir)
    assert bm.cash == 3000.0


def test_missing_file_starts_fresh(state_dir):
    # Branch 2: no claim — honest fallback, not fabrication.
    bm = _load(state_dir)
    assert bm.cash == 10000.0
    assert bm.shares == {}


def test_quarantined_file_starts_fresh(state_dir):
    # Branch 2: unparseable file quarantines loudly, then fresh start.
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / STATE_NAME).write_text("not json")
    bm = _load(state_dir)
    assert bm.cash == 10000.0
    assert bm.shares == {}
