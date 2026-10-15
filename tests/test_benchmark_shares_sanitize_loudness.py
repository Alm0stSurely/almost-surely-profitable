"""Loudness tests for the benchmark shares sanitizer.

Doctrine: PR #64 per-element drop — one loud warning per dropped element,
never silent. Pre-fix, ``LiveEqualWeightBenchmark._sanitize_shares`` dropped
malformed entries through a silent dict comprehension: the pruned dict was
then persisted by the next ``save_state()``, destroying the evidence of
which positions the corrupt file had claimed. The discriminators below
assert the *loudness* (the drop itself is unchanged behaviour and would
pass pre-fix); the healthy-path pins assert expected absences stay quiet
and must pass on BOTH sides of a fail-on-old stash.

Note: since the strict-reader convention (``parse_constant`` rejection of
NaN/Infinity tokens, pinned generically in
``tests/test_utils_strict_json_reader.py``), a file containing non-finite
tokens is quarantined whole at parse time — non-finite quantities are no
longer reachable at this layer. The malformed-value discriminators below
use *parseable* invalids (bool, string, None, negative); the per-entry
drop + loudness contract is unchanged for those.
"""

import json
import logging
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark

STATE_NAME = "equalweight_benchmark_state.json"
LOGGER_NAME = "benchmark"


@pytest.fixture
def state_dir(tmp_path):
    return tmp_path / "bench"


def _write_state(state_dir, shares, cash=2000.0):
    state = {
        "shares": shares,
        "cash": cash,
        "start_date": "2026-07-31",
        "last_rebalanced": "2026-07-31T10:00:00",
    }
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / STATE_NAME
    with open(state_path, "w") as f:
        json.dump(state, f)
    return state_path


def _load(state_dir):
    return LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(state_dir))


# --- Loudness discriminators: every one of these FAILS pre-fix ------------
# Pre-fix the module carried no logger at all, so caplog records nothing.


@pytest.mark.parametrize(
    "bad_quantity",
    [
        True,  # bool launders: True * price == price
        "10",  # string parses JSON but is not numeric
        None,
    ],
)
def test_malformed_quantity_dropped_loudly(
    state_dir, caplog, bad_quantity
):
    _write_state(state_dir, {"SPY": bad_quantity, "GLD": 5.0})
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert "SPY" not in bench.shares
    assert bench.shares == {"GLD": 5.0}  # healthy sibling survives (blast radius)
    spy_warnings = [
        r for r in caplog.records if "SPY" in r.getMessage()
    ]
    assert len(spy_warnings) == 1
    assert "not a finite number" in spy_warnings[0].getMessage()


def test_negative_quantity_dropped_loudly(state_dir, caplog):
    _write_state(state_dir, {"SPY": -5.0, "GLD": 5.0})
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert "SPY" not in bench.shares
    assert bench.shares == {"GLD": 5.0}
    spy_warnings = [r for r in caplog.records if "SPY" in r.getMessage()]
    assert len(spy_warnings) == 1
    assert "negative" in spy_warnings[0].getMessage()


@pytest.mark.parametrize("wrong_shape", [["SPY", 10.0], "SPY:10", 42])
def test_non_mapping_shares_loudly_treated_as_empty(state_dir, caplog, wrong_shape):
    _write_state(state_dir, wrong_shape)
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert bench.shares == {}
    messages = [r.getMessage() for r in caplog.records]
    assert any("'shares' is not a mapping" in m for m in messages)


def test_every_dropped_entry_gets_its_own_warning(state_dir, caplog):
    """Lipschitz loudness: one warning per element, not one for the batch."""
    _write_state(
        state_dir,
        {"AAA": None, "BBB": True, "CCC": -1.0, "DDD": "x", "GLD": 5.0},
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert bench.shares == {"GLD": 5.0}
    dropped = {"AAA", "BBB", "CCC", "DDD"}
    warned = {
        r.getMessage().split("entry ")[1].split(":")[0].strip("'\"")
        for r in caplog.records
        if "Dropping benchmark share entry" in r.getMessage()
    }
    assert dropped <= warned


# --- Bounded healthy-path pins: pass on BOTH sides of the stash -----------


def test_all_valid_shares_load_silently(state_dir, caplog):
    _write_state(state_dir, {"SPY": 10.0, "GLD": 5, "TLT": 0.0})
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert bench.shares == {"SPY": 10.0, "GLD": 5.0, "TLT": 0.0}
    assert caplog.records == []


def test_absent_shares_key_is_quiet_fresh_start(state_dir, caplog):
    """No-claim boundary (PR #81): an absent key makes no claim."""
    _write_state(state_dir, None)
    state_path = state_dir / STATE_NAME
    state = json.loads(state_path.read_text())
    del state["shares"]
    state_path.write_text(json.dumps(state))
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert bench.shares == {}
    assert caplog.records == []


def test_empty_shares_mapping_is_quiet(state_dir, caplog):
    _write_state(state_dir, {})
    with caplog.at_level(logging.WARNING, logger=LOGGER_NAME):
        bench = _load(state_dir)
    assert bench.shares == {}
    assert caplog.records == []
