"""Tests for the live equal-weight benchmark tracker.

Covers:
- State persistence (JSON-safe, no NaN/Infinity tokens)
- Non-finite price handling
- Zero/negative initial_capital rejection
- Rebalancing math and edge cases
- Load-state sanitization for corrupted files
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from benchmark import LiveEqualWeightBenchmark


@pytest.fixture
def tmp_data_dir(tmp_path):
    return tmp_path / "data"


def test_init_defaults(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    assert bm.initial_capital == 10000.0
    assert bm.cash == 10000.0
    assert bm.shares == {}
    assert bm.start_date is None


def test_reject_non_finite_initial_capital(tmp_data_dir):
    with pytest.raises(ValueError):
        LiveEqualWeightBenchmark(initial_capital=float("nan"), data_dir=str(tmp_data_dir))
    with pytest.raises(ValueError):
        LiveEqualWeightBenchmark(initial_capital=float("inf"), data_dir=str(tmp_data_dir))


def test_reject_zero_or_negative_initial_capital(tmp_data_dir):
    with pytest.raises(ValueError):
        LiveEqualWeightBenchmark(initial_capital=0.0, data_dir=str(tmp_data_dir))
    with pytest.raises(ValueError):
        LiveEqualWeightBenchmark(initial_capital=-1000.0, data_dir=str(tmp_data_dir))


def test_reject_non_finite_cash_buffer(tmp_data_dir):
    with pytest.raises(ValueError):
        LiveEqualWeightBenchmark(
            initial_capital=10000.0,
            data_dir=str(tmp_data_dir),
            target_cash_buffer_pct=float("nan"),
        )


def test_get_value_basic(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    bm.shares = {"SPY": 10.0, "GLD": 5.0}
    bm.cash = 2000.0
    result = bm.get_value({"SPY": 100.0, "GLD": 200.0})

    assert result["cash"] == 2000.0
    assert result["positions_value"] == 2000.0
    assert result["total_value"] == 4000.0
    assert result["total_return_pct"] == -60.0
    assert result["num_positions"] == 2


def test_get_value_ignores_non_finite_prices(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    bm.shares = {"SPY": 10.0, "GLD": 5.0, "FEZ": 2.0}
    bm.cash = 1000.0
    result = bm.get_value({"SPY": 100.0, "GLD": float("nan"), "FEZ": float("inf")})

    assert result["positions_value"] == 1000.0  # only SPY counts
    assert result["total_value"] == 2000.0
    assert result["num_positions"] == 3
    assert "SPY" in result["position_details"]
    assert "GLD" not in result["position_details"]
    assert "FEZ" not in result["position_details"]


def test_get_value_ignores_missing_prices(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    bm.shares = {"SPY": 10.0, "GLD": 5.0}
    result = bm.get_value({"SPY": 100.0})

    assert result["positions_value"] == 1000.0
    assert result["total_value"] == 11000.0


def test_rebalance_initial_allocation(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(
        initial_capital=10000.0,
        data_dir=str(tmp_data_dir),
        target_cash_buffer_pct=10.0,
    )
    result = bm.rebalance({"SPY": 100.0, "GLD": 200.0})

    # 90% invested equally -> 4500 per ticker
    assert pytest.approx(bm.shares["SPY"], rel=1e-9) == 45.0
    assert pytest.approx(bm.shares["GLD"], rel=1e-9) == 22.5
    assert bm.cash == pytest.approx(1000.0, rel=1e-9)
    assert result["num_positions"] == 2
    assert result["total_value"] == pytest.approx(10000.0, rel=1e-9)


def test_rebalance_ignores_non_finite_prices(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(
        initial_capital=10000.0,
        data_dir=str(tmp_data_dir),
        target_cash_buffer_pct=10.0,
    )
    result = bm.rebalance(
        {"SPY": 100.0, "GLD": float("nan"), "FEZ": float("inf"), "TLT": -50.0}
    )

    # Only SPY is valid -> all investable cash goes there
    assert list(bm.shares.keys()) == ["SPY"]
    assert bm.shares["SPY"] == pytest.approx(90.0, rel=1e-9)
    assert bm.cash == pytest.approx(1000.0, rel=1e-9)
    assert result["total_value"] == pytest.approx(10000.0, rel=1e-9)


def test_rebalance_no_valid_tickers(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    result = bm.rebalance({"SPY": float("nan"), "GLD": -10.0})

    assert bm.shares == {}
    assert result["total_value"] == 10000.0
    assert result["cash"] == 10000.0


def test_save_and_load_state(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    bm.rebalance({"SPY": 100.0, "GLD": 200.0})

    # Reload from disk
    bm2 = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    assert bm2.shares == bm.shares
    assert bm2.cash == bm.cash
    assert bm2.start_date == bm.start_date


def test_save_state_uses_strict_json(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    # Manually inject a non-finite value to ensure serialization survives it.
    bm.shares = {"SPY": float("nan")}
    bm.cash = 5000.0
    bm.start_date = "2026-07-31"
    bm.last_rebalanced = "2026-07-31T10:00:00"
    bm.save_state()

    raw = (tmp_data_dir / "equalweight_benchmark_state.json").read_text()
    assert "NaN" not in raw
    assert "Infinity" not in raw

    loaded = json.loads(raw)
    assert loaded["shares"]["SPY"] is None


def test_load_state_sanitizes_non_finite_shares(tmp_data_dir):
    state = {
        "shares": {"SPY": 10.0, "GLD": float("nan"), "FEZ": float("inf"), "BAD": -5.0},
        "cash": 2000.0,
        "start_date": "2026-07-31",
        "last_rebalanced": "2026-07-31T10:00:00",
    }
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_data_dir / "equalweight_benchmark_state.json", "w") as f:
        json.dump(state, f)

    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    assert bm.shares == {"SPY": 10.0}
    assert bm.cash == 2000.0


def test_load_state_sanitizes_non_finite_cash(tmp_data_dir):
    state = {
        "shares": {"SPY": 10.0},
        "cash": float("nan"),
        "start_date": "2026-07-31",
    }
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    with open(tmp_data_dir / "equalweight_benchmark_state.json", "w") as f:
        json.dump(state, f)

    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    assert bm.cash == 10000.0


def test_load_state_handles_corrupted_file(tmp_data_dir):
    tmp_data_dir.mkdir(parents=True, exist_ok=True)
    (tmp_data_dir / "equalweight_benchmark_state.json").write_text("not json")

    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    assert bm.shares == {}
    assert bm.cash == 10000.0


def test_daily_summary(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    bm.rebalance({"SPY": 100.0})
    summary = bm.get_daily_summary({"SPY": 110.0})

    assert summary["date"] is not None
    assert summary["num_positions"] == 1
    assert summary["total_value"] > 0
    assert "start_date" in summary
    assert "last_rebalanced" in summary


def test_numpy_scalar_prices_accepted(tmp_data_dir):
    bm = LiveEqualWeightBenchmark(initial_capital=10000.0, data_dir=str(tmp_data_dir))
    bm.shares = {"SPY": np.float64(10.0)}
    result = bm.get_value({"SPY": np.float64(100.0)})
    assert result["positions_value"] == pytest.approx(1000.0)
