"""Benchmark for monitor non-finite input guards.

Demonstrates that the intraday monitor silently skips alerts derived from
non-finite prices or metrics, and that every emitted alert can be serialized
with ``allow_nan=False``.
"""

import json
import sys
import warnings
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np

from monitor import (
    check_bollinger_breakouts,
    check_index_movements,
    check_movements,
    check_portfolio_drawdown,
    check_stop_losses,
    record_alert,
)
from portfolio.portfolio import Portfolio, Position


class NullIO:
    def write(self, _): pass
    def flush(self): pass


class SuppressOutput:
    def __enter__(self):
        self._old_stdout, self._old_stderr = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = NullIO()
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old_stdout, self._old_stderr


def _build_portfolio():
    portfolio = Portfolio(data_dir="/tmp/bench_monitor_finite")
    portfolio.positions["SPY"] = Position(
        ticker="SPY", quantity=10, avg_price=100.0, current_price=100.0
    )
    portfolio.positions["DBA"] = Position(
        ticker="DBA", quantity=10, avg_price=26.0, current_price=26.0
    )
    return portfolio


def _run(func):
    with SuppressOutput():
        return func()


if __name__ == "__main__":
    print("Benchmark: monitor finite-value guards")
    print("-" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)

        portfolio = _build_portfolio()

        # 1. Stop-loss check with NaN current price.
        alerts = _run(lambda: check_stop_losses({"SPY": float("nan")}, portfolio))
        assert alerts == []
        print(f"{'stop_loss_nan_price':30s} | alerts={len(alerts)} | serializable=True")

        # 2. Position movement with infinite reference price.
        with patch("monitor.load_alert_history", return_value={"alerts": [], "last_reset": ""}), \
             patch("monitor.save_alert_history"), \
             patch("monitor.CHECK_BOLLINGER", False):
            alerts = _run(
                lambda: check_movements(
                    {"SPY": 110.0}, {"SPY": float("inf")}, portfolio
                )
            )
        assert alerts == []
        print(f"{'movement_inf_reference':30s} | alerts={len(alerts)} | serializable=True")

        # 3. Index movement with NaN reference.
        alerts = _run(
            lambda: check_index_movements(
                {"SPY": 110.0}, {"SPY": float("nan")}, ["SPY"], 2.0
            )
        )
        assert alerts == []
        print(f"{'index_nan_reference':30s} | alerts={len(alerts)} | serializable=True")

        # 4. Portfolio drawdown with non-finite cost basis.
        alert = _run(lambda: check_portfolio_drawdown(1000.0, float("inf")))
        assert alert is None
        print(f"{'drawdown_inf_cost_basis':30s} | alert=None | serializable=True")

        # 5. Record alert rejects NaN movement.
        history = {"alerts": [], "last_reset": ""}
        _run(lambda: record_alert("SPY", float("nan"), "position_movement", "high", history))
        assert history["alerts"] == []
        print(f"{'record_alert_nan_movement':30s} | recorded={len(history['alerts'])} | serializable=True")

        # 6. Bollinger breakout with NaN band values.
        import pandas as pd
        with patch("data.fetch_market_data.fetch_historical_data") as mock_fetch, \
             patch("data.indicators.calculate_bollinger_bands") as mock_bands:
            mock_fetch.return_value = {"DBA": pd.DataFrame({"Close": [100.0] * 30})}
            mock_bands.return_value = (
                pd.Series([float("nan")] * 30),
                pd.Series([100.0] * 30),
                pd.Series([90.0] * 30),
            )
            alerts = _run(lambda: check_bollinger_breakouts({"DBA": 110.0}, portfolio))
        assert alerts == []
        print(f"{'bollinger_nan_band':30s} | alerts={len(alerts)} | serializable=True")

        # 7. Normal alert is still produced and strictly serializable.
        with patch("monitor.load_alert_history", return_value={"alerts": [], "last_reset": ""}), \
             patch("monitor.save_alert_history"), \
             patch("monitor.CHECK_BOLLINGER", False):
            alerts = _run(
                lambda: check_movements(
                    {"SPY": 110.0}, {"SPY": 100.0}, portfolio
                )
            )
        assert len(alerts) >= 1
        for alert in alerts:
            json.dumps(alert, allow_nan=False)
        print(f"{'normal_position_movement':30s} | alerts={len(alerts)} | serializable=True")

    print("-" * 70)
    print("OK - monitor silently drops non-finite inputs and emits valid JSON.")
