"""Tests for cash_drag_report.py."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.cash_drag_report import analyze_cash_drag


def _write_result(
    tmp_path,
    date,
    cash,
    total,
    regime="normal",
    trades=0,
    cap=None,
    trades_this_week=None,
):
    """Helper to build a minimal valid daily result."""
    result = {
        "date": date,
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
        "portfolio_after": {"cash": cash, "total_value": total},
        "executed_trades": [{"ticker": "SPY", "action": "buy"}] * trades,
    }
    status = {
        "current_vol_regime": regime,
        "config": {"current_vol_regime": regime},
    }
    if cap is not None:
        status["weekly_cap"] = cap
        status["trades_this_week"] = trades_this_week
    result["cooldown"] = {"status": status}
    with open(tmp_path / f"{date}.json", "w") as f:
        json.dump(result, f)


def test_cash_drag_flags_above_and_within_target(tmp_path):
    """Report should classify days as above/ok/below based on regime bounds."""
    _write_result(
        tmp_path, "2026-08-10", cash=4000, total=10000, regime="normal", trades=0
    )
    _write_result(
        tmp_path, "2026-08-11", cash=2500, total=10000, regime="normal", trades=0
    )
    _write_result(
        tmp_path, "2026-08-12", cash=500, total=10000, regime="normal", trades=0
    )

    text, rows = analyze_cash_drag(tmp_path)
    statuses = {r["date"]: r["status"] for r in rows}
    assert statuses["2026-08-10"] == "above"
    assert statuses["2026-08-11"] == "ok"
    assert statuses["2026-08-12"] == "below"
    assert "Days analyzed: 3" in text


def test_cash_drag_distinguishes_drag_from_cap_binding(tmp_path):
    """Days above target with cap headroom are drag days; days with cap reached are cap-binding."""
    # above target but cap headroom -> drag
    _write_result(
        tmp_path,
        "2026-08-10",
        cash=4000,
        total=10000,
        regime="normal",
        trades=0,
        cap=3,
        trades_this_week=1,
    )
    # above target and cap reached -> cap-binding
    _write_result(
        tmp_path,
        "2026-08-11",
        cash=4000,
        total=10000,
        regime="normal",
        trades=0,
        cap=3,
        trades_this_week=3,
    )

    text, _rows = analyze_cash_drag(tmp_path)
    assert "Cash-drag days (above target with cap headroom): 1" in text
    assert "Cap-binding days (above target but cap reached): 1" in text


def test_cash_drag_uses_regime_specific_bounds(tmp_path):
    """HIGH regime has a higher cash target; a level that is above NORMAL may be ok for HIGH."""
    _write_result(
        tmp_path, "2026-08-10", cash=4000, total=10000, regime="high", trades=0
    )
    text, rows = analyze_cash_drag(tmp_path)
    assert rows[0]["status"] == "ok"
    assert "HIGH 30-50%" in text


def test_cash_drag_invalid_zero_total(tmp_path):
    """A zero total_value makes the cash percentage undefined; row is flagged invalid."""
    _write_result(tmp_path, "2026-08-10", cash=1000, total=0, regime="normal", trades=0)
    text, rows = analyze_cash_drag(tmp_path)
    assert rows[0]["status"] == "invalid"
    assert rows[0]["cash_pct"] is None
    assert "Invalid data   : 1" in text


def test_cash_drag_invalid_nan_total(tmp_path):
    """A non-finite total_value is flagged invalid rather than producing a NaN percentage."""
    _write_result(
        tmp_path, "2026-08-10", cash=1000, total=float("nan"), regime="normal", trades=0
    )
    text, rows = analyze_cash_drag(tmp_path)
    assert rows[0]["status"] == "invalid"
    assert rows[0]["cash_pct"] is None
    assert "Invalid data   : 1" in text


def test_cash_drag_invalid_missing_portfolio(tmp_path):
    """A daily result with no portfolio_after is flagged invalid."""
    result = {
        "date": "2026-08-10",
        "dry_run": False,
        "market_summary": {"assets_analyzed": 32},
        "decision": {"reasoning": "Normal market analysis."},
        "executed_trades": [],
        "cooldown": {
            "status": {
                "current_vol_regime": "normal",
                "config": {"current_vol_regime": "normal"},
            }
        },
    }
    with open(tmp_path / "2026-08-10.json", "w") as f:
        json.dump(result, f)
    text, rows = analyze_cash_drag(tmp_path)
    assert rows[0]["status"] == "invalid"
    assert rows[0]["cash_pct"] is None
    assert "Invalid data   : 1" in text


def test_cash_drag_invalid_negative_total(tmp_path):
    """A negative total_value is meaningless for a cash percentage."""
    _write_result(
        tmp_path, "2026-08-10", cash=1000, total=-5000, regime="normal", trades=0
    )
    _text, rows = analyze_cash_drag(tmp_path)
    assert rows[0]["status"] == "invalid"
    assert rows[0]["cash_pct"] is None


def test_cash_drag_invalid_infinite_cash(tmp_path):
    """An infinite cash value is flagged invalid."""
    _write_result(
        tmp_path,
        "2026-08-10",
        cash=float("inf"),
        total=10000,
        regime="normal",
        trades=0,
    )
    _text, rows = analyze_cash_drag(tmp_path)
    assert rows[0]["status"] == "invalid"
    assert rows[0]["cash_pct"] is None


def test_cash_drag_invalid_rows_skipped_in_diagnosis(tmp_path):
    """Invalid rows are not counted as cash-drag or cap-binding days."""
    _write_result(tmp_path, "2026-08-10", cash=1000, total=0, regime="normal", trades=0)
    _write_result(
        tmp_path,
        "2026-08-11",
        cash=4000,
        total=10000,
        regime="normal",
        trades=0,
        cap=3,
        trades_this_week=1,
    )
    text, _rows = analyze_cash_drag(tmp_path)
    assert "Cash-drag days (above target with cap headroom): 1" in text
    assert "Cap-binding days (above target but cap reached): 0" in text
    assert "Invalid data   : 1" in text
