"""Functional gate for the Portfolio.load_state ingress guard.

Pre-fix, Portfolio.load_state passed raw JSON values straight into Position:
  * bool fields were laundered into arithmetic (bool is an int subclass:
    quantity=True * price == price — a fabricated market value);
  * NaN/Infinity tokens (json.load accepts them by default) loaded silently
    and crashed the NEXT save_state (allow_nan=False) or poisoned total_value;
  * string fields loaded fine and crashed later, at first property access
    (TypeError mid-pipeline, far from the cause);
  * one malformed entry gave no signal at all.

Post-fix, every malformed position is dropped loudly (print, per-entry) and
healthy siblings survive (per-element drop, PR #64 doctrine).

Exits 0 post-fix, 1 pre-fix. Run from the repo root:
    PYTHONPATH=src .venv/bin/python benchmarks/repro_portfolio_load_state_guard.py
"""

import io
import json
import math
import sys
import tempfile
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from portfolio.portfolio import Portfolio


def _write_state(tmpdir: str, positions: dict) -> None:
    """Write a state file by hand so non-standard tokens (NaN) can appear."""
    state = {
        "cash": 5000.0,
        "total_realized_pnl": 0.0,
        "positions": positions,
        "last_updated": "2026-10-11T00:00:00",
        "total_value": 5000.0,
    }
    # json.dump would refuse NaN (allow_nan=False everywhere in the repo), so
    # the file is assembled manually — only the gate needs the raw token.
    text = json.dumps(state, indent=2)
    for pos in positions.values():
        if not isinstance(pos, dict):
            continue
        for value in pos.values():
            if isinstance(value, float) and math.isnan(value):
                text = text.replace(json.dumps(value), "NaN", 1)
    Path(tmpdir, "portfolio_state.json").write_text(text)


def _load(tmpdir: str) -> tuple:
    buf = io.StringIO()
    with redirect_stdout(buf):
        portfolio = Portfolio(data_dir=tmpdir)
    return portfolio, buf.getvalue()


def gate_bool_laundering() -> bool:
    """bool quantity must not become a fabricated market value."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_state(tmpdir, {"SPY": {"quantity": True, "avg_price": 400.0, "current_price": 410.0}})
        portfolio, log = _load(tmpdir)
        if "SPY" in portfolio.positions:
            print(f"  [FAIL] bool quantity accepted: market_value={portfolio.positions['SPY'].market_value}")
            return False
        if "SPY" not in log:
            print("  [FAIL] bool quantity dropped without a loud log")
            return False
    print("  [ok] bool quantity rejected loudly")
    return True


def gate_bool_avg_price() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": True, "current_price": 410.0}})
        portfolio, log = _load(tmpdir)
        if "SPY" in portfolio.positions:
            print(f"  [FAIL] bool avg_price accepted: cost_basis={portfolio.positions['SPY'].cost_basis}")
            return False
        if "avg_price" not in log:
            print("  [FAIL] bool avg_price dropped without naming the field")
            return False
    print("  [ok] bool avg_price rejected loudly")
    return True


def gate_nan_quantity() -> bool:
    """NaN token loads pre-fix and kills the next save_state (allow_nan=False)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_state(tmpdir, {"SPY": {"quantity": float("nan"), "avg_price": 400.0, "current_price": 410.0}})
        portfolio, log = _load(tmpdir)
        if "SPY" in portfolio.positions:
            print(f"  [FAIL] NaN quantity accepted: total_value={portfolio.total_value}")
            return False
        if "quantity" not in log:
            print("  [FAIL] NaN quantity dropped without naming the field")
            return False
        # The downstream kill must be gone: a healthy save must succeed.
        portfolio.save_state()
    print("  [ok] NaN quantity rejected loudly, save_state survives")
    return True


def gate_string_avg_price() -> bool:
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_state(tmpdir, {"SPY": {"quantity": 10.0, "avg_price": "100", "current_price": 410.0}})
        portfolio, _ = _load(tmpdir)
        if "SPY" in portfolio.positions:
            try:
                _ = portfolio.positions["SPY"].unrealized_pnl
            except TypeError:
                print("  [ok] string avg_price loads, then crashes at first P&L access (pre-fix behavior)")
                return False
            print("  [FAIL] string avg_price accepted and silently usable")
            return False
    print("  [ok] string avg_price rejected loudly")
    return True


def gate_sibling_survival() -> bool:
    """One malformed position must not void the healthy sibling (PR #64)."""
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_state(tmpdir, {
            "SPY": {"quantity": True, "avg_price": 400.0, "current_price": 410.0},
            "GLD": {"quantity": 5.0, "avg_price": 200.0, "current_price": 210.0},
        })
        portfolio, _ = _load(tmpdir)
        if "GLD" not in portfolio.positions:
            print("  [FAIL] healthy sibling GLD lost alongside malformed SPY")
            return False
        if abs(portfolio.positions["GLD"].market_value - 5.0 * 210.0) > 1e-9:
            print("  [FAIL] healthy sibling loaded with wrong values")
            return False
        if "SPY" in portfolio.positions:
            print("  [FAIL] malformed SPY kept alongside healthy GLD")
            return False
    print("  [ok] per-element drop: healthy sibling survives, malformed entry loud")
    return True


def gate_shape() -> bool:
    """A non-dict position entry and a non-dict positions map must not crash.

    Blast-radius pin: the pre-fix outer ``except`` resets cash to
    INITIAL_CAPITAL (global hold-all); the per-element guard must preserve
    the persisted cash and drop only the malformed positions.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        _write_state(tmpdir, {"SPY": [10.0, 400.0]})
        portfolio, _ = _load(tmpdir)
        if portfolio.positions:
            print("  [FAIL] list-shaped position entry loaded as a position")
            return False
        if abs(portfolio.cash - 5000.0) > 1e-9:
            print(f"  [FAIL] cash reset to {portfolio.cash} (blast radius exceeded the malformed entry)")
            return False
    print("  [ok] non-dict position entry rejected loudly, persisted cash preserved")
    return True


GATES = [
    gate_bool_laundering,
    gate_bool_avg_price,
    gate_nan_quantity,
    gate_string_avg_price,
    gate_sibling_survival,
    gate_shape,
]


def main() -> int:
    failed = [g.__name__ for g in GATES if not g()]
    if failed:
        print(f"\nGATE FAILED ({len(failed)}/{len(GATES)}): {', '.join(failed)}")
        return 1
    print(f"\nAll {len(GATES)} gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
