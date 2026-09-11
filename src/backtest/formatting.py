"""Backtest console-formatting helpers.

Canonical implementations live in ``utils.formatting`` (shared, stdlib +
numpy only). This module re-exports them so existing imports — including
PR #45's regression tests and downstream benchmark imports — keep working
unchanged.
"""

from utils.formatting import _fmt_finite, _fmt_pct  # noqa: F401  (re-exported)
