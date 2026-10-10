"""Regression tests for bool rejection in numeric guards.

Contract: ``bool`` is a subclass of ``int``, so a naive
``isinstance(value, (int, float))`` guard accepts JSON booleans as numbers.
The repo convention (``utils._is_finite_number``, ``utils.formatting``,
``trading_agent``, ``monitor``, ``composite_regime``) explicitly rejects
bool. Two guard sites diverged from that convention:

- ``portfolio._is_valid_positive_scalar`` — order sizing (buy/sell pct),
  price updates, and the ``Position.unrealized_pnl_pct`` fallback. A
  ``true`` reaching it is read as the number 1 (e.g. a fabricated 100% of
  cash order, or a fabricated P&L percentage on a bool cost basis).
- ``behavioral_analysis._safe_cash_pct`` — cash-ratio column of the
  behavioral report from daily result JSON files. ``(True, True)`` used to
  fabricate a 100.0% cash reading instead of ``n/a``.

Discriminators assert the bool is rejected; bounded healthy-path pins
assert real numbers (int and float), zero, and None still behave exactly
as before on both sides of the fix.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.behavioral_analysis import _safe_cash_pct
from portfolio.portfolio import _is_valid_percentage, _is_valid_positive_scalar


class TestPositiveScalarBoolRejection:
    """portfolio._is_valid_positive_scalar must reject booleans."""

    def test_true_rejected(self):
        assert _is_valid_positive_scalar(True) is False

    def test_false_rejected(self):
        # False was already rejected (False > 0 is False) — pin it stays so.
        assert _is_valid_positive_scalar(False) is False

    def test_valid_percentage_rejects_true(self):
        # _is_valid_percentage builds on the positive-scalar guard; True
        # used to read as 1.0, i.e. a "valid" 1% order instead of refused.
        assert _is_valid_percentage(True) is False

    # Note: Position.unrealized_pnl_pct is NOT pinned here on purpose.
    # cost_basis = quantity * avg_price launders a bool operand into a
    # plain int before this guard ever sees it (10 * True == 10), so the
    # guard on the derived value cannot detect operand corruption — the
    # PR #77 lesson: validate operands at ingress, not derived values.
    # The bool-avg_price ingress is Portfolio.load_state (state JSON),
    # tracked as a follow-up; this PR covers the guard contract itself.


class TestPositiveScalarHealthyPaths:
    """Bounded pins: real numbers must keep passing on both sides."""

    def test_float_accepted(self):
        assert _is_valid_positive_scalar(50.0) is True

    def test_int_accepted(self):
        assert _is_valid_positive_scalar(1) is True

    def test_zero_rejected(self):
        assert _is_valid_positive_scalar(0.0) is False

    def test_negative_rejected(self):
        assert _is_valid_positive_scalar(-5.0) is False

    def test_none_rejected(self):
        assert _is_valid_positive_scalar(None) is False

    def test_string_rejected(self):
        assert _is_valid_positive_scalar("50") is False

    def test_nan_rejected(self):
        assert _is_valid_positive_scalar(float("nan")) is False

    def test_valid_percentage_float_accepted(self):
        assert _is_valid_percentage(25.0) is True

    def test_valid_percentage_boundary_100_accepted(self):
        assert _is_valid_percentage(100) is True

    def test_valid_percentage_over_100_rejected(self):
        assert _is_valid_percentage(100.1) is False


class TestSafeCashPctBoolRejection:
    """behavioral_analysis._safe_cash_pct must reject booleans."""

    def test_true_true_rejected(self):
        # Used to fabricate 100.0 (all-cash) from two JSON booleans.
        assert _safe_cash_pct(True, True) is None

    def test_true_cash_rejected(self):
        # Used to fabricate 0.01 (wrong scale, nonzero) from cash=true.
        assert _safe_cash_pct(True, 10000.0) is None

    def test_true_total_rejected(self):
        assert _safe_cash_pct(5000.0, True) is None

    def test_false_total_rejected(self):
        # Bounded pin: False was already refused via the total <= 0 check;
        # it must stay refused through the explicit bool rejection too.
        assert _safe_cash_pct(5000.0, False) is None


class TestSafeCashPctHealthyPaths:
    """Bounded pins: real ratios must keep passing on both sides."""

    def test_normal_ratio(self):
        assert _safe_cash_pct(2500.0, 10000.0) == 25.0

    def test_zero_cash(self):
        assert _safe_cash_pct(0.0, 10000.0) == 0.0

    def test_none_cash(self):
        assert _safe_cash_pct(None, 10000.0) is None

    def test_none_total(self):
        assert _safe_cash_pct(2500.0, None) is None

    def test_zero_total(self):
        assert _safe_cash_pct(2500.0, 0.0) is None

    def test_negative_total(self):
        assert _safe_cash_pct(2500.0, -100.0) is None

    def test_nan_cash(self):
        assert _safe_cash_pct(float("nan"), 10000.0) is None
