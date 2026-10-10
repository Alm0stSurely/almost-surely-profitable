"""Reproduction gate: bool-accepting numeric guards (bool is a subclass of int).

Two guard sites accepted JSON booleans as valid numbers today:

  1. portfolio._is_valid_positive_scalar  -> True passes: isfinite(True) and True > 0
  2. behavioral_analysis._safe_cash_pct   -> True/True * 100 = 100.0 fabricated

Repo convention (utils._is_finite_number, formatting, trading_agent,
monitor, composite_regime) explicitly rejects bool. These two sites diverge.

Gate: exits 1 pre-fix, 0 post-fix.

Known limitation (follow-up, not gated here): Position.unrealized_pnl_pct
guards the DERIVED cost_basis = quantity * avg_price; multiplication
launders a bool operand into a plain int (10 * True == 10) before the
guard sees it. Operand validation belongs at the state-JSON ingress
(Portfolio.load_state), tracked separately.
"""
import sys
sys.path.insert(0, "src")

from portfolio.portfolio import _is_valid_percentage, _is_valid_positive_scalar
from analysis.behavioral_analysis import _safe_cash_pct

failures = []

# --- Site 1: portfolio guards ---
if _is_valid_positive_scalar(True):
    failures.append("portfolio._is_valid_positive_scalar accepts True")
if _is_valid_percentage(True):
    failures.append("portfolio._is_valid_percentage accepts True (=> 100% of cash order path)")

# --- Site 2: behavioral _safe_cash_pct ---
r = _safe_cash_pct(True, True)
if r is not None:
    failures.append(f"behavioral._safe_cash_pct(True, True) fabricated: {r!r}")
r2 = _safe_cash_pct(True, 10000.0)
if r2 is not None:
    failures.append(f"behavioral._safe_cash_pct(True, 10000.0) fabricated: {r2!r}")
r3 = _safe_cash_pct(5000.0, True)
if r3 is not None:
    failures.append(f"behavioral._safe_cash_pct(5000.0, True) fabricated: {r3!r}")

# --- Healthy paths must be unaffected (bounded pins) ---
healthy = [
    (_is_valid_positive_scalar(50.0), True, "float price"),
    (_is_valid_positive_scalar(1), True, "int quantity"),
    (_is_valid_percentage(25.0), True, "float pct"),
    (_safe_cash_pct(2500.0, 10000.0), 25.0, "cash fraction pct"),
    (_safe_cash_pct(0.0, 10000.0), 0.0, "zero cash"),
    (_safe_cash_pct(None, 10000.0), None, "None cash"),
]
for got, want, label in healthy:
    ok = (got == want) if want is not None else (got is None)
    if not ok:
        failures.append(f"HEALTHY-PIN BROKEN: {label}: got {got!r}, want {want!r}")

if failures:
    print("REPRODUCED — bool accepted as numeric:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("ALL GATES PASS — bool rejected, healthy paths intact")
