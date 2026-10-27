#!/usr/bin/env python3
"""Repro gate: load_ledger_realized_pnl strict-reader quarantine.

Pre-fix: bare json.load ingests NaN tokens silently, crashes on garbage
bytes, crashes on wrong-shaped top level.
Post-fix: shared quarantine ingress makes every failure loud (ERROR log,
bytes preserved under .corrupt-*) and returns NaN.

Exit 0 post-fix, exit 1 pre-fix with named failures.
"""

import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.churn_analysis import load_ledger_realized_pnl  # noqa: E402


def scenario(name, content, expect_nan=True, expect_quarantine=True):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "portfolio_state.json"
        path.write_text(content)
        try:
            result = load_ledger_realized_pnl(td)
            error = None
        except Exception as exc:
            result = None
            error = f"{type(exc).__name__}: {exc}"

        checks = []
        if expect_nan:
            if error is not None:
                checks.append(f"CRASH: {error}")
            elif not math.isnan(result):
                checks.append(f"expected NaN, got {result!r}")

        if expect_quarantine:
            if path.exists():
                checks.append("file not quarantined (still in place)")
            corrupt = list(Path(td).glob("portfolio_state.json.corrupt-*"))
            if len(corrupt) == 0:
                checks.append("no .corrupt-* file found")

        return checks


def main():
    failures = []

    # 1. NaN token — strict writer cannot emit, lenient reader ingests silently
    checks = scenario("nan_token", '{"total_realized_pnl": NaN}')
    for c in checks:
        failures.append(f"nan_token: {c}")

    # 2. Infinity token
    checks = scenario("infinity_token", '{"total_realized_pnl": Infinity}')
    for c in checks:
        failures.append(f"infinity_token: {c}")

    # 3. Garbage bytes
    checks = scenario("garbage_bytes", "not json at all {{{")
    for c in checks:
        failures.append(f"garbage_bytes: {c}")

    # 4. Wrong shape: list instead of dict
    checks = scenario("wrong_shape_list", json.dumps([1, 2, 3]))
    for c in checks:
        failures.append(f"wrong_shape_list: {c}")

    # 5. Wrong shape: scalar
    checks = scenario("wrong_shape_scalar", "42")
    for c in checks:
        failures.append(f"wrong_shape_scalar: {c}")

    # 6. Wrong shape: null
    checks = scenario("wrong_shape_null", "null")
    for c in checks:
        failures.append(f"wrong_shape_null: {c}")

    # 7. Healthy path: valid dict — must NOT quarantine, must return value
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "portfolio_state.json"
        path.write_text(json.dumps({"total_realized_pnl": 123.45, "cash": 5000.0}))
        result = load_ledger_realized_pnl(td)
        if result != 123.45:
            failures.append(f"healthy_path: expected 123.45, got {result!r}")
        if not path.exists():
            failures.append("healthy_path: file was quarantined (should be untouched)")
        corrupt = list(Path(td).glob("portfolio_state.json.corrupt-*"))
        if corrupt:
            failures.append("healthy_path: spurious .corrupt-* file")

    # 8. Missing key — must NOT quarantine (valid dict, absent field)
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "portfolio_state.json"
        path.write_text(json.dumps({"cash": 1000.0}))
        result = load_ledger_realized_pnl(td)
        if not math.isnan(result):
            failures.append(f"missing_key: expected NaN, got {result!r}")
        if not path.exists():
            failures.append("missing_key: file was quarantined (should be untouched)")

    if failures:
        print(f"XX {len(failures)} FAILURE(S):")
        for f in failures:
            print(f"  XX {f}")
        sys.exit(1)

    print("OK all scenarios passed")
    sys.exit(0)


if __name__ == "__main__":
    main()
