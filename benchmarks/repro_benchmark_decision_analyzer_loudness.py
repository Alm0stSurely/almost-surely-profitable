"""Repro gate: _get_forward_return must log loudly on exception, stay quiet on expected degenerate paths.

Background (PR #95): decision_analyzer._get_forward_return wrapped its entire
body in try/except Exception and returned np.nan on ANY exception with zero
output. Expected degenerate inputs (missing ticker, empty DataFrame, <2 rows)
are handled by explicit guards upstream and are legitimately quiet. But an
unexpected exception (network failure, parse error, index misalignment) was
swallowed silently — the caller could not distinguish "no data yet" from
"data fetch exploded".

Post-fix contract: one print warning per exception with ticker, date, exception
type and message; expected degenerate paths stay byte-quiet.

Gate protocol (PR #86/#87 lessons): every scenario runs inside try/except,
checks are defensive, pre-fix exit is 1 with the FULL named failure set and
zero tracebacks; healthy pins pass on BOTH sides.
"""

import io
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from analysis.decision_analyzer import DecisionAnalyzer

FAILURES = []
PASSES = []


def _check(name, ok, detail=""):
    if ok:
        PASSES.append(name)
    else:
        FAILURES.append(f"XX {name} — {detail}")


def _run_quietly(fn):
    """Run fn with stdout/stderr captured; return (result, stdout, stderr, error)."""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    try:
        with redirect_stdout(buf_out), redirect_stderr(buf_err):
            result = fn()
        return (result, buf_out.getvalue(), buf_err.getvalue(), None)
    except Exception as e:  # noqa: BLE001 -- gate must record, not crash (PR #87)
        return (None, buf_out.getvalue(), buf_err.getvalue(), f"{type(e).__name__}: {e}")


def main():
    a = DecisionAnalyzer(results_dir=tempfile.mkdtemp())

    # --- Scenario 1: exception path must be loud (the defect) ---
    def s1():
        with patch("analysis.decision_analyzer.fetch_historical_data") as mock_fetch:
            mock_fetch.side_effect = Exception("Network error")
            return a._get_forward_return("SPY", "2026-05-10", 100.0, days=5)

    ret, out, err, exc = _run_quietly(s1)
    _check("s1 exception returns NaN", ret is not None and np.isnan(ret), f"ret={ret} exc={exc}")
    _check("s1 exception logs to stdout", "forward return calculation failed" in out,
           f"stdout={out[:120]!r}")
    _check("s1 exception names ticker", "SPY" in out, f"stdout={out[:120]!r}")
    _check("s1 exception names reason", "Network error" in out, f"stdout={out[:120]!r}")

    # --- Scenario 2: expected degenerate — missing ticker stays quiet ---
    def s2():
        with patch("analysis.decision_analyzer.fetch_historical_data") as mock_fetch:
            mock_fetch.return_value = {}
            return a._get_forward_return("XXX", "2026-05-10", 100.0, days=5)

    ret, out, err, exc = _run_quietly(s2)
    _check("s2 missing ticker returns NaN", ret is not None and np.isnan(ret), f"ret={ret} exc={exc}")
    _check("s2 missing ticker stays quiet", out == "" and err == "",
           f"stdout={out[:120]!r} stderr={err[:120]!r}")

    # --- Scenario 3: expected degenerate — empty DataFrame stays quiet ---
    def s3():
        with patch("analysis.decision_analyzer.fetch_historical_data") as mock_fetch:
            mock_fetch.return_value = {"SPY": pd.DataFrame()}
            return a._get_forward_return("SPY", "2026-05-10", 100.0, days=5)

    ret, out, err, exc = _run_quietly(s3)
    _check("s3 empty df returns NaN", ret is not None and np.isnan(ret), f"ret={ret} exc={exc}")
    _check("s3 empty df stays quiet", out == "" and err == "",
           f"stdout={out[:120]!r} stderr={err[:120]!r}")

    # --- Scenario 4: expected degenerate — <2 rows stays quiet ---
    def s4():
        with patch("analysis.decision_analyzer.fetch_historical_data") as mock_fetch:
            mock_fetch.return_value = {
                "SPY": pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2026-05-10"]))
            }
            return a._get_forward_return("SPY", "2026-05-10", 100.0, days=5)

    ret, out, err, exc = _run_quietly(s4)
    _check("s4 <2 rows returns NaN", ret is not None and np.isnan(ret), f"ret={ret} exc={exc}")
    _check("s4 <2 rows stays quiet", out == "" and err == "",
           f"stdout={out[:120]!r} stderr={err[:120]!r}")

    # --- Scenario 5: healthy path — valid data returns finite, no warning ---
    def s5():
        with patch("analysis.decision_analyzer.fetch_historical_data") as mock_fetch:
            dates = pd.date_range("2026-05-10", periods=10, freq="D")
            df = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates)
            mock_fetch.return_value = {"SPY": df}
            return a._get_forward_return("SPY", "2026-05-10", 100.0, days=5)

    ret, out, err, exc = _run_quietly(s5)
    _check("s5 healthy returns finite", ret is not None and np.isfinite(ret), f"ret={ret} exc={exc}")
    _check("s5 healthy stays quiet", out == "" and err == "",
           f"stdout={out[:120]!r} stderr={err[:120]!r}")

    # --- Report ---
    print(f"\n{'='*60}")
    print(f"REPRO GATE: {len(PASSES)} checks passed, {len(FAILURES)} failed")
    for f in FAILURES:
        print(f)
    if FAILURES:
        print("\nGATE RESULT: FAIL (exit 1)")
        sys.exit(1)
    print("\nGATE RESULT: PASS (exit 0)")


if __name__ == "__main__":
    main()
