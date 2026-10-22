#!/usr/bin/env python3
"""Repro gate: evaluation's portfolio-state reader must share the writer's trust model.

Pre-fix contract (the bug): ``evaluation.load_portfolio_data()`` read
``data/portfolio_state.json`` leniently while its writer pair
``Portfolio.save_state`` / ``Portfolio.load_state`` (PR #85) refused the
same malformations. Divergence, concretely:

- a ``NaN`` / ``Infinity`` token (which no strict writer can emit) parsed
  silently; the report printed ``€nan`` / rendered sections from a
  partially corrupt file;
- ``.get("total_value", 0)`` fabricated a confident €0.00 for a missing
  display key, and ``.get("total_value", 10000)`` in the Summary block
  fabricated a +0.00% return on the same fabrication;
- a missing ``total_realized_pnl`` / ``positions`` key crashed the whole
  report with a KeyError far from the cause;
- a ``null`` / list top level was silently indistinguishable from "no
  portfolio" (or crashed on ``.get``).

Post-fix contract: the reader applies the PR #81/#84/#85 refusal
doctrine (strict parse, monetary-base validation, wrong-shape refusal),
the report renders the refusal IN the artifact (PR #87 doctrine), absent
display-only fields render n/a (PR #74 doctrine), and the file is never
mutated by this read-only research reader (churn_analysis doctrine).

Gate exits 1 with named XX failures on pre-fix code, 0 on post-fix code.
Every scenario runs in a sandboxed CWD; a pre-fix crash is recorded as a
named failure, never a traceback that hides later scenarios.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import evaluation  # noqa: E402  (lazy per the PR #84 lesson)


def _run_loader_scenario(name, file_text, expect):
    """Load a state file and check the post-fix contract.

    *expect* is one of:
      - ``("refused", snippet)`` — load_portfolio_data must raise ValueError
        whose message contains *snippet*;
      - ``("parse_error",)`` — json.JSONDecodeError must propagate;
      - ``("valid", cash)`` — the loader returns the dict with *cash*.
    """
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        state_file = data_dir / "portfolio_state.json"
        state_file.write_text(file_text)
        old_cwd = Path.cwd()
        try:
            import os
            os.chdir(tmp)
            try:
                result = evaluation.load_portfolio_data()
            except json.JSONDecodeError as exc:
                if expect[0] != "parse_error":
                    failures.append(
                        f"{name}: expected {expect[0]}, got JSONDecodeError: {exc}"
                    )
            except ValueError as exc:
                if expect[0] == "refused":
                    if expect[1] not in str(exc):
                        failures.append(
                            f"{name}: refusal message missing {expect[1]!r}: {exc}"
                        )
                else:
                    failures.append(
                        f"{name}: expected {expect[0]}, got ValueError: {exc}"
                    )
            else:
                if expect[0] == "refused":
                    failures.append(
                        f"{name}: XX refused state was rendered (returned "
                        f"{result!r:.120})"
                    )
                elif expect[0] == "parse_error":
                    failures.append(
                        f"{name}: XX unparseable file did not raise "
                        f"(returned {result!r:.120})"
                    )
                else:
                    if result.get("cash") != expect[1]:
                        failures.append(
                            f"{name}: expected cash {expect[1]!r}, got "
                            f"{result.get('cash')!r}"
                        )
        finally:
            os.chdir(old_cwd)
    return failures


def _run_report_scenario(name, file_text, expect_in_out, forbid_in_out):
    """Run generate_comprehensive_report on a state file; check the artifact."""
    failures = []
    import io
    import os
    from contextlib import redirect_stdout

    with tempfile.TemporaryDirectory() as tmp:
        data_dir = Path(tmp) / "data"
        data_dir.mkdir()
        (data_dir / "portfolio_state.json").write_text(file_text)
        old_cwd = Path.cwd()
        try:
            os.chdir(tmp)
            buffer = io.StringIO()
            try:
                with redirect_stdout(buffer):
                    evaluation.generate_comprehensive_report()
            except Exception as exc:  # pre-fix code must fail, not crash
                return [f"{name}: report raised {type(exc).__name__}: {exc}"]
        finally:
            os.chdir(old_cwd)
    out = buffer.getvalue()
    for snippet in expect_in_out:
        if snippet not in out:
            failures.append(f"{name}: XX artifact missing {snippet!r}")
    for snippet in forbid_in_out:
        if snippet in out:
            failures.append(f"{name}: XX artifact contains forbidden {snippet!r}")
    return failures


def main():
    failures = []

    loader_cases = [
        ("nan-cash-token",
         '{"cash": NaN, "total_realized_pnl": 0.0, "positions": {}}',
         ("refused", "non-standard JSON constant")),
        ("infinity-pnl-token",
         '{"cash": 100.0, "total_realized_pnl": Infinity, "positions": {}}',
         ("refused", "non-standard JSON constant")),
        ("negative-cash",
         json.dumps({"cash": -100.0, "total_realized_pnl": 0.0, "positions": {}}),
         ("refused", "'cash' is -100.0")),
        ("bool-cash",
         json.dumps({"cash": True, "total_realized_pnl": 0.0, "positions": {}}),
         ("refused", "'cash' is True")),
        ("string-cash",
         json.dumps({"cash": "1000.0", "total_realized_pnl": 0.0, "positions": {}}),
         ("refused", "'cash' is '1000.0'")),
        ("missing-cash",
         json.dumps({"total_realized_pnl": 0.0, "positions": {}}),
         ("refused", "'cash' is None")),
        ("string-pnl",
         json.dumps({"cash": 1000.0, "total_realized_pnl": "n/a", "positions": {}}),
         ("refused", "'total_realized_pnl' is 'n/a'")),
        ("missing-pnl",
         json.dumps({"cash": 1000.0, "positions": {}}),
         ("refused", "'total_realized_pnl' is None")),
        ("null-top-level", "null", ("refused", "top level is 'NoneType'")),
        ("list-top-level", "[1, 2, 3]", ("refused", "top level is 'list'")),
        ("positions-list",
         json.dumps({"cash": 1000.0, "total_realized_pnl": 0.0, "positions": []}),
         ("refused", "'positions' is []")),
        ("missing-positions",
         json.dumps({"cash": 1000.0, "total_realized_pnl": 0.0}),
         ("refused", "'positions' is None")),
        ("garbage-bytes", "not valid json", ("parse_error",)),
        # Both-sides pins: valid state passes; display-only total_value
        # absence must NOT refuse.
        ("healthy", json.dumps({
            "cash": 7500.0, "total_value": 9500.0, "total_realized_pnl": -500.0,
            "positions": {"SPY": {"quantity": 10, "avg_price": 100.0,
                                  "current_price": 95.0}},
        }), ("valid", 7500.0)),
        ("old-format-no-total-value", json.dumps({
            "cash": 1000.0, "total_realized_pnl": 0.0, "positions": {},
        }), ("valid", 1000.0)),
    ]
    for name, text, expect in loader_cases:
        failures.extend(_run_loader_scenario(name, text, expect))

    report_cases = [
        ("report-nan-token",
         '{"cash": NaN, "total_value": 5000.0, "total_realized_pnl": 0.0,'
         ' "positions": {}}',
         ["PORTFOLIO STATUS", "Refusing to render portfolio state",
          "Total Return: n/a", "Evaluation complete"],
         ["nan%", "€nan"]),
        ("report-negative-cash",
         json.dumps({"cash": -5.0, "total_value": 5000.0,
                     "total_realized_pnl": 0.0, "positions": {}}),
         ["PORTFOLIO STATUS", "Refusing to render portfolio state",
          "Total Return: n/a", "Evaluation complete"],
         ["€-5.00"]),
        ("report-missing-display-key",
         json.dumps({"cash": 1000.0, "total_realized_pnl": 0.0,
                     "positions": {}}),
         ["Total Value: n/a", "Cash: €1,000.00 (—)",
          "Realized P&L: €0.00", "Evaluation complete"],
         ["Total Value: €0.00"]),
        ("report-missing-pnl-key",
         json.dumps({"cash": 1000.0, "total_value": 5000.0,
                     "positions": {}}),
         ["PORTFOLIO STATUS", "Refusing to render portfolio state",
          "Evaluation complete"],
         ["€0.00 P&L", "Traceback"]),
        ("report-null-top-level", "null",
         ["PORTFOLIO STATUS", "Refusing to render portfolio state",
          "Evaluation complete"],
         ["Traceback"]),
    ]
    for name, text, expect_in, forbid_in in report_cases:
        failures.extend(_run_report_scenario(name, text, expect_in, forbid_in))

    if failures:
        print(f"GATE FAIL — {len(failures)} named failures:")
        for f in failures:
            print(f"  XX {f}")
        return 1
    print(f"GATE PASS — {len(loader_cases) + len(report_cases)} scenarios, "
          f"all checks green")
    return 0


if __name__ == "__main__":
    sys.exit(main())
