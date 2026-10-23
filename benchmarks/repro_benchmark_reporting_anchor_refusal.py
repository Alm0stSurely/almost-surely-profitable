#!/usr/bin/env python3
"""Repro gate: periodic reports must refuse anchor fabrication and survive
null-shaped portfolio blocks (PR #92).

Pre-fix contract (the bug family): ``reporting.py`` — the twin of
``weekly_report.py`` — anchored period metrics with fabricated values and
crashed on null-shaped input that the ingress validator admits:

- ``_safe_start_value`` fabricated a €10,000 start anchor when the first
  day's ``total_value`` was missing/invalid, then measured the weekly /
  monthly return against the fiction;
- ``.get('total_value', 0)`` for the end anchor fabricated a -100% period
  on a missing key (0 is finite; the refusal asymmetry start-vs-end made
  the fabrication invisible);
- an explicit ``portfolio_after: null`` (admitted by the ingress validator,
  which checks dry-run / reasoning / asset count, not this field) crashed
  the whole report with AttributeError — blast radius: the entire period,
  malformation: one field;
- ``positions: null`` crashed position aggregation with TypeError;
- a single daily return produced ``volatility: 0.00%`` — a confident
  fabricated number for an undefined sample std.

Post-fix contract: anchors refuse to None (rendered n/a downstream, the
weekly_report.py ``_safe_positive_scalar(..., default=None)`` convention),
explicit-null shapes render as empty/absent (report survives, valid
siblings intact — Lipschitz), and undefined volatility is None, not 0.

Gate exits 1 with named XX failures on pre-fix code, 0 on post-fix code.
A pre-fix crash is recorded as a named failure, never a traceback that
hides later scenarios.
"""

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import reporting  # noqa: E402  (lazy per the PR #84 lesson)


def _day(date, value, positions=None):
    return {
        "date": date,
        "portfolio_after": {
            "total_value": value,
            "total_return_pct": (value / 10000.0 - 1) * 100,
            "positions": positions if positions is not None else [],
        },
        "executed_trades": [],
    }


def _write_week(tmp, days):
    tmp = Path(tmp)
    for date, payload in days.items():
        (tmp / f"{date}.json").write_text(json.dumps(payload))


def _check(label, failures, cond):
    if cond:
        print(f"  ok  {label}")
    else:
        failures.append(label)
        print(f"  XX  {label}")


def scenario_missing_start_anchor():
    print("scenario: missing start total_value → refusal, no fabrication")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        d0 = _day("2026-02-02", 10000.0)
        del d0["portfolio_after"]["total_value"]
        _write_week(tmp, {"2026-02-02": d0, "2026-02-03": _day("2026-02-03", 10100.0)})
        try:
            report = reporting.ReportGenerator(results_dir=tmp).generate_weekly_report(2026, 6)
            _check("start_value is None (not fabricated 10000)", failures,
                   report.get("start_value") is None)
            _check("weekly_return_pct is None (not measured vs fiction)", failures,
                   report.get("weekly_return_pct") is None)
            _check("end anchor from valid day survives", failures,
                   report.get("end_value") == 10100.0)
        except Exception as exc:  # noqa: BLE001 - gate records, never propagates
            failures.append(f"weekly report crashed ({type(exc).__name__})")
            print(f"  XX  weekly report crashed: {type(exc).__name__}")
    return failures


def scenario_missing_end_anchor():
    print("scenario: missing end total_value → refusal, not fabricated -100%")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        d1 = _day("2026-02-03", 10100.0)
        del d1["portfolio_after"]["total_value"]
        _write_week(tmp, {"2026-02-02": _day("2026-02-02", 10000.0), "2026-02-03": d1})
        try:
            report = reporting.ReportGenerator(results_dir=tmp).generate_weekly_report(2026, 6)
            _check("end_value is None (not fabricated 0)", failures,
                   report.get("end_value") is None)
            _check("weekly_return_pct is None (not -100%)", failures,
                   report.get("weekly_return_pct") is None)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"weekly report crashed ({type(exc).__name__})")
            print(f"  XX  weekly report crashed: {type(exc).__name__}")
    return failures


def scenario_null_portfolio_after():
    print("scenario: explicit portfolio_after null → report survives")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        d0 = _day("2026-02-02", 10000.0)
        d0["portfolio_after"] = None
        _write_week(tmp, {"2026-02-02": d0, "2026-02-03": _day("2026-02-03", 10100.0)})
        try:
            report = reporting.ReportGenerator(results_dir=tmp).generate_weekly_report(2026, 6)
            _check("start_value is None", failures, report.get("start_value") is None)
            _check("weekly_return_pct is None", failures,
                   report.get("weekly_return_pct") is None)
            _check("valid day still anchors the end", failures,
                   report.get("end_value") == 10100.0)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"weekly report crashed ({type(exc).__name__})")
            print(f"  XX  weekly report crashed: {type(exc).__name__}")
    return failures


def scenario_null_positions():
    print("scenario: positions null → empty set, no TypeError")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        d1 = _day("2026-02-03", 10100.0)
        d1["portfolio_after"]["positions"] = None
        _write_week(tmp, {"2026-02-02": _day("2026-02-02", 10000.0), "2026-02-03": d1})
        try:
            rg = reporting.ReportGenerator(results_dir=tmp)
            rg._get_benchmark_return = lambda *_a, **_k: None  # offline gate
            weekly = rg.generate_weekly_report(2026, 6)
            _check("weekly final_positions == []", failures,
                   weekly.get("final_positions") == [])
            monthly = rg.generate_monthly_report(2026, 2)
            _check("monthly final_positions_count == 0", failures,
                   monthly.get("final_positions_count") == 0)
            _check("monthly best/worst positions are None", failures,
                   monthly.get("best_position") is None
                   and monthly.get("worst_position") is None)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"report crashed ({type(exc).__name__})")
            print(f"  XX  report crashed: {type(exc).__name__}")
    return failures


def scenario_garbage_position_elements():
    print("scenario: garbage position elements → valid siblings survive")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        d1 = _day("2026-02-03", 10100.0, positions=[
            {"ticker": "SPY", "quantity": 10, "market_value": 6060.0,
             "unrealized_pnl_pct": 5.0},
            "garbage",
            42,
        ])
        _write_week(tmp, {"2026-02-02": _day("2026-02-02", 10000.0), "2026-02-03": d1})
        try:
            report = reporting.ReportGenerator(results_dir=tmp).generate_weekly_report(2026, 6)
            tickers = [p.get("ticker") for p in report.get("final_positions", [])]
            _check("only the valid position survives", failures, tickers == ["SPY"])
        except Exception as exc:  # noqa: BLE001
            failures.append(f"weekly report crashed ({type(exc).__name__})")
            print(f"  XX  weekly report crashed: {type(exc).__name__}")
    return failures


def scenario_single_observation_volatility():
    print("scenario: single daily return → volatility undefined (None), not 0")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        _write_week(tmp, {"2026-02-02": _day("2026-02-02", 10000.0),
                          "2026-02-03": _day("2026-02-03", 10100.0)})
        try:
            report = reporting.ReportGenerator(results_dir=tmp).generate_weekly_report(2026, 6)
            _check("volatility is None (not fabricated 0)", failures,
                   report.get("volatility") is None)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"weekly report crashed ({type(exc).__name__})")
            print(f"  XX  weekly report crashed: {type(exc).__name__}")
    return failures


def scenario_healthy_week():
    print("scenario: healthy two-day week → real anchors (both-sides pin)")
    failures = []
    with tempfile.TemporaryDirectory() as tmp:
        _write_week(tmp, {"2026-02-02": _day("2026-02-02", 10000.0),
                          "2026-02-03": _day("2026-02-03", 10100.0)})
        try:
            report = reporting.ReportGenerator(results_dir=tmp).generate_weekly_report(2026, 6)
            _check("start_value 10000", failures, report.get("start_value") == 10000.0)
            _check("end_value 10100", failures, report.get("end_value") == 10100.0)
            ret = report.get("weekly_return_pct")
            _check("weekly_return_pct ≈ 1.0", failures,
                   ret is not None and abs(ret - 1.0) < 0.01)
        except Exception as exc:  # noqa: BLE001
            failures.append(f"weekly report crashed ({type(exc).__name__})")
            print(f"  XX  weekly report crashed: {type(exc).__name__}")
    return failures


SCENARIOS = [
    scenario_missing_start_anchor,
    scenario_missing_end_anchor,
    scenario_null_portfolio_after,
    scenario_null_positions,
    scenario_garbage_position_elements,
    scenario_single_observation_volatility,
    scenario_healthy_week,
]


def main():
    all_failures = []
    for scenario in SCENARIOS:
        all_failures.extend(scenario())
        print()
    if all_failures:
        print(f"GATE FAILED — {len(all_failures)} named failure(s):")
        for failure in all_failures:
            print(f"  XX {failure}")
        return 1
    print("GATE PASSED — all scenarios honor the anchor-refusal contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
