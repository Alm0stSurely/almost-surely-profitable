"""Functional gate for the load_state monetary-base guard (PR #81).

Exits 0 on fixed code, 1 on pre-fix code. For every malformed-cash and
malformed-realized-pnl class the pre-fix loader either silently fabricates
(NaN/bool/absent), silently blocks all orders (bool cash -> €1 base), or
crashes into the ghost-reset except (string cash). The gate asserts the
post-fix contract — ValueError out of Portfolio(), file untouched — and
the healthy contract (legitimate states load exactly).

Run: .venv/bin/python benchmarks/repro_portfolio_load_state_monetary_guard.py
"""

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio

HEALTHY = {
    "cash": 2295.17,
    "total_realized_pnl": -112.50,
    "positions": {
        "SPY": {"quantity": 10.0, "avg_price": 400.0, "current_price": 410.0}
    },
}


def write_state(dirpath: Path, state: dict) -> Path:
    data_dir = dirpath / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    state_file = data_dir / "portfolio_state.json"
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, allow_nan=False)
    return state_file


def write_state_with_nan(dirpath: Path, nan_key: str) -> Path:
    data_dir = dirpath / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    state_file = data_dir / "portfolio_state.json"
    state = dict(HEALTHY)
    state[nan_key] = "__NAN_PLACEHOLDER__"
    text = json.dumps(state, indent=2).replace('"__NAN_PLACEHOLDER__"', "NaN")
    state_file.write_text(text)
    return state_file


def expect_refusal(name: str, dirpath: Path) -> bool:
    try:
        Portfolio(data_dir=str(dirpath / "data"))
    except ValueError as e:
        print(f"  PASS {name}: refused ({e})")
        return True
    except Exception as e:  # noqa: BLE001 — any non-ValueError is a failure
        print(f"  FAIL {name}: raised {type(e).__name__}, expected ValueError")
        return False
    print(f"  FAIL {name}: loaded silently — fabrication or ghost reset")
    return False


def main() -> int:
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="repro-monetary-guard-"))
    try:
        cases = []
        d = tmp / "nan_cash"
        write_state_with_nan(d, "cash")
        cases.append(("NaN cash", d))
        d = tmp / "bool_cash"
        write_state(d, dict(HEALTHY, cash=True))
        cases.append(("bool cash", d))
        d = tmp / "string_cash"
        write_state(d, dict(HEALTHY, cash="2295.17"))
        cases.append(("string cash", d))
        d = tmp / "negative_cash"
        write_state(d, dict(HEALTHY, cash=-500.0))
        cases.append(("negative cash", d))
        d = tmp / "absent_cash"
        write_state(d, {k: v for k, v in HEALTHY.items() if k != "cash"})
        cases.append(("absent cash", d))
        d = tmp / "nan_pnl"
        write_state_with_nan(d, "total_realized_pnl")
        cases.append(("NaN realized_pnl", d))
        d = tmp / "bool_pnl"
        write_state(d, dict(HEALTHY, total_realized_pnl=True))
        cases.append(("bool realized_pnl", d))
        d = tmp / "string_pnl"
        write_state(d, dict(HEALTHY, total_realized_pnl="-112.50"))
        cases.append(("string realized_pnl", d))
        d = tmp / "absent_pnl"
        write_state(d, {k: v for k, v in HEALTHY.items() if k != "total_realized_pnl"})
        cases.append(("absent realized_pnl", d))

        print("Refusal classes (9):")
        for name, d in cases:
            results.append(expect_refusal(name, d))

        # File-untouched pin on one representative case
        d = tmp / "untouched"
        sf = write_state_with_nan(d, "cash")
        before = sf.read_bytes()
        try:
            Portfolio(data_dir=str(d / "data"))
        except ValueError:
            pass
        ok = sf.read_bytes() == before
        print(f"  {'PASS' if ok else 'FAIL'} refusal leaves file byte-untouched")
        results.append(ok)

        # Healthy contract pins (these pass on both sides)
        print("Healthy contract (5):")
        healthy_cases = [
            ("healthy exact", HEALTHY, lambda p: p.cash == 2295.17 and p.total_realized_pnl == -112.50),
            ("zero cash (all-in)", dict(HEALTHY, cash=0.0), lambda p: p.cash == 0.0),
            ("zero realized_pnl", dict(HEALTHY, total_realized_pnl=0.0), lambda p: p.total_realized_pnl == 0.0),
            ("negative realized_pnl", dict(HEALTHY, total_realized_pnl=-987.65), lambda p: p.total_realized_pnl == -987.65),
            ("int cash", dict(HEALTHY, cash=5000), lambda p: p.cash == 5000),
        ]
        for name, state, check in healthy_cases:
            d = tmp / f"healthy_{len(results)}"
            write_state(d, state)
            try:
                p = Portfolio(data_dir=str(d / "data"))
                ok = check(p)
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL {name}: raised {type(e).__name__}: {e}")
                ok = False
            print(f"  {'PASS' if ok else 'FAIL'} {name}")
            results.append(ok)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    passed = sum(results)
    print(f"\nGate: {passed}/{len(results)} {'GREEN' if passed == len(results) else 'RED'}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
