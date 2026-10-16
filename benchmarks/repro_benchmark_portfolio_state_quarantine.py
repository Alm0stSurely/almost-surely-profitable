"""Repro + benchmark for the portfolio state quarantine ingress (PR #85).

Repro gate (the functional proof):
    1. A corrupt ``portfolio_state.json`` — garbage bytes, a NaN/Infinity
       token (data no repo writer can emit: ``save_state`` is strict
       ``allow_nan=False``), or a valid-JSON non-mapping top level — was
       handled pre-fix by ``except Exception`` -> ``print`` -> fabricate
       INITIAL_CAPITAL, with the corrupt file LEFT IN PLACE. The next
       ``save_state()`` then silently overwrote the evidence: the PR #60
       class (corrupt state destroyed by the system's own write path).
       Worse, a second ``except Exception`` wrapped the positions
       restore and reset *just-validated real cash* to the €10,000
       default on any unexpected error — the widest possible blast
       radius for an unspecified failure.
    2. Post-fix, the shared quarantine helper renames the file aside as
       ``portfolio_state.json.corrupt-<timestamp>`` (bytes preserved),
       logs one loud error, and the constructor takes the honest
       no-claim fresh-start path. Parseable monetary lies still REFUSE
       (ValueError) — the layering is: quarantine owns unparseable,
       refusal owns parseable lies, per-element drop owns malformed
       positions.

    Exit code 0 iff every post-fix expectation holds. Pre-fix the gate
    fails (corrupt file found in place after the load, no quarantine
    rename, fabricated state).

Benchmark (honest cold-path claim): the guarded load runs once per
session on a cold path; a timing delta is not the point. We still time
the helper-based load vs a plain ``json.load`` over a healthy state
file so the PR can state observed numbers instead of asserting "no
regression" from assumption.
"""

import json
import logging
import statistics
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio

logging.basicConfig(level=logging.CRITICAL)  # keep the benchmark table readable

STATE_NAME = "portfolio_state.json"

HEALTHY = {
    "cash": 2295.17,
    "total_realized_pnl": -112.50,
    "positions": {
        ticker: {"quantity": 5.0, "avg_price": 400.0, "current_price": 410.0}
        for ticker in ("SPY", "GLD", "QQQ", "TLT", "MC.PA")
    },
}


def _write(tmp: Path, text: str) -> Path:
    path = tmp / STATE_NAME
    path.write_text(text)
    return path


def _token_state(key: str, token: str) -> str:
    state = dict(HEALTHY)
    state[key] = "__TOKEN_PLACEHOLDER__"
    return json.dumps(state, indent=2).replace('"__TOKEN_PLACEHOLDER__"', token)


def check(name: str, ok: bool, failures: list) -> None:
    print(f"  {'✓' if ok else '✗'} {name}")
    if not ok:
        failures.append(name)


def run_repro() -> int:
    failures: list[str] = []
    print("Repro: portfolio state quarantine ingress")

    # Case 1 — garbage bytes: quarantine rename, bytes preserved, fresh state.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        original = b"not json {{{"
        _write(tmp, original.decode())
        p = Portfolio(data_dir=str(tmp))
        q = list(tmp.glob(f"{STATE_NAME}.corrupt-*"))
        check("garbage bytes: corrupt file renamed aside", not (tmp / STATE_NAME).exists(), failures)
        check("garbage bytes: exactly one quarantine file", len(q) == 1, failures)
        check("garbage bytes: original bytes preserved", len(q) == 1 and q[0].read_bytes() == original, failures)
        check("garbage bytes: fresh honest state", p.cash == Portfolio.INITIAL_CAPITAL and p.positions == {}, failures)

    # Case 2 — NaN token (data no repo writer can emit): whole-file quarantine.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        text = _token_state("cash", "NaN")
        path = _write(tmp, text)
        original = path.read_bytes()
        p = Portfolio(data_dir=str(tmp))
        q = list(tmp.glob(f"{STATE_NAME}.corrupt-*"))
        check("NaN token: corrupt file renamed aside", not path.exists(), failures)
        check("NaN token: bytes preserved", len(q) == 1 and q[0].read_bytes() == original, failures)
        check("NaN token: fresh honest state", p.cash == Portfolio.INITIAL_CAPITAL, failures)

    # Case 3 — valid-JSON non-mapping top level (a list): no claim, quarantine.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        original = b"[1, 2, 3]"
        _write(tmp, original.decode())
        p = Portfolio(data_dir=str(tmp))
        q = list(tmp.glob(f"{STATE_NAME}.corrupt-*"))
        check("list top level: corrupt file renamed aside", not (tmp / STATE_NAME).exists(), failures)
        check("list top level: bytes preserved", len(q) == 1 and q[0].read_bytes() == original, failures)
        check("list top level: fresh honest state", p.cash == Portfolio.INITIAL_CAPITAL, failures)

    # Case 4 — layering: a parseable monetary lie still REFUSES, file
    # untouched, no quarantine. Quarantine must not swallow the refusal class.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        state = dict(HEALTHY, cash="2295.17")
        path = _write(tmp, json.dumps(state, indent=2, allow_nan=False))
        original = path.read_bytes()
        refused = False
        try:
            Portfolio(data_dir=str(tmp))
        except ValueError:
            refused = True
        check("string cash: parseable lie still refuses (ValueError)", refused, failures)
        check("string cash: file byte-untouched, no quarantine", path.read_bytes() == original and not list(tmp.glob("*.corrupt-*")), failures)

    # Case 5 — the overwrite class is closed: after quarantine, save_state
    # writes a clean file and it reloads normally (single quarantine total).
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write(tmp, _token_state("total_realized_pnl", "Infinity"))
        p = Portfolio(data_dir=str(tmp))
        p.save_state()
        reloaded = Portfolio(data_dir=str(tmp))
        q = list(tmp.glob(f"{STATE_NAME}.corrupt-*"))
        check("post-quarantine save/reload cycle clean", reloaded.cash == Portfolio.INITIAL_CAPITAL, failures)
        check("exactly one quarantine across the cycle", len(q) == 1, failures)

    # Case 6 — healthy path untouched: no quarantine, exact state, no error.
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        _write(tmp, json.dumps(HEALTHY, indent=2, allow_nan=False))
        p = Portfolio(data_dir=str(tmp))
        check(
            "healthy: exact load, no quarantine",
            p.cash == 2295.17 and len(p.positions) == 5 and not list(tmp.glob("*.corrupt-*")),
            failures,
        )

    print(f"Repro gate: {6 - len(failures)}/6 groups clean" if not failures else f"Repro gate FAILED: {failures}")
    return 1 if failures else 0


def _bench_guarded(path: Path, n: int) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with _suppress_output():  # the module prints its load summary
            Portfolio(data_dir=str(path.parent))
        times.append((time.perf_counter() - t0) * 1e3)
    return times


class _NullIO:
    def write(self, _):
        pass

    def flush(self):
        pass


class _suppress_output:
    """Discard stdout+stderr so the module's load summary cannot pollute
    the benchmark table or the timing numbers."""

    def __enter__(self):
        self._old = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = _NullIO()
        return self

    def __exit__(self, *args):
        sys.stdout, sys.stderr = self._old


def _bench_plain(path: Path, n: int) -> list[float]:
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        with open(path, "r") as f:
            json.load(f)
        times.append((time.perf_counter() - t0) * 1e3)
    return times


def run_benchmark() -> None:
    print("\nBenchmark: guarded load vs plain json.load (healthy 5-position state)")
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        path = _write(tmp, json.dumps(HEALTHY, indent=2, allow_nan=False))
        n, rounds = 500, 5
        guarded = [statistics.mean(_bench_guarded(path, n)) for _ in range(rounds)]
        plain = [statistics.mean(_bench_plain(path, n)) for _ in range(rounds)]
        best_g, best_p = min(guarded), min(plain)
        print(f"  guarded load (Portfolio() incl. guards): best-of-{rounds} {best_g:.4f} ms/load")
        print(f"  plain json.load (parse only):            best-of-{rounds} {best_p:.4f} ms/load")
        print(f"  delta: {best_g - best_p:+.4f} ms — helper + ingress guard overhead on a cold path")
        print("  (constructor/validation overhead dominates; the hook itself is one parse_constant call)")


if __name__ == "__main__":
    rc = run_repro()
    run_benchmark()
    sys.exit(rc)
