"""Pre-fix reproduction: swallowed read-fallback failures are silent.

Sites:
1. utils.load_valid_daily_results - corrupt file silently excluded
2. reporting._get_benchmark_return - fetch failure silent -> None
3. evaluation._get_benchmark_return - fetch failure silent -> None
4. monitor.check_bollinger_breakouts - per-ticker calc crash silently skipped
"""
import json
import logging
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, "src")

# Capture ALL log output to prove silence
records = []
handler = logging.Handler()
handler.emit = records.append
root = logging.getLogger()
root.addHandler(handler)
root.setLevel(logging.DEBUG)


def loud_print(msg):
    print(f"  [captured] {msg}")


class CapturePrint:
    def __enter__(self):
        import io
        self.buf = io.StringIO()
        self._old = sys.stdout
        sys.stdout = self.buf
        return self.buf

    def __exit__(self, *a):
        sys.stdout = self._old


# --- Site 1: utils.load_valid_daily_results ---
print("=== Site 1: utils.load_valid_daily_results (corrupt file) ===")
from utils import load_valid_daily_results

tmp = Path(tempfile.mkdtemp())
(tmp / "2026-10-01.json").write_text("{ corrupt json !!")
(tmp / "2026-10-02.json").write_text(json.dumps({
    "date": "2026-10-02", "dry_run": False,
    "assets_analyzed": 10,
    "decision": {"reasoning": "normal run"},
    "portfolio_after": {"total_value": 10000, "cash": 5000, "num_positions": 3},
}))
with CapturePrint() as buf:
    res = load_valid_daily_results(str(tmp))
log_msgs = [r.getMessage() for r in records]
print(f"loaded={len(res)} (expected 1 valid sibling)")
print(f"log records mentioning corrupt: {[m for m in log_msgs if 'corrupt' in m.lower() or '2026-10-01' in m]}")
print(f"stdout: {buf.getvalue()!r}")
assert len(res) == 1, "valid sibling must survive"
assert [m for m in log_msgs if '2026-10-01' in m], 'NO LOG: corrupt file still silently excluded'
print('>>> CONFIRMED LOUD')

# --- Site 2: reporting._get_benchmark_return ---
print("\n=== Site 2: reporting._get_benchmark_return (fetch raises) ===")
import reporting
import yfinance as yf

orig_download = yf.download
def boom(*a, **k):
    raise ConnectionError("network down")
yf.download = boom
try:
    out = reporting.ReportGenerator()._get_benchmark_return("SPY", "2026-01-01", "2026-02-01")
finally:
    yf.download = orig_download
log_msgs = [r.getMessage() for r in records]
print(f"returned={out}")
print(f"log records mentioning benchmark/SPY/fail: {[m for m in log_msgs if 'SPY' in m or 'benchmark' in m.lower()][:5]}")
assert out is None
assert [m for m in log_msgs if 'Could not fetch benchmark SPY' in m], "NO LOG: benchmark failure still silent"
print(">>> CONFIRMED LOUD")

# --- Site 3: evaluation._get_benchmark_return ---
print("\n=== Site 3: evaluation._get_benchmark_return (fetch raises) ===")
import evaluation

evaluation.fetch_historical_data = boom
try:
    with CapturePrint() as buf2:
        out3 = evaluation._get_benchmark_return("2026-01-01", "2026-02-01")
finally:
    evaluation.fetch_historical_data = None  # restore below
import data.fetch_market_data as fmm
evaluation.fetch_historical_data = fmm.fetch_historical_data
print(f"returned={out3}")
print(f"stdout: {buf2.getvalue()!r}")
assert out3 is None
assert "Could not fetch benchmark SPY" in buf2.getvalue(), "NO LOG: evaluation benchmark failure still silent"
print(">>> CONFIRMED LOUD")

# --- Site 4: monitor.check_bollinger_breakouts (calc crashes per ticker) ---
print("\n=== Site 4: monitor.check_bollinger_breakouts (calc raises) ===")
import monitor
from portfolio.portfolio import Portfolio

monitor.CHECK_BOLLINGER = True
monitor.load_alert_history = lambda: {}
monitor.save_alert_history = lambda h: None
orig_fhd2 = fmm.fetch_historical_data
fmm.fetch_historical_data = boom

pf = Portfolio()
pf.positions["SPY"] = {"quantity": 1, "avg_price": 100}
with CapturePrint() as buf3:
    alerts4 = monitor.check_bollinger_breakouts({"SPY": 150.0}, pf)
fmm.fetch_historical_data = orig_fhd2
print(f"alerts={alerts4}")
print(f"stdout: {buf3.getvalue()!r}")
assert "Bollinger check failed for SPY" in buf3.getvalue(), "NO LOG: monitor calc crash still silent"
print(">>> CONFIRMED LOUD")

print("\nALL SITES LOUD (functional gate passes on fixed code)")
