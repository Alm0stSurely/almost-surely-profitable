"""Gate check: LLM prompt risk-enrichment liveness and shape-collision laundering.

Defect under gate (three stacked layers, all on the live LLM decision path):
  L0. trading_agent guards the sections with RISK_MODULE_AVAILABLE /
      DSR_MODULE_AVAILABLE, but the flags are computed from RELATIVE
      imports (from ..risk / from ..backtest...) that raise ImportError
      on the top-level import path production and tests both use
      (python src/daily_run.py -> from llm.trading_agent import ...).
      The flags are therefore permanently False and both sections are
      unreachable regardless of data.
  L1. daily_run never populates market_data['historical_prices'], so even
      with L0 fixed the PORTFOLIO RISK METRICS and DEFLATED SHARPE RATIO
      ANALYSIS sections would have no input.
  L2. The two consumers demand incompatible shapes for the same key
      (risk.metrics: Dict[str, Series]; _calculate_portfolio_returns:
      Dict[str, DataFrame] with 'Close'). Wiring the obvious producer
      (the OHLCV dict from fetch_historical_data) silently launders every
      portfolio risk metric to 0.0 via _to_finite, so the prompt would
      show "Volatility: 0.00%, VaR 95%: 0.00%" — misinformation worse
      than absence.

Run:  .venv/bin/python scripts/gate_prompt_enrichment.py
Exit: 0 when all checks pass (post-fix), 1 with named failures (pre-fix).
"""

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llm.trading_agent as ta
from llm.trading_agent import TradingAgent  # noqa: E402


def _ohlcv(seed: int, anchor: float, periods: int = 70) -> pd.DataFrame:
    """Deterministic stochastic OHLCV frame (seeded GBM-like walk)."""
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, size=periods)
    close = anchor * np.exp(np.cumsum(rets))
    dates = pd.date_range("2026-06-01", periods=periods, freq="B")
    return pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": [1000.0] * periods,
        },
        index=dates,
    )


def main() -> int:
    failures = []

    # L0: enrichment flags must be True on the top-level import path that
    # production (python src/daily_run.py) and the test suite both use.
    if not ta.RISK_MODULE_AVAILABLE:
        failures.append("L0: RISK_MODULE_AVAILABLE is False on the top-level llm.trading_agent import path")
    if ta._load_deflated_sharpe() is None:
        failures.append("L0: DeflatedSharpeRatio unavailable on the top-level llm.trading_agent import path")

    agent = TradingAgent.__new__(TradingAgent)  # build_prompt only needs self
    agent.config = {"model": "gate-check"}
    agent.decision_history_dir = Path("/tmp/gate-no-such-dir")

    historical_prices = {"SPY": _ohlcv(7, 100.0), "TLT": _ohlcv(13, 90.0)}

    # Gate 1: production wiring liveness — simulate what daily_run must provide.
    market_data = {
        "assets": {
            "SPY": {"latest": {"price": 103.0}, "returns": [0.0] * 69},
            "TLT": {"latest": {"price": 92.0}, "returns": [0.0] * 69},
        },
        "analysis_date": "2026-08-12",
        # The fix wires this in run_daily_pipeline; pre-fix no producer exists.
        "historical_prices": historical_prices,
    }
    portfolio_summary = {
        "cash": 8000.0,
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "positions": [
            {
                "ticker": "SPY",
                "quantity": 20.0,
                "avg_price": 100.0,
                "current_price": 103.0,
                "market_value": 2000.0,
                "unrealized_pnl_pct": 3.0,
            }
        ],
    }

    prompt = agent.build_prompt(market_data, portfolio_summary, [], None)

    if "PORTFOLIO RISK METRICS" not in prompt:
        failures.append("G1: risk-metrics prompt section absent even with historical_prices wired")

    if "DEFLATED SHARPE RATIO ANALYSIS" not in prompt:
        failures.append("G2: DSR prompt section absent even with historical_prices wired")

    # Gate 3: laundering — trending synthetic data must produce non-zero,
    # finite volatility in the rendered risk section.
    vol_match = re.search(r"[Vv]olatility[^0-9\-]*(-?[\d.]+)", prompt)
    if "PORTFOLIO RISK METRICS" in prompt:
        if vol_match is None:
            failures.append("G3: no volatility figure rendered in risk-metrics section")
        else:
            vol = float(vol_match.group(1))
            if vol == 0.0:
                failures.append(
                    "G3: risk metrics laundered to zero (OHLCV frames hit "
                    "Dict[str, Series] consumer; MultiIndex columns silently "
                    "zero every metric)"
                )

    if failures:
        print(f"GATE FAIL ({len(failures)} named failures):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GATE PASS: prompt risk enrichment live, no laundering")
    return 0


if __name__ == "__main__":
    sys.exit(main())
