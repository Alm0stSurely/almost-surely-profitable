"""Regression guard: LLM prompt risk-enrichment wiring (L0/L1/L2).

Background (2026-11-06 research session): the PORTFOLIO RISK METRICS and
DEFLATED SHARPE RATIO ANALYSIS sections of ``TradingAgent.build_prompt``
never fired in production, for three stacked reasons:

  L0. The availability flags came from RELATIVE imports
      (``from ..risk import ...``) that raise ImportError on the top-level
      import path production uses (``python src/daily_run.py``), leaving
      ``RISK_MODULE_AVAILABLE`` permanently False. The Deflated Sharpe
      import was additionally eager and circular (backtest/__init__ ->
      backtest.backtest -> llm.trading_agent, partially initialized), so
      its flag was False too.
  L1. ``daily_run.py`` never populated
      ``market_data['historical_prices']``, so even with L0 fixed the
      sections had no input.
  L2. The two prompt consumers demanded incompatible shapes for the same
      key (risk metrics: Dict[str, Series]; portfolio returns:
      Dict[str, DataFrame] with 'Close'). Wiring the obvious OHLCV
      producer silently laundered every metric to 0.0 via the finite
      formatter — misinformation worse than absence. A shared
      ``_coerce_close_series`` normalizes both shapes.

The live gate lives at ``scripts/gate_prompt_enrichment.py``; these tests
pin the same invariants inside the suite so a refactor cannot silently
re-break any layer.
"""

import ast
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from llm.trading_agent import (  # noqa: E402
    RISK_MODULE_AVAILABLE,
    TradingAgent,
    _coerce_close_series,
    _load_deflated_sharpe,
)


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


def _agent() -> TradingAgent:
    agent = TradingAgent.__new__(TradingAgent)
    agent.config = {"model": "gate-check"}
    agent.decision_history_dir = Path("/tmp/wiring-test-no-such-dir")
    return agent


# --------------------------------------------------------------------------
# L0: availability flags must be True on the top-level import path that
# production (python src/daily_run.py) and the test suite both use.
# --------------------------------------------------------------------------

def test_risk_module_available_on_top_level_import_path():
    assert RISK_MODULE_AVAILABLE, (
        "relative-only risk import leaves RISK_MODULE_AVAILABLE False on "
        "the top-level path; PORTFOLIO RISK METRICS never renders"
    )


def test_deflated_sharpe_loader_returns_class():
    cls = _load_deflated_sharpe()
    assert cls is not None, (
        "eager/circular DSR import disabled the DEFLATED SHARPE RATIO "
        "ANALYSIS section in production"
    )
    # The lazy loader must return the real class, not a stub.
    from backtest.deflated_sharpe import DeflatedSharpeRatio
    assert cls is DeflatedSharpeRatio


# --------------------------------------------------------------------------
# L2: _coerce_close_series must normalize both producer shapes and refuse
# everything else (never launder garbage into zero metrics).
# --------------------------------------------------------------------------

def test_coerce_close_series_accepts_series():
    s = pd.Series([1.0, 2.0, 3.0])
    out = _coerce_close_series(s)
    assert out is not None
    assert list(out) == [1.0, 2.0, 3.0]


def test_coerce_close_series_accepts_ohlcv_frame():
    df = _ohlcv(1, 100.0)
    out = _coerce_close_series(df)
    assert out is not None
    pd.testing.assert_series_equal(out, df["Close"], check_names=False)


def test_coerce_close_series_drops_non_finite():
    s = pd.Series([1.0, np.nan, np.inf, 4.0])
    out = _coerce_close_series(s)
    assert out is not None
    assert list(out) == [1.0, 4.0]


def test_coerce_close_series_rejects_frame_without_close():
    df = pd.DataFrame({"Open": [1.0, 2.0]})
    assert _coerce_close_series(df) is None


def test_coerce_close_series_rejects_non_price_objects():
    assert _coerce_close_series([1.0, 2.0]) is None
    assert _coerce_close_series({"Close": [1.0]}) is None
    assert _coerce_close_series(None) is None
    assert _coerce_close_series(42.0) is None


def test_coerce_close_series_rejects_empty_after_drop():
    assert _coerce_close_series(pd.Series([np.nan, np.nan])) is None
    assert _coerce_close_series(pd.Series([], dtype=float)) is None


# --------------------------------------------------------------------------
# End-to-end: build_prompt with the production OHLCV producer shape must
# render both enriched sections with real (non-laundered) numbers.
# --------------------------------------------------------------------------

def _market_data(historical_prices=None):
    data = {
        "assets": {
            "SPY": {"latest": {"price": 103.0}, "returns": [0.0] * 69},
            "TLT": {"latest": {"price": 92.0}, "returns": [0.0] * 69},
        },
        "analysis_date": "2026-08-12",
    }
    if historical_prices is not None:
        data["historical_prices"] = historical_prices
    return data


_PORTFOLIO = {
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


def test_build_prompt_renders_enriched_sections_from_ohlcv():
    prices = {"SPY": _ohlcv(7, 100.0), "TLT": _ohlcv(13, 90.0)}
    prompt = _agent().build_prompt(_market_data(prices), _PORTFOLIO, [], None)

    assert "PORTFOLIO RISK METRICS" in prompt
    assert "DEFLATED SHARPE RATIO ANALYSIS" in prompt

    vol_match = re.search(r"[Vv]olatility[^0-9\-]*(-?[\d.]+)", prompt)
    assert vol_match is not None, "no volatility figure rendered"
    assert float(vol_match.group(1)) != 0.0, (
        "risk metrics laundered to zero: OHLCV frames reached a "
        "Dict[str, Series] consumer and MultiIndex columns zeroed "
        "every metric (L2 regression)"
    )


def test_build_prompt_enriched_sections_accept_series_dict():
    prices = {
        "SPY": _ohlcv(7, 100.0)["Close"],
        "TLT": _ohlcv(13, 90.0)["Close"],
    }
    prompt = _agent().build_prompt(_market_data(prices), _PORTFOLIO, [], None)
    assert "PORTFOLIO RISK METRICS" in prompt
    assert "DEFLATED SHARPE RATIO ANALYSIS" in prompt


def test_build_prompt_graceful_without_historical_prices():
    prompt = _agent().build_prompt(_market_data(), _PORTFOLIO, [], None)
    assert "PORTFOLIO RISK METRICS" not in prompt
    assert "DEFLATED SHARPE RATIO ANALYSIS" not in prompt


def test_build_prompt_skips_garbage_price_entries():
    prices = {"SPY": _ohlcv(7, 100.0), "TLT": {"not": "prices"}, "GLD": None}
    prompt = _agent().build_prompt(_market_data(prices), _PORTFOLIO, [], None)
    assert "PORTFOLIO RISK METRICS" in prompt


# --------------------------------------------------------------------------
# L1: daily_run must attach the OHLCV frames to market_analysis so the
# prompt sections have an input. AST guard, mirroring
# test_daily_run_history_path.py (immune to naming refactors).
# --------------------------------------------------------------------------

def test_daily_run_wires_historical_prices():
    source = (REPO_ROOT / "src" / "daily_run.py").read_text()
    tree = ast.parse(source)

    wired = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if not any(isinstance(t, ast.Subscript) for t in node.targets):
            continue
        for target in node.targets:
            if not isinstance(target, ast.Subscript):
                continue
            if not isinstance(target.value, ast.Name) or target.value.id != "market_analysis":
                continue
            sl = target.slice
            if isinstance(sl, ast.Constant) and sl.value == "historical_prices":
                wired = True

    assert wired, (
        "run_daily_pipeline must populate market_analysis['historical_prices']; "
        "without it the prompt enrichment sections have no input (L1)"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
