"""Micro-benchmark for TradingAgent prompt and risk/metrics LLM-summary guards."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd

from llm.trading_agent import TradingAgent, _safe_format
from risk.metrics import RiskMetrics
from risk.metrics import _safe_format as _risk_safe_format


def _make_agent():
    return TradingAgent(api_key="test", history_file="/tmp/benchmark_trading_agent_decisions.json")


def _make_market_data():
    return {
        "assets": {
            "SPY": {
                "latest": {
                    "price": 400.0,
                    "rsi_14": 45.0,
                    "bb_position": 0.5,
                    "sma_20": 395.0,
                    "sma_50": 390.0,
                    "volatility_annual": 0.15,
                    "drawdown": -0.02,
                    "daily_return": 0.005,
                }
            },
            "TLT": {
                "latest": {
                    "price": 100.0,
                    "rsi_14": 55.0,
                    "bb_position": 0.6,
                    "sma_20": 99.0,
                    "sma_50": 98.0,
                    "volatility_annual": 0.10,
                    "drawdown": -0.01,
                    "daily_return": 0.002,
                }
            },
        },
        "correlations": pd.DataFrame(),
        "regime": None,
    }


def _make_portfolio():
    return {
        "cash": 8000.0,
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "total_pnl": 0.0,
        "positions": [
            {
                "ticker": "SPY",
                "quantity": 5,
                "avg_price": 390.0,
                "current_price": 400.0,
                "unrealized_pnl_pct": 2.56,
                "market_value": 2000.0,
            }
        ],
        "risk_metrics": {
            "cvar_95": -0.02,
            "var_95": -0.015,
            "max_drawdown": -0.05,
            "sortino_ratio": 1.2,
            "skewness": -0.1,
            "kurtosis": 3.0,
        },
    }


def _make_cooldown_status():
    return {
        "trades_this_week": 1,
        "weekly_cap": 2,
        "active_entries": {"SPY": {"entry_date": "2026-06-16T21:00:00", "hold_days": 2.0}},
        "recent_exits": {"GLD": {"exit_date": "2026-06-18T16:00:00", "days_since_exit": 0.5}},
        "config": {"min_hold_days": 5, "flip_cooldown_days": 10},
    }


def _make_non_finite_market_data():
    data = _make_market_data()
    data["assets"]["SPY"]["latest"]["price"] = float("nan")
    data["assets"]["SPY"]["latest"]["rsi_14"] = float("inf")
    data["assets"]["TLT"]["latest"]["volatility_annual"] = float("-inf")
    data["correlations"] = {"SPY/TLT": float("nan")}
    return data


def _make_non_finite_portfolio():
    portfolio = _make_portfolio()
    portfolio["cash"] = float("nan")
    portfolio["total_value"] = float("inf")
    portfolio["total_return_pct"] = float("-inf")
    portfolio["total_pnl"] = float("nan")
    portfolio["positions"][0]["quantity"] = float("nan")
    portfolio["positions"][0]["avg_price"] = float("inf")
    portfolio["positions"][0]["current_price"] = float("-inf")
    portfolio["positions"][0]["unrealized_pnl_pct"] = float("nan")
    return portfolio


def _make_non_finite_cooldown_status():
    status = _make_cooldown_status()
    status["trades_this_week"] = float("nan")
    status["weekly_cap"] = float("inf")
    status["active_entries"]["SPY"]["hold_days"] = float("nan")
    status["recent_exits"]["GLD"]["days_since_exit"] = float("inf")
    return status


def _run(func):
    start = time.perf_counter()
    result = func()
    elapsed = (time.perf_counter() - start) * 1e6
    return result, elapsed


def main():
    agent = _make_agent()
    market_data = _make_market_data()
    portfolio = _make_portfolio()
    cooldown_status = _make_cooldown_status()

    _, us_prompt_full = _run(
        lambda: agent.build_prompt(market_data, portfolio, cooldown_status=cooldown_status)
    )
    _, us_prompt_nonfinite = _run(
        lambda: agent.build_prompt(
            _make_non_finite_market_data(),
            _make_non_finite_portfolio(),
            cooldown_status=_make_non_finite_cooldown_status(),
        )
    )
    _, us_safe_format_finite = _run(lambda: _safe_format(1.2345, ".2f"))
    _, us_safe_format_nan = _run(lambda: _safe_format(float("nan"), ".2f"))

    metrics = RiskMetrics(
        var_95=-0.02,
        var_99=-0.05,
        cvar_95=-0.03,
        cvar_99=-0.06,
        volatility=0.20,
        downside_volatility=0.15,
        max_drawdown=-0.10,
        current_drawdown=-0.05,
        sortino_ratio=1.5,
        calmar_ratio=2.0,
        skewness=-0.5,
        kurtosis=3.0,
    )
    _, us_risk_summary = _run(lambda: _risk_safe_format(metrics.var_95, ".2%"))
    _, us_risk_summary_nan = _run(lambda: _risk_safe_format(float("nan"), ".2%"))

    print("| Helper | Input | µs/call |")
    print("|---|---|---|")
    print(f"| `TradingAgent.build_prompt` (full) | finite inputs | {us_prompt_full:.3f} |")
    print(f"| `TradingAgent.build_prompt` (full) | non-finite inputs | {us_prompt_nonfinite:.3f} |")
    print(f"| `_safe_format` (trading_agent) | finite | {us_safe_format_finite:.3f} |")
    print(f"| `_safe_format` (trading_agent) | NaN | {us_safe_format_nan:.3f} |")
    print(f"| `_safe_format` (risk/metrics) | finite | {us_risk_summary:.3f} |")
    print(f"| `_safe_format` (risk/metrics) | NaN | {us_risk_summary_nan:.3f} |")


if __name__ == "__main__":
    main()
