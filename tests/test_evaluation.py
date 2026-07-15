"""Tests for evaluation.py trade-count consistency."""

import sys
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import evaluation


def test_trade_counts_use_analyzed_outcomes(capsys):
    """Ensure trade counts match the analyzable trades from DecisionAnalyzer."""
    fake_decisions = [
        {
            "date": f"2026-07-{i:02d}",
            "actions": [],
            "trades": [],
            "reasoning": "",
            "portfolio_before": {},
            "portfolio_after": {},
        }
        for i in range(1, 11)
    ]
    fake_outcomes = {
        "buy_count": 1,
        "sell_count": 0,
        "win_rate": 0.0,
        "buy_accuracy": 0.0,
        "sell_accuracy": 0.0,
    }

    with mock.patch("evaluation.load_portfolio_data", return_value=None), \
         mock.patch("evaluation.load_recent_results", return_value=[]), \
         mock.patch("data.fetch_market_data.fetch_current_prices", return_value={"SPY": 750.0}), \
         mock.patch("evaluation.DecisionAnalyzer") as MockAnalyzer:
        instance = MockAnalyzer.return_value
        instance.load_decisions.return_value = fake_decisions
        instance.analyze_outcomes.return_value = fake_outcomes

        evaluation.generate_comprehensive_report()

        captured = capsys.readouterr()
        assert "Total Trades: 1" in captured.out
        assert "Avg Trades/Day: 0.1" in captured.out
        assert "Win Rate: 0.0%" in captured.out
