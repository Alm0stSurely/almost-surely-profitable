"""Tests for behavioral analysis module."""

import sys
import json
import copy
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from analysis import behavioral_analysis as ba


@pytest.fixture
def sample_decisions():
    """Two valid decisions with known keyword coverage."""
    return [
        {
            "timestamp": "2026-07-01T21:00:00",
            "actions": [
                {"ticker": "SPY", "action": "hold"},
                {"ticker": "TTE.PA", "action": "hold"},
            ],
            "reasoning": "The weekly trade cap of 3/3 has been reached. "
                         "Loss aversion and cash buffer guide the hold. "
                         "Tail risk is elevated; we respect the trade limit.",
            "error": False,
        },
        {
            "timestamp": "2026-07-02T21:00:00",
            "actions": [
                {"ticker": "SPY", "action": "hold"},
                {"ticker": "QQQ", "action": "buy", "pct": 10},
            ],
            "reasoning": "Cash buffer is high; mean reversion on QQQ. "
                         "Regime is normal. We use stop-loss discipline.",
            "error": False,
        },
    ]


@pytest.fixture
def sample_trades():
    return []


def run_analysis_with_data(decisions, trades, tmp_path):
    """Helper: run main() with temporary data files."""
    data_dir = tmp_path / "data"
    results_dir = tmp_path / "results" / "daily"
    results_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    with open(data_dir / "decision_history.json", "w") as f:
        json.dump(decisions, f)
    with open(data_dir / "trades_history.json", "w") as f:
        json.dump(trades, f)

    # Create a minimal daily result so churn section has a start date
    with open(results_dir / "2026-07-01.json", "w") as f:
        json.dump({
            "date": "2026-07-01",
            "portfolio_after": {
                "cash": 5000.0,
                "total_value": 10000.0,
                "num_positions": 2,
                "total_return_pct": 0.0,
            },
            "executed_trades": [],
        }, f)

    with patch.object(ba, "DATA_DIR", data_dir), \
         patch.object(ba, "RESULTS_DIR", results_dir), \
         patch.object(ba, "OUTPUT_DIR", tmp_path / "results" / "analysis"):
        ba.main()

    return tmp_path / "results" / "analysis" / f"behavioral_analysis_{datetime.now().strftime('%Y%m%d')}.txt"


def test_action_distribution_percentages(tmp_path, sample_decisions, sample_trades):
    """Action percentages should be relative to total actions, not decisions."""
    report_path = run_analysis_with_data(sample_decisions, sample_trades, tmp_path)
    report = report_path.read_text()

    # Total actions: 2 + 2 = 4; holds = 3, buys = 1
    assert "Total actions: 4 (2.0 per decision)" in report
    assert "hold  :    3 ( 75.0% of actions)" in report
    assert "buy   :    1 ( 25.0% of actions)" in report


def test_trade_cap_keyword_matches_variants(tmp_path, sample_decisions, sample_trades):
    """The 'trade cap' concept should match 'weekly trade cap' and 'trade limit'."""
    # Make sure the second decision also contains a variant so both count.
    decisions = copy.deepcopy(sample_decisions)
    decisions[1]["reasoning"] += " The weekly trade cap resets Monday."
    report_path = run_analysis_with_data(decisions, sample_trades, tmp_path)
    report = report_path.read_text()

    lines = [line for line in report.splitlines() if "trade cap" in line]
    assert len(lines) == 1
    # 2 out of 2 valid decisions mention the concept
    assert "trade cap" in lines[0]
    assert ":    2" in lines[0]


def test_keyword_counts_are_case_insensitive(tmp_path, sample_decisions, sample_trades):
    """Keyword matching should be case-insensitive."""
    # Upper-case reasoning to test case-insensitivity
    decisions = [{
        "timestamp": "2026-07-01T21:00:00",
        "actions": [{"ticker": "SPY", "action": "hold"}],
        "reasoning": "CVaR AND LOSS AVERSION ARE KEY.",
        "error": False,
    }]
    report_path = run_analysis_with_data(decisions, sample_trades, tmp_path)
    report = report_path.read_text()

    assert "CVaR" in report
    assert "loss aversion" in report
    # Extract the count lines for each concept
    cvar_line = [line for line in report.splitlines() if "CVaR" in line][0]
    loss_line = [line for line in report.splitlines() if "loss aversion" in line][0]
    assert ":    1" in cvar_line
    assert ":    1" in loss_line


def test_error_rate_evolution(tmp_path, sample_decisions, sample_trades):
    """Error rate should be reported by month."""
    decisions_with_error = sample_decisions + [
        {
            "timestamp": "2026-07-03T21:00:00",
            "actions": [],
            "reasoning": "API error",
            "error": True,
        }
    ]
    report_path = run_analysis_with_data(decisions_with_error, sample_trades, tmp_path)
    report = report_path.read_text()

    assert "2026-07:   1/  3 errors" in report
