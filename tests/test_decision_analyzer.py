"""
Comprehensive tests for the decision analysis module.

Covers:
- Decision loading from JSON results
- Forward return calculation (mocked market data)
- Outcome analysis (buy/sell accuracy, win rate)
- Behavioral pattern detection (loss aversion, overconfidence, diversification)
- Report generation
- Edge cases: empty data, missing files, timezone handling, zero prices
"""

import sys
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from analysis.decision_analyzer import DecisionAnalyzer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def analyzer():
    return DecisionAnalyzer(results_dir="results/daily")


@pytest.fixture
def sample_decisions():
    """Two decision records with trades."""
    return [
        {
            "date": "2026-05-10",
            "timestamp": "2026-05-10T21:00:00Z",
            "actions": [{"ticker": "AI.PA", "action": "buy", "pct": 10}],
            "trades": [{"ticker": "AI.PA", "action": "buy", "price": 176.70, "shares": 5.0}],
            "reasoning": "Mean reversion signal on Air Liquide",
            "portfolio_before": {"cash": 10000.0},
            "portfolio_after": {"cash": 9116.5},
        },
        {
            "date": "2026-05-11",
            "timestamp": "2026-05-11T21:00:00Z",
            "actions": [{"ticker": "SAN.PA", "action": "buy", "pct": 8}],
            "trades": [{"ticker": "SAN.PA", "action": "buy", "price": 73.05, "shares": 19.0}],
            "reasoning": "RSI at 30, Bollinger at 0.24",
            "portfolio_before": {"cash": 9116.5},
            "portfolio_after": {"cash": 7728.55},
        },
    ]


@pytest.fixture
def mock_price_data():
    """Synthetic price DataFrame for forward return mocks."""
    dates = pd.date_range("2026-05-10", periods=10, freq="D")
    return pd.DataFrame({
        "Open": np.linspace(176, 185, 10),
        "High": np.linspace(177, 186, 10),
        "Low": np.linspace(175, 184, 10),
        "Close": np.linspace(176, 185, 10),
        "Volume": np.full(10, 1000000),
    }, index=dates)


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

def test_analyzer_default_init():
    """Default results_dir should be 'results/daily'."""
    a = DecisionAnalyzer()
    assert a.results_dir == Path("results/daily")
    assert a.analysis_cache == {}


def test_analyzer_custom_init():
    """Custom results_dir should be respected."""
    a = DecisionAnalyzer(results_dir="/tmp/my_results")
    assert a.results_dir == Path("/tmp/my_results")


# ---------------------------------------------------------------------------
# load_decisions
# ---------------------------------------------------------------------------

def test_load_decisions_missing_dir(analyzer, capsys):
    """Missing results dir should warn and return empty list."""
    analyzer.results_dir = Path("/nonexistent/path")
    decisions = analyzer.load_decisions(days=30)
    assert decisions == []
    captured = capsys.readouterr()
    assert "not found" in captured.out


def test_load_decisions_empty_dir(analyzer, tmp_path):
    """Empty results dir should return empty list."""
    analyzer.results_dir = tmp_path
    decisions = analyzer.load_decisions(days=30)
    assert decisions == []


def test_load_decisions_valid_files(analyzer, tmp_path, sample_decisions):
    """Valid JSON files with decisions and trades should be loaded."""
    analyzer.results_dir = tmp_path
    for i, dec in enumerate(sample_decisions):
        file_path = tmp_path / f"2026-05-{10+i}.json"
        with open(file_path, "w") as f:
            json.dump({
                "date": dec["date"],
                "timestamp": dec["timestamp"],
                "decision": {"actions": dec["actions"], "reasoning": dec["reasoning"]},
                "executed_trades": dec["trades"],
                "portfolio_before": dec["portfolio_before"],
                "portfolio_after": dec["portfolio_after"],
            }, f)

    decisions = analyzer.load_decisions(days=30)
    assert len(decisions) == 2
    assert decisions[0]["date"] == "2026-05-10"
    assert decisions[1]["date"] == "2026-05-11"


def test_load_decisions_skips_no_trades(analyzer, tmp_path):
    """Files without executed_trades should be skipped."""
    analyzer.results_dir = tmp_path
    with open(tmp_path / "2026-05-10.json", "w") as f:
        json.dump({
            "date": "2026-05-10",
            "decision": {"actions": [], "reasoning": "Hold everything"},
            "executed_trades": [],
        }, f)

    decisions = analyzer.load_decisions(days=30)
    assert decisions == []


def test_load_decisions_skips_missing_keys(analyzer, tmp_path):
    """Files missing 'decision' or 'executed_trades' should be skipped."""
    analyzer.results_dir = tmp_path
    with open(tmp_path / "2026-05-10.json", "w") as f:
        json.dump({"date": "2026-05-10", "portfolio": {}}, f)

    decisions = analyzer.load_decisions(days=30)
    assert decisions == []


def test_load_decisions_invalid_json(analyzer, tmp_path, capsys):
    """Invalid JSON files should be warned and skipped."""
    analyzer.results_dir = tmp_path
    with open(tmp_path / "2026-05-10.json", "w") as f:
        f.write("not valid json")

    decisions = analyzer.load_decisions(days=30)
    assert decisions == []
    captured = capsys.readouterr()
    assert "Could not load" in captured.out


def test_load_decisions_limits_days(analyzer, tmp_path):
    """Should only load last N files."""
    analyzer.results_dir = tmp_path
    for day in range(1, 11):
        with open(tmp_path / f"2026-05-{day:02d}.json", "w") as f:
            json.dump({
                "date": f"2026-05-{day:02d}",
                "decision": {"actions": [], "reasoning": ""},
                "executed_trades": [{"ticker": "SPY", "action": "buy", "price": 100.0}],
            }, f)

    decisions = analyzer.load_decisions(days=5)
    assert len(decisions) == 5
    assert decisions[0]["date"] == "2026-05-06"
    assert decisions[-1]["date"] == "2026-05-10"


# ---------------------------------------------------------------------------
# _get_forward_return
# ---------------------------------------------------------------------------

@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_basic(mock_fetch, analyzer, mock_price_data):
    """Basic forward return calculation."""
    mock_fetch.return_value = {"AI.PA": mock_price_data}
    ret = analyzer._get_forward_return("AI.PA", "2026-05-10", 176.0, days=5)
    # Close goes from 176 to ~181 over 5 days (linear interpolation)
    expected = (181.0 - 176.0) / 176.0
    assert pytest.approx(ret, rel=0.05) == expected
    mock_fetch.assert_called_once()


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_sell_scenario(mock_fetch, analyzer, mock_price_data):
    """Forward return when price drops — relevant for sell decisions."""
    declining = mock_price_data.copy()
    declining["Close"] = np.linspace(185, 176, 10)
    mock_fetch.return_value = {"SAN.PA": declining}
    ret = analyzer._get_forward_return("SAN.PA", "2026-05-10", 185.0, days=5)
    assert ret < 0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_missing_ticker(mock_fetch, analyzer):
    """Missing ticker in fetched data should return 0.0."""
    mock_fetch.return_value = {}
    ret = analyzer._get_forward_return("XXX", "2026-05-10", 100.0, days=5)
    assert ret == 0.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_empty_dataframe(mock_fetch, analyzer):
    """Empty DataFrame should return 0.0."""
    mock_fetch.return_value = {"SPY": pd.DataFrame()}
    ret = analyzer._get_forward_return("SPY", "2026-05-10", 100.0, days=5)
    assert ret == 0.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_not_enough_data(mock_fetch, analyzer):
    """Less than 2 rows of future data should return 0.0."""
    mock_fetch.return_value = {
        "SPY": pd.DataFrame({"Close": [100]}, index=pd.to_datetime(["2026-05-10"]))
    }
    ret = analyzer._get_forward_return("SPY", "2026-05-10", 100.0, days=5)
    assert ret == 0.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_timezone_aware(mock_fetch, analyzer):
    """Timezone-aware index should be handled gracefully."""
    dates = pd.date_range("2026-05-10", periods=10, freq="D", tz="UTC")
    df = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates)
    mock_fetch.return_value = {"SPY": df}
    ret = analyzer._get_forward_return("SPY", "2026-05-10", 100.0, days=5)
    assert ret > 0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_exception_handling(mock_fetch, analyzer):
    """Exceptions in fetch should return 0.0 silently."""
    mock_fetch.side_effect = Exception("Network error")
    ret = analyzer._get_forward_return("SPY", "2026-05-10", 100.0, days=5)
    assert ret == 0.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_forward_return_zero_price(mock_fetch, analyzer, mock_price_data):
    """Zero entry price should be skipped at the analyze_outcomes level,
    but _get_forward_return itself doesn't check for zero."""
    mock_fetch.return_value = {"SPY": mock_price_data}
    ret = analyzer._get_forward_return("SPY", "2026-05-10", 0.0, days=5)
    # Division by zero would happen if we calculated return, but the
    # method uses entry_day_price from fetched data, not the passed price
    assert isinstance(ret, float)


# ---------------------------------------------------------------------------
# analyze_outcomes
# ---------------------------------------------------------------------------

@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_buy_success(mock_fetch, analyzer, mock_price_data):
    """Buy decision is successful when price goes up."""
    mock_fetch.return_value = {"AI.PA": mock_price_data}
    decisions = [{
        "date": "2026-05-10",
        "trades": [{"ticker": "AI.PA", "action": "buy", "price": 176.0}],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["buy_count"] == 1
    assert metrics["buy_accuracy"] == 1.0  # Price went up
    assert metrics["win_rate"] == 1.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_buy_failure(mock_fetch, analyzer):
    """Buy decision fails when price goes down."""
    dates = pd.date_range("2026-05-10", periods=10, freq="D")
    declining = pd.DataFrame({"Close": np.linspace(180, 170, 10)}, index=dates)
    mock_fetch.return_value = {"AI.PA": declining}
    decisions = [{
        "date": "2026-05-10",
        "trades": [{"ticker": "AI.PA", "action": "buy", "price": 180.0}],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["buy_count"] == 1
    assert metrics["buy_accuracy"] == 0.0
    assert metrics["win_rate"] == 0.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_sell_success(mock_fetch, analyzer):
    """Sell decision is successful when price goes down (we avoided loss)."""
    dates = pd.date_range("2026-05-10", periods=10, freq="D")
    declining = pd.DataFrame({"Close": np.linspace(180, 170, 10)}, index=dates)
    mock_fetch.return_value = {"AI.PA": declining}
    decisions = [{
        "date": "2026-05-10",
        "trades": [{"ticker": "AI.PA", "action": "sell", "price": 180.0}],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["sell_count"] == 1
    assert metrics["sell_accuracy"] == 1.0  # Price went down
    assert metrics["win_rate"] == 1.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_sell_failure(mock_fetch, analyzer, mock_price_data):
    """Sell decision fails when price goes up (we missed gains)."""
    mock_fetch.return_value = {"AI.PA": mock_price_data}
    decisions = [{
        "date": "2026-05-10",
        "trades": [{"ticker": "AI.PA", "action": "sell", "price": 176.0}],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["sell_count"] == 1
    assert metrics["sell_accuracy"] == 0.0
    assert metrics["win_rate"] == 0.0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_mixed(mock_fetch, analyzer):
    """Mixed buy/sell with different outcomes."""
    dates = pd.date_range("2026-05-10", periods=10, freq="D")
    spy_up = pd.DataFrame({"Close": np.linspace(100, 110, 10)}, index=dates)
    spy_down = pd.DataFrame({"Close": np.linspace(100, 90, 10)}, index=dates)

    def side_effect(tickers, period):
        return {"SPY": spy_up, "TLT": spy_down}

    mock_fetch.side_effect = side_effect
    decisions = [{
        "date": "2026-05-10",
        "trades": [
            {"ticker": "SPY", "action": "buy", "price": 100.0},
            {"ticker": "TLT", "action": "sell", "price": 100.0},
        ],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["buy_count"] == 1
    assert metrics["sell_count"] == 1
    assert metrics["buy_accuracy"] == 1.0
    assert metrics["sell_accuracy"] == 1.0  # TLT went down
    assert metrics["win_rate"] == 1.0


def test_analyze_outcomes_empty_decisions(analyzer):
    """Empty decisions should return empty metrics."""
    metrics = analyzer.analyze_outcomes([], forward_days=5)
    assert metrics["total_decisions"] == 0
    assert metrics["win_rate"] == 0.0


def test_analyze_outcomes_zero_price_skipped(analyzer):
    """Trades with zero price should be skipped."""
    decisions = [{
        "date": "2026-05-10",
        "trades": [{"ticker": "SPY", "action": "buy", "price": 0.0}],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["total_decisions"] == 0


# ---------------------------------------------------------------------------
# _calculate_metrics
# ---------------------------------------------------------------------------

def test_calculate_metrics_empty(analyzer):
    """Empty outcomes should return zeroed metrics."""
    outcomes = {"buys": [], "sells": []}
    metrics = analyzer._calculate_metrics(outcomes)
    assert metrics["total_decisions"] == 0
    assert metrics["win_rate"] == 0.0
    assert metrics["sharpe_of_decisions"] == 0.0


def test_calculate_metrics_buy_only(analyzer):
    """Buy-only outcomes should have sell_count=0."""
    outcomes = {
        "buys": [
            {"forward_return": 0.05, "success": True},
            {"forward_return": -0.02, "success": False},
            {"forward_return": 0.03, "success": True},
        ],
        "sells": [],
    }
    metrics = analyzer._calculate_metrics(outcomes)
    assert metrics["buy_count"] == 3
    assert metrics["sell_count"] == 0
    assert metrics["buy_accuracy"] == pytest.approx(2/3)
    assert metrics["avg_forward_return_buy"] == pytest.approx(0.02)
    assert metrics["win_rate"] == pytest.approx(2/3)


def test_calculate_metrics_sell_only(analyzer):
    """Sell-only outcomes: avg_forward_return_sell is negated mean."""
    outcomes = {
        "buys": [],
        "sells": [
            {"forward_return": -0.05, "success": True},   # price dropped → good sell
            {"forward_return": 0.02, "success": False},   # price rose → bad sell
        ],
    }
    metrics = analyzer._calculate_metrics(outcomes)
    assert metrics["sell_count"] == 2
    assert metrics["buy_count"] == 0
    assert metrics["sell_accuracy"] == pytest.approx(0.5)
    # avg_forward_return_sell = -mean([-0.05, 0.02]) = -(-0.015) = 0.015
    assert metrics["avg_forward_return_sell"] == pytest.approx(0.015)


def test_calculate_metrics_sharpe_zero_vol(analyzer):
    """Zero volatility in returns should yield sharpe=0.0."""
    outcomes = {
        "buys": [{"forward_return": 0.01, "success": True}],
        "sells": [],
    }
    metrics = analyzer._calculate_metrics(outcomes)
    assert metrics["sharpe_of_decisions"] == 0.0


def test_calculate_metrics_sharpe_with_vol(analyzer):
    """Non-trivial volatility should produce non-zero Sharpe."""
    outcomes = {
        "buys": [
            {"forward_return": 0.05, "success": True},
            {"forward_return": -0.01, "success": False},
            {"forward_return": 0.03, "success": True},
        ],
        "sells": [],
    }
    metrics = analyzer._calculate_metrics(outcomes)
    assert metrics["sharpe_of_decisions"] > 0


def test_calculate_metrics_ignores_nan_forward_returns(analyzer):
    """NaN forward returns should not poison the average return metrics."""
    outcomes = {
        "buys": [
            {"forward_return": 0.05, "success": True},
            {"forward_return": float("nan"), "success": False},
            {"forward_return": -0.03, "success": False},
        ],
        "sells": [
            {"forward_return": -0.04, "success": True},
            {"forward_return": float("nan"), "success": False},
        ],
    }
    metrics = analyzer._calculate_metrics(outcomes)
    # NaN should be ignored in the average, not propagate.
    assert metrics["buy_count"] == 3
    assert metrics["sell_count"] == 2
    assert metrics["avg_forward_return_buy"] == pytest.approx((0.05 - 0.03) / 2)
    assert metrics["avg_forward_return_sell"] == pytest.approx(0.04)
    assert metrics["buy_accuracy"] == pytest.approx(1/3)
    assert metrics["sell_accuracy"] == pytest.approx(0.5)


def test_calculate_metrics_all_nan_returns_default_to_zero(analyzer):
    """If all returns are NaN, average metrics should default to 0.0."""
    outcomes = {
        "buys": [{"forward_return": float("nan"), "success": False}],
        "sells": [],
    }
    metrics = analyzer._calculate_metrics(outcomes)
    assert metrics["avg_forward_return_buy"] == 0.0
    assert metrics["sharpe_of_decisions"] == 0.0


# ---------------------------------------------------------------------------
# analyze_behavioral_patterns
# ---------------------------------------------------------------------------

def test_behavioral_patterns_empty(analyzer):
    """Empty decisions should return default patterns."""
    patterns = analyzer.analyze_behavioral_patterns([])
    assert patterns["avg_trades_per_day"] == 0.0
    assert patterns["unique_assets_traded"] == 0
    assert patterns["overconfidence_check"] == 0.0
    assert patterns["diversification_score"] == 0.0


def test_behavioral_patterns_low_frequency(analyzer):
    """≤2 trades/day should score 1.0 on overconfidence."""
    decisions = [
        {"trades": [{"ticker": "SPY", "action": "buy"}]},
        {"trades": [{"ticker": "TLT", "action": "buy"}]},
    ]
    patterns = analyzer.analyze_behavioral_patterns(decisions)
    assert patterns["avg_trades_per_day"] == 1.0
    assert patterns["overconfidence_check"] == 1.0
    assert patterns["unique_assets_traded"] == 2
    assert patterns["diversification_score"] == 0.2


def test_behavioral_patterns_high_frequency(analyzer):
    """>4 trades/day should score 0.4 on overconfidence."""
    decisions = [
        {"trades": [{"ticker": "SPY", "action": "buy"} for _ in range(5)]},
    ]
    patterns = analyzer.analyze_behavioral_patterns(decisions)
    assert patterns["avg_trades_per_day"] == 5.0
    assert patterns["overconfidence_check"] == 0.4


def test_behavioral_patterns_diversification_cap(analyzer):
    """Diversification score should cap at 1.0."""
    decisions = [
        {"trades": [{"ticker": f"T{i}", "action": "buy"} for i in range(15)]},
    ]
    patterns = analyzer.analyze_behavioral_patterns(decisions)
    assert patterns["diversification_score"] == 1.0


# ---------------------------------------------------------------------------
# _calculate_loss_aversion
# ---------------------------------------------------------------------------

def test_loss_aversion_empty(analyzer):
    """No trades should return neutral 0.5."""
    score = analyzer._calculate_loss_aversion([])
    assert score == 0.5


def test_loss_aversion_all_buys(analyzer):
    """All buys → no sells → score 0.0."""
    decisions = [{"trades": [{"action": "buy"}]} for _ in range(5)]
    score = analyzer._calculate_loss_aversion(decisions)
    assert score == 0.0


def test_loss_aversion_all_sells(analyzer):
    """All sells → score capped at 1.0."""
    decisions = [{"trades": [{"action": "sell"}]} for _ in range(5)]
    score = analyzer._calculate_loss_aversion(decisions)
    assert score == 1.0


def test_loss_aversion_mixed(analyzer):
    """Mixed buy/sell → proportional score."""
    decisions = [
        {"trades": [{"action": "buy"}, {"action": "buy"}]},
        {"trades": [{"action": "sell"}]},
    ]
    score = analyzer._calculate_loss_aversion(decisions)
    # sells=1, total=3, ratio=1/3, score=min(2/3, 1.0)=2/3
    assert score == pytest.approx(2/3)


def test_loss_aversion_more_sells_than_half(analyzer):
    """When sells > 50% of total, the ratio*2 can exceed 1.0 and gets capped."""
    decisions = [
        {"trades": [{"action": "sell"}, {"action": "sell"}]},
        {"trades": [{"action": "buy"}]},
    ]
    score = analyzer._calculate_loss_aversion(decisions)
    # sells=2, total=3, ratio=2/3, raw_score=4/3, capped at 1.0
    assert score == 1.0


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

@patch("analysis.decision_analyzer.fetch_historical_data")
def test_generate_report_with_data(mock_fetch, analyzer, tmp_path, mock_price_data):
    """Report generation should produce formatted output with all sections."""
    analyzer.results_dir = tmp_path
    mock_fetch.return_value = {"SPY": mock_price_data}

    with open(tmp_path / "2026-05-10.json", "w") as f:
        json.dump({
            "date": "2026-05-10",
            "decision": {"actions": [{"ticker": "SPY", "action": "buy"}], "reasoning": "Bullish momentum"},
            "executed_trades": [{"ticker": "SPY", "action": "buy", "price": 176.0}],
            "portfolio_before": {},
            "portfolio_after": {},
        }, f)

    report = analyzer.generate_report(days=30)
    assert "LLM DECISION QUALITY ANALYSIS" in report
    assert "TRADE STATISTICS" in report
    assert "PERFORMANCE METRICS (5-Day Forward)" in report
    assert "BEHAVIORAL ANALYSIS" in report
    assert "ASSESSMENT" in report


def test_generate_report_no_data(analyzer, tmp_path):
    """No data should return a clear message."""
    analyzer.results_dir = tmp_path
    report = analyzer.generate_report(days=30)
    assert "No decision data available" in report


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_skips_no_price(mock_fetch, analyzer):
    """Trades missing the 'price' key should be skipped."""
    mock_fetch.return_value = {"SPY": pd.DataFrame()}
    decisions = [{
        "date": "2026-05-10",
        "trades": [{"ticker": "SPY", "action": "buy"}],  # no price
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    assert metrics["total_decisions"] == 0


@patch("analysis.decision_analyzer.fetch_historical_data")
def test_analyze_outcomes_date_before_data(mock_fetch, analyzer):
    """Decision date before available data should return 0.0 forward return."""
    dates = pd.date_range("2026-05-15", periods=5, freq="D")
    df = pd.DataFrame({"Close": [100, 101, 102, 103, 104]}, index=dates)
    mock_fetch.return_value = {"SPY": df}
    decisions = [{
        "date": "2026-05-10",  # Before data starts
        "trades": [{"ticker": "SPY", "action": "buy", "price": 100.0}],
    }]
    metrics = analyzer.analyze_outcomes(decisions, forward_days=5)
    # mask = df.index >= "2026-05-10" → all True, so it actually uses the data
    assert metrics["buy_count"] == 1
    assert metrics["win_rate"] == 1.0  # Price went up

    # If date is after all data:
    decisions2 = [{
        "date": "2026-05-20",  # After data ends
        "trades": [{"ticker": "SPY", "action": "buy", "price": 100.0}],
    }]
    metrics2 = analyzer.analyze_outcomes(decisions2, forward_days=5)
    # mask = df.index >= "2026-05-20" → all False → _get_forward_return returns 0.0
    # forward_return=0.0 → success=False → total_decisions=1, win_rate=0.0
    assert metrics2["total_decisions"] == 1
    assert metrics2["win_rate"] == 0.0
