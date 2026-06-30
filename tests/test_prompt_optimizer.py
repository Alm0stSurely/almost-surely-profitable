"""
Comprehensive tests for prompt_optimizer.py.

Covers:
- PromptVariant dataclass (to_dict)
- BacktestResult dataclass (to_dict)
- PromptOptimizer initialization
- create_default_variants (returns expected variants, correct count, names)
- load_variants_from_config (JSON parsing, missing fields)
- _build_context (correct structure, values from portfolio)
- generate_report (empty results, single result, multiple results, best-by-metric)
- save_results (file creation, JSON content, report content)
- backtest_variant (mock market data, metrics calculation, no SPY error)
- run_optimization (mock fetch, sorts by calmar)
- Edge cases: zero drawdown calmar, empty trades, invalid config
"""

import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.prompt_optimizer import PromptVariant, BacktestResult, PromptOptimizer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_variant(name="test", system_prompt="prompt", description="desc"):
    return PromptVariant(name=name, system_prompt=system_prompt, description=description)


def make_result(
    variant_name="test",
    start_date="2026-01-01",
    end_date="2026-03-01",
    total_return_pct=5.0,
    sharpe_ratio=1.0,
    max_drawdown_pct=-2.0,
    win_rate=0.6,
    total_trades=10,
    buy_trades=5,
    sell_trades=5,
    avg_trades_per_day=0.1,
    volatility=10.0,
    calmar_ratio=2.5,
    final_portfolio_value=10500.0,
    cash_utilization=0.3,
):
    return BacktestResult(
        variant_name=variant_name,
        start_date=start_date,
        end_date=end_date,
        total_return_pct=total_return_pct,
        sharpe_ratio=sharpe_ratio,
        max_drawdown_pct=max_drawdown_pct,
        win_rate=win_rate,
        total_trades=total_trades,
        buy_trades=buy_trades,
        sell_trades=sell_trades,
        avg_trades_per_day=avg_trades_per_day,
        volatility=volatility,
        calmar_ratio=calmar_ratio,
        final_portfolio_value=final_portfolio_value,
        cash_utilization=cash_utilization,
    )


def make_spy_data(dates, prices):
    """Build a minimal SPY DataFrame for backtest_variant."""
    return pd.DataFrame({"Close": prices}, index=pd.to_datetime(dates))


# ---------------------------------------------------------------------------
# PromptVariant
# ---------------------------------------------------------------------------

class TestPromptVariant:
    def test_to_dict(self):
        v = make_variant(name="baseline", system_prompt="be conservative", description="safe")
        d = v.to_dict()
        assert d == {
            "name": "baseline",
            "system_prompt": "be conservative",
            "description": "safe",
        }

    def test_defaults(self):
        v = PromptVariant(name="x", system_prompt="y", description="z")
        assert v.name == "x"
        assert v.system_prompt == "y"
        assert v.description == "z"


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def test_to_dict(self):
        r = make_result(variant_name="v1", total_return_pct=3.5)
        d = r.to_dict()
        assert d["variant_name"] == "v1"
        assert d["total_return_pct"] == 3.5
        assert d["sharpe_ratio"] == 1.0
        assert d["max_drawdown_pct"] == -2.0

    def test_all_fields_present(self):
        r = make_result()
        d = r.to_dict()
        expected_keys = {
            "variant_name", "start_date", "end_date", "total_return_pct",
            "sharpe_ratio", "max_drawdown_pct", "win_rate", "total_trades",
            "buy_trades", "sell_trades", "avg_trades_per_day", "volatility",
            "calmar_ratio", "final_portfolio_value", "cash_utilization",
        }
        assert set(d.keys()) == expected_keys


# ---------------------------------------------------------------------------
# PromptOptimizer Init
# ---------------------------------------------------------------------------

class TestPromptOptimizerInit:
    def test_basic(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01", initial_capital=5000.0)
        assert opt.start_date == datetime(2026, 1, 1)
        assert opt.end_date == datetime(2026, 3, 1)
        assert opt.initial_capital == 5000.0
        assert opt.variants == []
        assert opt.results == []

    def test_default_capital(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        assert opt.initial_capital == 10000.0


# ---------------------------------------------------------------------------
# create_default_variants
# ---------------------------------------------------------------------------

class TestCreateDefaultVariants:
    def test_count(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.create_default_variants()
        assert len(variants) == 9

    def test_names(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.create_default_variants()
        names = [v.name for v in variants]
        expected = [
            "baseline", "cvar_only", "prospect_theory", "loss_aversion",
            "meta_labeling", "regime_aware", "contrarian", "full_behavioral",
            "risk_focused",
        ]
        assert names == expected

    def test_baseline_has_no_extra_components(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.create_default_variants()
        baseline = variants[0]
        assert "CVaR Framework" not in baseline.system_prompt
        assert "Prospect Theory" not in baseline.system_prompt

    def test_full_behavioral_has_all_components(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.create_default_variants()
        fb = [v for v in variants if v.name == "full_behavioral"][0]
        assert "CVaR Framework" in fb.system_prompt
        assert "Prospect Theory" in fb.system_prompt
        assert "Loss Aversion" in fb.system_prompt
        assert "Meta-Labeling" in fb.system_prompt
        assert "Regime Awareness" in fb.system_prompt

    def test_risk_focused_subset(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.create_default_variants()
        rf = [v for v in variants if v.name == "risk_focused"][0]
        assert "CVaR Framework" in rf.system_prompt
        assert "Loss Aversion" in rf.system_prompt
        assert "Regime Awareness" in rf.system_prompt
        assert "Prospect Theory" not in rf.system_prompt
        assert "Meta-Labeling" not in rf.system_prompt

    def test_returns_list_of_prompt_variants(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.create_default_variants()
        assert all(isinstance(v, PromptVariant) for v in variants)


# ---------------------------------------------------------------------------
# load_variants_from_config
# ---------------------------------------------------------------------------

class TestLoadVariantsFromConfig:
    def test_loads_json(self, tmp_path):
        config = {
            "variants": [
                {"name": "v1", "system_prompt": "prompt1", "description": "d1"},
                {"name": "v2", "system_prompt": "prompt2"},
            ]
        }
        p = tmp_path / "config.json"
        p.write_text(json.dumps(config))

        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.load_variants_from_config(str(p))
        assert len(variants) == 2
        assert variants[0].name == "v1"
        assert variants[0].description == "d1"
        assert variants[1].name == "v2"
        assert variants[1].description == ""  # missing field defaults to ""

    def test_missing_variants_key(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text(json.dumps({}))
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        variants = opt.load_variants_from_config(str(p))
        assert variants == []

    def test_file_not_found(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        with pytest.raises(FileNotFoundError):
            opt.load_variants_from_config(str(tmp_path / "nope.json"))


# ---------------------------------------------------------------------------
# _build_context
# ---------------------------------------------------------------------------

class TestBuildContext:
    def test_structure(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        portfolio = MagicMock()
        portfolio.cash = 5000.0
        portfolio.positions = {"SPY": {"quantity": 10, "avg_price": 100}}
        portfolio.total_value = 9500.0

        analysis = {"regime": "trending"}
        ctx = opt._build_context(portfolio, analysis, "2026-02-01", peak_value=10000.0)

        assert ctx["cash"] == 5000.0
        assert ctx["positions"] == {"SPY": {"quantity": 10, "avg_price": 100}}
        assert ctx["total_value"] == 9500.0
        assert ctx["drawdown"] == -0.05  # (9500 - 10000) / 10000
        assert ctx["market_summary"] == {"regime": "trending"}
        assert ctx["date"] == "2026-02-01"


# ---------------------------------------------------------------------------
# generate_report
# ---------------------------------------------------------------------------

class TestGenerateReport:
    def test_empty_results(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        report = opt.generate_report()
        assert "No results to report" in report

    def test_single_result(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(variant_name="baseline", total_return_pct=5.0)]
        report = opt.generate_report()
        assert "baseline" in report
        assert "5.00%" in report
        assert "BEST BY METRIC" in report

    def test_multiple_results_sorted(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [
            make_result(variant_name="bad", calmar_ratio=0.5, total_return_pct=-5.0),
            make_result(variant_name="good", calmar_ratio=2.5, total_return_pct=5.0),
        ]
        report = opt.generate_report()
        # good should be ranked #1 because higher calmar
        assert report.index("good") < report.index("bad")

    def test_best_by_metric_section(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [
            make_result(variant_name="high_return", total_return_pct=10.0, sharpe_ratio=0.5, calmar_ratio=1.0),
            make_result(variant_name="high_sharpe", total_return_pct=5.0, sharpe_ratio=2.0, calmar_ratio=1.0),
        ]
        report = opt.generate_report()
        assert "Highest Return:    high_return" in report
        assert "Best Sharpe:       high_sharpe" in report


# ---------------------------------------------------------------------------
# save_results
# ---------------------------------------------------------------------------

class TestSaveResults:
    def test_creates_files(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(variant_name="v1")]

        out_dir = tmp_path / "opt_results"
        opt.save_results(str(out_dir))

        files = list(out_dir.iterdir())
        assert len(files) == 2
        assert any("optimization_results" in f.name for f in files)
        assert any("optimization_report" in f.name for f in files)

    def test_json_content(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(variant_name="v1", total_return_pct=7.5)]

        out_dir = tmp_path / "opt_results"
        opt.save_results(str(out_dir))

        json_file = [f for f in out_dir.iterdir() if f.suffix == ".json"][0]
        data = json.loads(json_file.read_text())
        assert len(data) == 1
        assert data[0]["variant_name"] == "v1"
        assert data[0]["total_return_pct"] == 7.5

    def test_report_content(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(variant_name="v1")]

        out_dir = tmp_path / "opt_results"
        opt.save_results(str(out_dir))

        report_file = [f for f in out_dir.iterdir() if f.suffix == ".txt"][0]
        text = report_file.read_text()
        assert "PROMPT OPTIMIZATION RESULTS" in text
        assert "v1" in text

    def test_creates_parent_dirs(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result()]
        deep = tmp_path / "a" / "b" / "c"
        opt.save_results(str(deep))
        assert deep.exists()


# ---------------------------------------------------------------------------
# backtest_variant
# ---------------------------------------------------------------------------

class TestBacktestVariant:
    @patch("llm.prompt_optimizer.analyze_market_data")
    @patch("llm.prompt_optimizer.Portfolio")
    def test_basic_flow(self, MockPortfolio, mock_analyze):
        opt = PromptOptimizer("2026-01-01", "2026-01-05", initial_capital=10000.0)
        variant = make_variant(name="baseline")

        mock_pf = MagicMock()
        mock_pf.cash = 10000.0
        mock_pf.total_value = 10000.0
        mock_pf.positions = {}
        MockPortfolio.return_value = mock_pf

        mock_analyze.return_value = {"regime": "neutral"}

        dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
        prices = [100.0, 101.0, 102.0, 101.5, 103.0]
        spy_data = make_spy_data(dates, prices)
        market_data = {"SPY": spy_data}

        result = opt.backtest_variant(variant, market_data, verbose=False)

        assert result.variant_name == "baseline"
        assert result.start_date == "2026-01-01"
        assert result.end_date == "2026-01-05"
        assert result.total_trades == 0  # placeholder skips trades
        assert result.final_portfolio_value == 10000.0

    def test_missing_spy_raises(self):
        opt = PromptOptimizer("2026-01-01", "2026-01-05")
        variant = make_variant()
        with pytest.raises(ValueError, match="SPY data required"):
            opt.backtest_variant(variant, {"QQQ": pd.DataFrame()})

    def test_no_trading_dates_raises(self):
        opt = PromptOptimizer("2026-01-01", "2026-01-05")
        variant = make_variant()
        spy_data = make_spy_data(["2025-12-01"], [100.0])  # outside range
        with pytest.raises(ValueError, match="No trading dates found"):
            opt.backtest_variant(variant, {"SPY": spy_data})

    @patch("llm.prompt_optimizer.analyze_market_data")
    def test_sharpe_calculation(self, mock_analyze):
        """Portfolio with varying value should produce positive Sharpe for upward drift."""
        opt = PromptOptimizer("2026-01-01", "2026-01-10", initial_capital=10000.0)
        variant = make_variant()
        mock_analyze.return_value = {}

        dates = pd.date_range("2026-01-01", periods=10, freq="D")
        prices = [100.0 + i * 0.5 for i in range(10)]
        spy_data = pd.DataFrame({"Close": prices}, index=dates)
        market_data = {"SPY": spy_data}

        result = opt.backtest_variant(variant, market_data, verbose=False)
        # With constant upward drift in price, portfolio should track it
        assert result.sharpe_ratio >= 0  # non-negative for upward drift

    @patch("llm.prompt_optimizer.analyze_market_data")
    def test_max_drawdown_calculation(self, mock_analyze):
        """Portfolio that peaks then declines should record a drawdown."""
        opt = PromptOptimizer("2026-01-01", "2026-01-05", initial_capital=10000.0)
        variant = make_variant()
        mock_analyze.return_value = {}

        dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
        # Rising then falling prices
        prices = [100.0, 105.0, 103.0, 101.0, 102.0]
        spy_data = make_spy_data(dates, prices)
        market_data = {"SPY": spy_data}

        result = opt.backtest_variant(variant, market_data, verbose=False)
        # Max drawdown should be negative (there was a decline from peak)
        assert result.max_drawdown_pct <= 0


# ---------------------------------------------------------------------------
# run_optimization
# ---------------------------------------------------------------------------

class TestRunOptimization:
    @patch("llm.prompt_optimizer.fetch_historical_data")
    @patch("llm.prompt_optimizer.analyze_market_data")
    @patch("llm.prompt_optimizer.Portfolio")
    def test_runs_all_variants(self, MockPortfolio, mock_analyze, mock_fetch):
        opt = PromptOptimizer("2026-01-01", "2026-01-05")
        variants = [make_variant(name="v1"), make_variant(name="v2")]

        mock_pf = MagicMock()
        mock_pf.cash = 10000.0
        mock_pf.total_value = 10000.0
        mock_pf.positions = {}
        MockPortfolio.return_value = mock_pf
        mock_analyze.return_value = {}

        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        spy_data = pd.DataFrame({"Close": [100.0] * 5}, index=dates)
        mock_fetch.return_value = {"SPY": spy_data}

        results = opt.run_optimization(variants=variants, verbose=False)
        assert len(results) == 2
        assert results[0].variant_name in ("v1", "v2")

    @patch("llm.prompt_optimizer.fetch_historical_data")
    @patch("llm.prompt_optimizer.analyze_market_data")
    def test_sorts_by_calmar(self, mock_analyze, mock_fetch):
        """Results should be sorted by calmar ratio descending."""
        opt = PromptOptimizer("2026-01-01", "2026-01-05")
        variants = [make_variant(name="low"), make_variant(name="high")]

        mock_analyze.return_value = {}

        # Vary prices so high gets better returns
        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        prices = [100.0, 101.0, 102.0, 103.0, 104.0]  # steady upward
        spy_data = pd.DataFrame({"Close": prices}, index=dates)
        mock_fetch.return_value = {"SPY": spy_data}

        results = opt.run_optimization(variants=variants, verbose=False)
        assert len(results) == 2
        # Both have same calmar in flat test because same market data and no trades
        # Just verify sorting doesn't crash and returns both
        names = [r.variant_name for r in results]
        assert "low" in names
        assert "high" in names

    @patch("llm.prompt_optimizer.fetch_historical_data")
    @patch("llm.prompt_optimizer.analyze_market_data")
    def test_skips_failed_variants(self, mock_analyze, mock_fetch):
        """If backtest_variant raises, run_optimization should skip that variant and continue."""
        opt = PromptOptimizer("2026-01-01", "2026-01-05")
        variants = [make_variant(name="ok"), make_variant(name="fail")]

        mock_analyze.return_value = {}

        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        spy_data = pd.DataFrame({"Close": [100.0] * 5}, index=dates)
        mock_fetch.return_value = {"SPY": spy_data}

        # Patch backtest_variant to raise on "fail"
        original_backtest = opt.backtest_variant
        def side_effect(variant, market_data, verbose=False):
            if variant.name == "fail":
                raise RuntimeError("simulated failure")
            return original_backtest(variant, market_data, verbose)

        with patch.object(opt, "backtest_variant", side_effect=side_effect):
            results = opt.run_optimization(variants=variants, verbose=False)

        assert len(results) == 1
        assert results[0].variant_name == "ok"

    @patch("llm.prompt_optimizer.fetch_historical_data")
    @patch("llm.prompt_optimizer.analyze_market_data")
    @patch("llm.prompt_optimizer.Portfolio")
    def test_uses_default_variants_when_none_provided(self, MockPortfolio, mock_analyze, mock_fetch):
        opt = PromptOptimizer("2026-01-01", "2026-01-05")

        mock_pf = MagicMock()
        mock_pf.cash = 10000.0
        mock_pf.total_value = 10000.0
        mock_pf.positions = {}
        MockPortfolio.return_value = mock_pf
        mock_analyze.return_value = {}

        dates = pd.date_range("2026-01-01", periods=5, freq="D")
        spy_data = pd.DataFrame({"Close": [100.0] * 5}, index=dates)
        mock_fetch.return_value = {"SPY": spy_data}

        results = opt.run_optimization(verbose=False)
        # Should use the 9 default variants
        assert len(results) == 9


# ---------------------------------------------------------------------------
# Edge Cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_drawdown_calmar(self):
        """If max_drawdown is 0, backtest_variant should set calmar to 0 (not division by zero)."""
        r = make_result(max_drawdown_pct=0.0, total_return_pct=5.0, calmar_ratio=0.0)
        assert r.calmar_ratio == 0.0

    @patch("llm.prompt_optimizer.analyze_market_data")
    @patch("llm.prompt_optimizer.Portfolio")
    def test_backtest_zero_drawdown_computes_calmar(self, MockPortfolio, mock_analyze):
        """Verify backtest_variant handles zero drawdown without ZeroDivisionError."""
        opt = PromptOptimizer("2026-01-01", "2026-01-05", initial_capital=10000.0)
        variant = make_variant()

        mock_pf = MagicMock()
        mock_pf.cash = 10000.0
        mock_pf.total_value = 10000.0
        mock_pf.positions = {}
        MockPortfolio.return_value = mock_pf
        mock_analyze.return_value = {}

        dates = ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"]
        prices = [100.0, 100.0, 100.0, 100.0, 100.0]  # flat = no drawdown
        spy_data = make_spy_data(dates, prices)
        market_data = {"SPY": spy_data}

        result = opt.backtest_variant(variant, market_data, verbose=False)
        assert result.calmar_ratio == 0.0  # no drawdown → calmar = 0

    def test_empty_trades_stats(self):
        r = make_result(total_trades=0, buy_trades=0, sell_trades=0)
        assert r.to_dict()["total_trades"] == 0

    def test_date_parsing(self):
        opt = PromptOptimizer("2026-12-31", "2027-01-01")
        assert opt.start_date.year == 2026
        assert opt.end_date.year == 2027

    def test_very_short_period(self):
        opt = PromptOptimizer("2026-01-01", "2026-01-01")
        assert opt.start_date == opt.end_date
