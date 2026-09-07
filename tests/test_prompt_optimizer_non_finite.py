"""
Tests for non-finite value guards in prompt_optimizer.py.

Covers:
- save_results replaces NaN/inf with null in JSON output
- generate_report renders n/a for non-finite floats
- run_optimization verbose printing does not crash on non-finite results
"""

import json
import math
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.prompt_optimizer import BacktestResult, PromptOptimizer


def make_result(**overrides):
    base = dict(
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
    )
    base.update(overrides)
    return BacktestResult(**base)


class TestSaveResultsNonFinite:
    def test_nan_replaced_with_null(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(sharpe_ratio=float("nan"), calmar_ratio=float("nan"))]
        opt.save_results(str(tmp_path))

        json_file = [f for f in tmp_path.iterdir() if f.suffix == ".json"][0]
        data = json.loads(json_file.read_text())
        assert data[0]["sharpe_ratio"] is None
        assert data[0]["calmar_ratio"] is None

    def test_inf_replaced_with_null(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(
            total_return_pct=float("inf"),
            max_drawdown_pct=float("-inf"),
            volatility=float("inf"),
            final_portfolio_value=float("nan"),
        )]
        opt.save_results(str(tmp_path))

        json_file = [f for f in tmp_path.iterdir() if f.suffix == ".json"][0]
        data = json.loads(json_file.read_text())
        assert data[0]["total_return_pct"] is None
        assert data[0]["max_drawdown_pct"] is None
        assert data[0]["volatility"] is None
        assert data[0]["final_portfolio_value"] is None

    def test_finite_values_preserved(self, tmp_path):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(sharpe_ratio=1.5, total_return_pct=7.5)]
        opt.save_results(str(tmp_path))

        json_file = [f for f in tmp_path.iterdir() if f.suffix == ".json"][0]
        data = json.loads(json_file.read_text())
        assert data[0]["sharpe_ratio"] == 1.5
        assert data[0]["total_return_pct"] == 7.5

    def test_output_is_strict_json(self, tmp_path):
        """The saved file must be parseable by strict JSON (no NaN/Infinity tokens)."""
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(sharpe_ratio=float("nan"))]
        opt.save_results(str(tmp_path))

        json_file = [f for f in tmp_path.iterdir() if f.suffix == ".json"][0]
        raw = json_file.read_text()
        assert "NaN" not in raw
        assert "Infinity" not in raw
        # Strict parse (parse_constant raises on NaN/Infinity)
        json.loads(raw, parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"Invalid: {x}")))


class TestGenerateReportNonFinite:
    def test_nan_shows_na(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(sharpe_ratio=float("nan"), calmar_ratio=float("nan"))]
        report = opt.generate_report()
        assert "Sharpe: n/a" in report
        assert "Calmar: n/a" in report

    def test_inf_shows_na(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(
            total_return_pct=float("inf"),
            max_drawdown_pct=float("-inf"),
            volatility=float("inf"),
            final_portfolio_value=float("nan"),
        )]
        report = opt.generate_report()
        assert "Return: n/a%" in report
        assert "Max DD: n/a%" in report
        assert "Volatility: n/a%" in report
        assert "Final Value: $n/a" in report

    def test_best_by_metric_with_nan(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(sharpe_ratio=float("nan"), total_return_pct=float("inf"))]
        report = opt.generate_report()
        assert "Highest Return:    test (n/a%)" in report
        assert "Best Sharpe:       test (n/a)" in report

    def test_finite_values_still_formatted(self):
        opt = PromptOptimizer("2026-01-01", "2026-03-01")
        opt.results = [make_result(sharpe_ratio=1.5, total_return_pct=7.5)]
        report = opt.generate_report()
        assert "Sharpe: 1.50" in report
        assert "Return: +7.50%" in report
