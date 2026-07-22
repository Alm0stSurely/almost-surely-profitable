"""
Benchmark for the performance_metrics non-finite input/output guards.

Exercises calculate_sortino_ratio, calculate_calmar_ratio, and calculate_all_metrics
with NaN/Inf inputs and with degenerate finite inputs (zero drawdown, no downside),
asserting that the output is always finite and JSON-serializable.

Run under -W error::RuntimeWarning to ensure no silent numerical failures.
"""

import sys
import json
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from risk.performance_metrics import (
    calculate_sortino_ratio,
    calculate_calmar_ratio,
    calculate_all_metrics,
    PerformanceMetrics,
)


def _assert_zeroed(metrics: PerformanceMetrics):
    assert metrics.total_return == 0.0
    assert metrics.annualized_return == 0.0
    assert metrics.sharpe_ratio == 0.0
    assert metrics.sortino_ratio == 0.0
    assert metrics.calmar_ratio == 0.0
    assert metrics.volatility == 0.0
    assert metrics.max_drawdown == 0.0
    assert metrics.treynor_ratio is None
    assert metrics.beta is None
    assert metrics.alpha is None
    assert metrics.information_ratio is None
    assert metrics.tracking_error is None


def benchmark_sortino_nan():
    returns = np.array([0.01, -0.02, np.nan, 0.005])
    assert calculate_sortino_ratio(returns) == 0.0


def benchmark_sortino_inf():
    returns = np.array([0.01, np.inf, -0.02, 0.005])
    assert calculate_sortino_ratio(returns) == 0.0


def benchmark_sortino_no_downside():
    """All-positive returns used to produce +inf; now the pipeline stays finite."""
    returns = np.full(30, 0.002)
    sortino = calculate_sortino_ratio(returns, risk_free_rate=0.0)
    assert sortino == 0.0


def benchmark_calmar_nan():
    returns = np.array([0.01, -0.02, np.nan, 0.005])
    assert calculate_calmar_ratio(returns) == 0.0


def benchmark_calmar_inf():
    returns = np.array([0.01, np.inf, -0.02, 0.005])
    assert calculate_calmar_ratio(returns) == 0.0


def benchmark_calmar_no_drawdown():
    """A monotonic positive series used to produce +inf; now it stays finite."""
    returns = np.full(30, 0.001)
    calmar = calculate_calmar_ratio(returns)
    assert calmar == 0.0


def benchmark_all_metrics_nan():
    returns = np.array([0.01, -0.02, np.nan, 0.005])
    metrics = calculate_all_metrics(returns)
    _assert_zeroed(metrics)


def benchmark_all_metrics_inf():
    returns = np.array([0.01, np.inf, -0.02, 0.005])
    metrics = calculate_all_metrics(returns)
    _assert_zeroed(metrics)


def benchmark_all_metrics_degenerate_finite():
    """A short, all-positive vector should yield a fully finite metrics object."""
    returns = np.array([0.005, 0.003, 0.002])
    metrics = calculate_all_metrics(returns)
    assert np.isfinite(metrics.sharpe_ratio)
    assert metrics.sortino_ratio == 0.0
    assert metrics.calmar_ratio == 0.0
    assert np.isfinite(metrics.volatility)
    assert np.isfinite(metrics.max_drawdown)
    assert metrics.max_drawdown <= 0.0


def benchmark_all_metrics_json_serializable():
    """Downstream consumers (LLM prompts, reports) require valid JSON."""
    returns = np.array([0.005, 0.003, -0.002])
    metrics = calculate_all_metrics(returns)
    payload = {
        'total_return': metrics.total_return,
        'annualized_return': metrics.annualized_return,
        'sharpe_ratio': metrics.sharpe_ratio,
        'sortino_ratio': metrics.sortino_ratio,
        'calmar_ratio': metrics.calmar_ratio,
        'volatility': metrics.volatility,
        'max_drawdown': metrics.max_drawdown,
        'beta': metrics.beta,
        'alpha': metrics.alpha,
        'treynor_ratio': metrics.treynor_ratio,
        'information_ratio': metrics.information_ratio,
        'tracking_error': metrics.tracking_error,
    }
    text = json.dumps(payload, allow_nan=False)
    assert 'Infinity' not in text
    assert 'NaN' not in text
    assert json.loads(text) == payload


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        benchmark_sortino_nan()
        benchmark_sortino_inf()
        benchmark_sortino_no_downside()
        benchmark_calmar_nan()
        benchmark_calmar_inf()
        benchmark_calmar_no_drawdown()
        benchmark_all_metrics_nan()
        benchmark_all_metrics_inf()
        benchmark_all_metrics_degenerate_finite()
        benchmark_all_metrics_json_serializable()
    print("All performance_metrics non-finite benchmarks passed without RuntimeWarning.")
