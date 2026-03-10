"""
Example usage of Deflated Sharpe Ratio.

Demonstrates how to:
1. Calculate DSR for single strategy
2. Compare multiple strategies with multiple testing correction
3. Use Probabilistic Sharpe Ratio
4. Calculate minimum track record length
"""

import numpy as np
import matplotlib.pyplot as plt
from src.backtest.deflated_sharpe import (
    DeflatedSharpeRatio,
    probabilistic_sharpe_ratio,
    minimum_track_record_length,
    SharpeMetrics
)


def generate_strategy_returns(
    annual_return: float,
    annual_vol: float,
    n_days: int = 252,
    skewness: float = 0.0,
    seed: int = None
) -> np.ndarray:
    """Generate synthetic strategy returns with specified characteristics."""
    if seed is not None:
        np.random.seed(seed)
    
    daily_return = annual_return / 252
    daily_vol = annual_vol / np.sqrt(252)
    
    # Generate returns with approximate skewness
    # (simplified - not exact, but sufficient for demo)
    returns = np.random.normal(daily_return, daily_vol, n_days)
    
    if skewness != 0:
        # Add skewness through power transformation
        # Positive skew: use exp, negative: use log-like
        if skewness > 0:
            returns = np.exp(returns) - 1
            returns = (returns - np.mean(returns)) / np.std(returns) * daily_vol + daily_return
        else:
            returns = np.sign(returns) * np.log1p(np.abs(returns))
            returns = (returns - np.mean(returns)) / np.std(returns) * daily_vol + daily_return
    
    return returns


def example_single_strategy():
    """Example 1: Calculate DSR for a single strategy."""
    print("=" * 60)
    print("Example 1: Single Strategy DSR")
    print("=" * 60)
    
    # Generate returns for a "good" strategy
    returns = generate_strategy_returns(
        annual_return=0.15,  # 15% annual return
        annual_vol=0.10,     # 10% volatility
        n_days=252,
        seed=42
    )
    
    # Case 1: Testing only this strategy
    print("\nCase 1: Testing 1 strategy")
    dsr = DeflatedSharpeRatio(n_trials=1)
    metrics = dsr.calculate(returns)
    
    print(f"Sharpe Ratio: {metrics.sharpe_ratio:.3f}")
    print(f"Deflated SR:  {metrics.deflated_sharpe:.3f}")
    print(f"Skewness:     {metrics.skewness:.3f}")
    print(f"Kurtosis:     {metrics.kurtosis:.3f}")
    print(f"P-value:      {metrics.p_value:.4f}")
    print(f"Significant:  {metrics.is_significant}")
    
    # Case 2: Testing 100 strategies (multiple testing)
    print("\nCase 2: Testing 100 strategies (multiple testing)")
    dsr_100 = DeflatedSharpeRatio(n_trials=100)
    metrics_100 = dsr_100.calculate(returns)
    
    print(f"Sharpe Ratio: {metrics_100.sharpe_ratio:.3f}")
    print(f"Deflated SR:  {metrics_100.deflated_sharpe:.3f}")
    print(f"P-value:      {metrics_100.p_value:.4f}")
    print(f"Significant:  {metrics_100.is_significant}")
    print("\nNote: Same returns, but DSR is lower due to multiple testing correction!")


def example_compare_strategies():
    """Example 2: Compare multiple strategies with multiple testing correction."""
    print("\n" + "=" * 60)
    print("Example 2: Comparing Multiple Strategies")
    print("=" * 60)
    
    # Generate returns for multiple strategies
    strategies = [
        ("Strategy A (Good)", generate_strategy_returns(0.12, 0.08, seed=1)),
        ("Strategy B (OK)", generate_strategy_returns(0.08, 0.10, seed=2)),
        ("Strategy C (Noise)", generate_strategy_returns(0.02, 0.15, seed=3)),
        ("Strategy D (Lucky)", generate_strategy_returns(0.15, 0.20, seed=4)),  # High vol
        ("Strategy E (Bad)", generate_strategy_returns(-0.05, 0.12, seed=5)),
    ]
    
    # Compare with multiple testing correction
    dsr = DeflatedSharpeRatio(n_trials=len(strategies))
    results = dsr.compare_strategies(strategies)
    
    print(f"\n{'Rank':<5} {'Strategy':<20} {'SR':<8} {'DSR':<8} {'P-value':<10} {'Signif.'}")
    print("-" * 65)
    
    for rank, (name, metrics) in enumerate(results, 1):
        sig_marker = "✓" if metrics.is_significant else "✗"
        print(f"{rank:<5} {name:<20} {metrics.sharpe_ratio:<8.3f} "
              f"{metrics.deflated_sharpe:<8.3f} {metrics.p_value:<10.4f} {sig_marker}")
    
    print("\nNote: DSR ranking may differ from SR ranking due to non-normality adjustments!")


def example_probabilistic_sharpe():
    """Example 3: Probabilistic Sharpe Ratio."""
    print("\n" + "=" * 60)
    print("Example 3: Probabilistic Sharpe Ratio (PSR)")
    print("=" * 60)
    
    # Strategy with SR = 1.2 over 2 years
    returns = generate_strategy_returns(0.12, 0.10, n_days=504, seed=42)
    
    observed_sr = np.mean(returns) / np.std(returns) * np.sqrt(252)
    print(f"\nObserved Sharpe Ratio: {observed_sr:.3f}")
    
    # PSR vs different benchmarks
    benchmarks = [0.0, 0.5, 1.0, 1.2, 1.5]
    
    print(f"\n{'Benchmark SR':<15} {'PSR (prob > benchmark)':<25}")
    print("-" * 45)
    
    for bench in benchmarks:
        psr = probabilistic_sharpe_ratio(
            observed_sr=observed_sr,
            benchmark_sr=bench,
            n_observations=len(returns),
            skewness=stats.skew(returns),
            kurtosis=stats.kurtosis(returns, fisher=False)
        )
        print(f"{bench:<15.2f} {psr:<25.3f}")
    
    print("\nPSR answers: 'What's the probability this strategy's SR is better than X?'")


def example_track_record_length():
    """Example 4: Minimum track record length."""
    print("\n" + "=" * 60)
    print("Example 4: Minimum Track Record Length")
    print("=" * 60)
    
    target_sharpes = [0.5, 1.0, 1.5, 2.0]
    confidence_levels = [0.90, 0.95, 0.99]
    
    print(f"\nMinimum observations needed to confirm SR target with confidence:")
    print(f"\n{'Target SR':<12} {'90% conf':<12} {'95% conf':<12} {'99% conf':<12}")
    print("-" * 50)
    
    for target_sr in target_sharpes:
        row = [f"{target_sr:.1f}"]
        for conf in confidence_levels:
            n_min = minimum_track_record_length(
                target_sharpe=target_sr,
                confidence_level=conf
            )
            row.append(f"{n_min}")
        print(f"{row[0]:<12} {row[1]:<12} {row[2]:<12} {row[3]:<12}")
    
    print("\nInterpretation: A SR of 1.0 requires ~250 days (1 year) at 95% confidence.")
    print("Many 'good' strategies fail this test - they haven't been tested long enough!")


def example_false_discovery():
    """Example 5: False Discovery Rate control."""
    print("\n" + "=" * 60)
    print("Example 5: False Discovery Rate (FDR) Control")
    print("=" * 60)
    
    # Simulate p-values from 20 strategy tests
    # (5 truly significant, 15 noise)
    np.random.seed(42)
    
    # 5 good strategies (low p-values)
    good_pvalues = np.random.uniform(0.001, 0.03, 5)
    
    # 15 noise strategies (higher p-values)
    noise_pvalues = np.random.uniform(0.05, 0.8, 15)
    
    all_pvalues = np.concatenate([good_pvalues, noise_pvalues])
    np.random.shuffle(all_pvalues)
    
    dsr = DeflatedSharpeRatio(significance_level=0.05)
    
    # Bonferroni correction (conservative)
    print("\nBonferroni correction (controls Family-Wise Error Rate):")
    bonferroni_results = dsr.false_discovery_rate(
        all_pvalues.tolist(),
        method="bonferroni"
    )
    
    n_sig_bonf = sum(1 for _, _, sig in bonferroni_results if sig)
    print(f"Significant strategies: {n_sig_bonf} / {len(all_pvalues)}")
    
    # Benjamini-Hochberg (less conservative)
    print("\nBenjamini-Hochberg correction (controls False Discovery Rate):")
    bh_results = dsr.false_discovery_rate(
        all_pvalues.tolist(),
        method="benjamini-hochberg"
    )
    
    n_sig_bh = sum(1 for _, _, sig in bh_results if sig)
    print(f"Significant strategies: {n_sig_bh} / {len(all_pvalues)}")
    
    print("\nFDR control is less conservative but allows some false positives.")
    print("Choose based on your tolerance for false discoveries vs missed opportunities.")


def plot_dsr_vs_sr():
    """Plot showing how DSR differs from SR with multiple testing."""
    print("\n" + "=" * 60)
    print("Visualization: DSR vs SR with Multiple Testing")
    print("=" * 60)
    
    # Generate returns for strategy with SR ≈ 1.0
    returns = generate_strategy_returns(0.10, 0.10, n_days=252, seed=42)
    
    n_trials_range = [1, 5, 10, 20, 50, 100, 200, 500, 1000]
    sr_values = []
    dsr_values = []
    
    for n_trials in n_trials_range:
        dsr_calc = DeflatedSharpeRatio(n_trials=n_trials)
        metrics = dsr_calc.calculate(returns)
        sr_values.append(metrics.sharpe_ratio)
        dsr_values.append(metrics.deflated_sharpe)
    
    print(f"\nSharpe Ratio (constant): {sr_values[0]:.3f}")
    print(f"\n{'N Trials':<12} {'SR':<10} {'DSR':<10} {'Difference':<12}")
    print("-" * 50)
    
    for i, n in enumerate(n_trials_range):
        diff = sr_values[i] - dsr_values[i]
        print(f"{n:<12} {sr_values[i]:<10.3f} {dsr_values[i]:<10.3f} {diff:<12.3f}")
    
    print("\nAs number of tested strategies increases, DSR decreases.")
    print("A strategy with SR=1.0 may have DSR<0 if enough strategies were tested!")


if __name__ == "__main__":
    from scipy import stats
    
    print("\n" + "=" * 60)
    print("Deflated Sharpe Ratio - Examples")
    print("Based on Lopez de Prado (2018)")
    print("=" * 60)
    
    example_single_strategy()
    example_compare_strategies()
    example_probabilistic_sharpe()
    example_track_record_length()
    example_false_discovery()
    plot_dsr_vs_sr()
    
    print("\n" + "=" * 60)
    print("Key Takeaways:")
    print("=" * 60)
    print("1. DSR adjusts SR for multiple testing and non-normality")
    print("2. The more strategies you test, the higher the bar for significance")
    print("3. PSR gives probability that strategy beats a benchmark")
    print("4. Minimum track record length prevents false discoveries")
    print("5. FDR control balances false positives vs missed opportunities")
    print("=" * 60)
