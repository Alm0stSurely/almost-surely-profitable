"""
Visualization script for backtest results.
Generates equity curves, drawdown charts, and performance comparisons.
"""

import json
import sys
from pathlib import Path
from typing import Dict

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from datetime import datetime
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    mdates = None


def check_matplotlib():
    """Check if matplotlib is available."""
    if not HAS_MATPLOTLIB:
        raise ImportError("matplotlib required for visualization. Install with: pip install matplotlib")


def load_backtest_results(filepath: str) -> Dict:
    """Load backtest results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def plot_equity_curves(results: Dict, output_path: str = "results/backtest_equity.png"):
    """Plot equity curves for all strategies."""
    check_matplotlib()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    colors = {'buy_and_hold': '#2196F3', 'equal_weight': '#4CAF50', 'llm': '#FF9800'}
    
    for strategy_name, result in results.items():
        if not result or 'daily_results' not in result:
            continue
        
        dates = [datetime.strptime(r['date'], '%Y-%m-%d') for r in result['daily_results']]
        values = [r['total_value'] for r in result['daily_results']]
        
        color = colors.get(strategy_name, '#757575')
        ax1.plot(dates, values, label=strategy_name.replace('_', ' ').title(), 
                color=color, linewidth=1.5)
    
    ax1.set_title('Portfolio Equity Curves', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Date')
    ax1.set_ylabel('Portfolio Value (€)')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Plot drawdowns
    for strategy_name, result in results.items():
        if not result or 'drawdown_curve' not in result:
            continue
        
        dates = [datetime.strptime(r['date'], '%Y-%m-%d') for r in result['daily_results']]
        drawdowns = [d * 100 for d in result['drawdown_curve']]  # Convert to percentage
        
        color = colors.get(strategy_name, '#757575')
        ax2.fill_between(dates, drawdowns, 0, alpha=0.3, color=color, label=strategy_name)
        ax2.plot(dates, drawdowns, color=color, linewidth=1)
    
    ax2.set_title('Drawdowns', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Drawdown (%)')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Equity curves saved to: {output_path}")
    plt.close()


def plot_metrics_comparison(results: Dict, output_path: str = "results/backtest_metrics.png"):
    """Plot metrics comparison bar chart."""
    check_matplotlib()
    strategies = list(results.keys())
    
    # Extract metrics
    total_returns = [results[s]['total_return'] * 100 if results[s] else 0 for s in strategies]
    sharpe_ratios = [results[s]['sharpe_ratio'] if results[s] else 0 for s in strategies]
    max_drawdowns = [results[s]['max_drawdown'] * 100 if results[s] else 0 for s in strategies]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    # Total Return
    axes[0].bar(strategies, total_returns, color=colors[:len(strategies)])
    axes[0].set_title('Total Return (%)', fontweight='bold')
    axes[0].set_ylabel('Return (%)')
    axes[0].grid(True, alpha=0.3, axis='y')
    axes[0].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Sharpe Ratio
    axes[1].bar(strategies, sharpe_ratios, color=colors[:len(strategies)])
    axes[1].set_title('Sharpe Ratio', fontweight='bold')
    axes[1].set_ylabel('Sharpe')
    axes[1].grid(True, alpha=0.3, axis='y')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    
    # Max Drawdown
    axes[2].bar(strategies, max_drawdowns, color=colors[:len(strategies)])
    axes[2].set_title('Max Drawdown (%)', fontweight='bold')
    axes[2].set_ylabel('Drawdown (%)')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    # Rotate x labels
    for ax in axes:
        ax.set_xticklabels([s.replace('_', '\n').title() for s in strategies], rotation=0, ha='center')
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Metrics comparison saved to: {output_path}")
    plt.close()


def print_summary_table(results: Dict):
    """Print a formatted summary table."""
    print("\n" + "="*100)
    print(f"{'Strategy':<20} {'Return':>10} {'Ann. Return':>12} {'Sharpe':>8} {'Max DD':>10} {'Trades':>8} {'Win Rate':>10}")
    print("="*100)
    
    for strategy_name, result in results.items():
        if not result:
            continue
        
        print(f"{strategy_name.replace('_', ' ').title():<20} "
              f"{result['total_return']*100:>9.2f}% "
              f"{result['annualized_return']*100:>11.2f}% "
              f"{result['sharpe_ratio']:>7.2f} "
              f"{result['max_drawdown']*100:>9.2f}% "
              f"{result['num_trades']:>7} "
              f"{result['win_rate']*100:>9.1f}%")
    
    print("="*100 + "\n")


def plot_backtest_results(result: Dict, output_path: str):
    """Plot backtest results (single strategy).
    
    Args:
        result: Backtest result dictionary
        output_path: Path to save the plot
    """
    check_matplotlib()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
    
    # Extract dates and values
    daily_results = result.get('daily_results', [])
    if not daily_results:
        print("No daily results to plot")
        return
    
    dates = [datetime.strptime(d['date'], '%Y-%m-%d') for d in daily_results]
    values = [d['total_value'] for d in daily_results]
    
    # Calculate drawdowns
    peak = values[0]
    drawdowns = []
    for value in values:
        if value > peak:
            peak = value
        drawdowns.append((peak - value) / peak * 100)
    
    # Plot equity curve
    ax1.plot(dates, values, linewidth=2, color='#2196F3', label='Portfolio Value')
    ax1.axhline(y=result['initial_capital'], color='gray', linestyle='--', alpha=0.5, label='Initial Capital')
    ax1.set_title(f'Backtest: {result["strategy"]} ({result["start_date"]} to {result["end_date"]})', 
                  fontsize=14, fontweight='bold')
    ax1.set_ylabel('Portfolio Value (€)', fontsize=12)
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # Plot drawdown
    ax2.fill_between(dates, drawdowns, 0, color='red', alpha=0.3, label='Drawdown %')
    ax2.plot(dates, drawdowns, color='red', linewidth=1)
    ax2.set_ylabel('Drawdown (%)', fontsize=12)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.legend(loc='lower left')
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize backtest results")
    parser.add_argument("--input", default="results/backtest_comparison.json", 
                       help="Input JSON file with backtest results")
    parser.add_argument("--output-dir", default="results", 
                       help="Output directory for charts")
    
    args = parser.parse_args()
    
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        print("Run backtest first: python -m src.backtest.backtest")
        sys.exit(1)
    
    print(f"Loading results from: {args.input}")
    results = load_backtest_results(args.input)
    
    print_summary_table(results)
    
    print("Generating charts...")
    plot_equity_curves(results, f"{args.output_dir}/backtest_equity.png")
    plot_metrics_comparison(results, f"{args.output_dir}/backtest_metrics.png")
    
    print("\nVisualization complete!")
