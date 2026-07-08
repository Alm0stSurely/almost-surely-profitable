#!/usr/bin/env python3
"""
CLI tool to run backtests for the trading agent.

Usage:
    python run_backtest.py --start 2024-01-01 --end 2024-03-31 --strategy llm
    python run_backtest.py --start 2024-01-01 --end 2024-12-31 --strategy buy_and_hold --tickers SPY QQQ GLD
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from backtest.backtest import BacktestEngine, print_backtest_report

try:
    from backtest.visualize import plot_backtest_results
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plot_backtest_results = None


def validate_date(date_str: str) -> str:
    """Validate date format."""
    try:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str
    except ValueError:
        raise argparse.ArgumentTypeError(f"Invalid date format: {date_str}. Use YYYY-MM-DD.")


def main():
    parser = argparse.ArgumentParser(
        description="Run backtest for trading strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # LLM strategy backtest
  python run_backtest.py --start 2024-01-01 --end 2024-03-31 --strategy llm

  # Buy and hold benchmark
  python run_backtest.py --start 2024-01-01 --end 2024-12-31 --strategy buy_and_hold

  # Equal weight with custom tickers
  python run_backtest.py --start 2024-06-01 --end 2024-12-31 \\
                         --strategy equal_weight \\
                         --tickers SPY QQQ GLD IWM TLT

  # Save results to file
  python run_backtest.py --start 2024-01-01 --end 2024-12-31 \\
                         --strategy llm --output results/backtest-2024.json
        """
    )
    
    parser.add_argument(
        "--start",
        type=validate_date,
        required=True,
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        type=validate_date,
        required=True,
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--strategy",
        type=str,
        choices=["buy_and_hold", "equal_weight", "llm", "random"],
        default="buy_and_hold",
        help="Trading strategy to test (default: buy_and_hold)"
    )
    parser.add_argument(
        "--tickers",
        nargs="+",
        default=["SPY", "QQQ", "GLD", "IWM", "TLT"],
        help="List of tickers to trade (default: SPY QQQ GLD IWM TLT)"
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="Initial capital (default: 10000)"
    )
    parser.add_argument(
        "--frequency",
        type=str,
        choices=["daily", "weekly"],
        default="daily",
        help="Rebalance frequency (default: daily)"
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="SPY",
        help="Benchmark ticker for comparison (default: SPY)"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results (JSON)"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate visualization plots"
    )
    parser.add_argument(
        "--cooldowns",
        action="store_true",
        help="Enable position cooldown guardrails (min_hold, flip_cooldown, weekly_cap)"
    )
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=5,
        help="Minimum holding period in days (default: 5)"
    )
    parser.add_argument(
        "--flip-cooldown-days",
        type=int,
        default=10,
        help="Flip cooldown in days (default: 10)"
    )
    parser.add_argument(
        "--max-trades-per-week",
        type=int,
        default=2,
        help="Maximum trades per week (default: 2)"
    )
    parser.add_argument(
        "--vol-regime",
        type=str,
        choices=["high", "normal", "low"],
        default="normal",
        help="Volatility regime for adaptive cooldown (default: normal)"
    )
    parser.add_argument(
        "--max-trades-high-vol",
        type=int,
        default=2,
        help="Weekly trade cap in high volatility regime (default: 2)"
    )
    parser.add_argument(
        "--max-trades-normal-vol",
        type=int,
        default=3,
        help="Weekly trade cap in normal volatility regime (default: 3)"
    )
    parser.add_argument(
        "--max-trades-low-vol",
        type=int,
        default=4,
        help="Weekly trade cap in low volatility regime (default: 4)"
    )
    parser.add_argument(
        "--stop-loss-high-vol",
        type=float,
        default=3.0,
        help="Stop-loss threshold in high volatility regime (default: 3.0%%)"
    )
    parser.add_argument(
        "--stop-loss-normal-vol",
        type=float,
        default=5.0,
        help="Stop-loss threshold in normal volatility regime (default: 5.0%%)"
    )
    parser.add_argument(
        "--stop-loss-low-vol",
        type=float,
        default=7.0,
        help="Stop-loss threshold in low volatility regime (default: 7.0%%)"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Run comparison of all strategies (buy_and_hold, equal_weight, random, llm)"
    )

    args = parser.parse_args()

    # Validate date range
    start_dt = datetime.strptime(args.start, "%Y-%m-%d")
    end_dt = datetime.strptime(args.end, "%Y-%m-%d")
    if start_dt >= end_dt:
        print("Error: Start date must be before end date")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"TRADING AGENT BACKTEST")
    print(f"{'='*70}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Strategy: {args.strategy}")
    print(f"Tickers: {', '.join(args.tickers)}")
    print(f"Initial Capital: €{args.capital:,.2f}")
    print(f"Frequency: {args.frequency}")
    print(f"Cooldowns: {'Enabled' if args.cooldowns else 'Disabled'}")
    print(f"{'='*70}\n")

    # Build cooldown config if enabled
    cooldown_config = None
    if args.cooldowns:
        from backtest.backtest_cooldown import CooldownConfig
        cooldown_config = CooldownConfig(
            min_hold_days=args.min_hold_days,
            flip_cooldown_days=args.flip_cooldown_days,
            max_trades_per_week=args.max_trades_per_week,
            current_vol_regime=args.vol_regime,
            max_trades_high_vol=args.max_trades_high_vol,
            max_trades_normal_vol=args.max_trades_normal_vol,
            max_trades_low_vol=args.max_trades_low_vol,
            stop_loss_high_vol=args.stop_loss_high_vol,
            stop_loss_normal_vol=args.stop_loss_normal_vol,
            stop_loss_low_vol=args.stop_loss_low_vol,
        )

    if getattr(args, 'compare', False):
        # Run comparison of all strategies
        results = {}
        for strategy in ["buy_and_hold", "equal_weight", "random", "llm"]:
            if strategy == "llm":
                print(f"\n⚠️  LLM strategy may take longer due to API calls...")
            
            engine = BacktestEngine(
                start_date=args.start,
                end_date=args.end,
                initial_capital=args.capital,
                tickers=args.tickers,
                rebalance_frequency=args.frequency,
                enable_cooldowns=args.cooldowns,
                cooldown_config=cooldown_config
            )
            
            kwargs = {
                "use_llm": (strategy == "llm"),
                "strategy": strategy,
                "benchmark": args.benchmark
            }
            if strategy == "random":
                kwargs["random_seed"] = 42
            
            result = engine.run_backtest(**kwargs)
            results[strategy] = result
            print_backtest_report(result, strategy)
        
        # Print comparison
        print(f"\n{'='*70}")
        print("STRATEGY COMPARISON")
        print(f"{'='*70}")
        print(f"{'Strategy':<15} {'Return':>10} {'Sharpe':>8} {'Max DD':>8} {'Final €':>12}")
        print("-" * 70)
        for strategy, result in results.items():
            if result:
                print(f"{strategy:<15} "
                      f"{result['total_return']*100:>9.2f}% "
                      f"{result['sharpe_ratio']:>8.2f} "
                      f"{result['max_drawdown']*100:>7.2f}% "
                      f"€{result['final_value']:>10,.2f}")
    else:
        # Run single strategy
        if args.strategy == "llm":
            print("⚠️  LLM strategy may take longer due to API calls...")
            print("   Consider using weekly frequency for faster testing.\n")
        
        engine = BacktestEngine(
            start_date=args.start,
            end_date=args.end,
            initial_capital=args.capital,
            tickers=args.tickers,
            rebalance_frequency=args.frequency,
            enable_cooldowns=args.cooldowns,
            cooldown_config=cooldown_config
        )
        
        kwargs = {
            "use_llm": (args.strategy == "llm"),
            "strategy": args.strategy,
            "benchmark": args.benchmark
        }
        if args.strategy == "random":
            kwargs["random_seed"] = 42
        
        result = engine.run_backtest(**kwargs)
        
        print_backtest_report(result, args.strategy)
        
        # Save results if requested
        if args.output:
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert numpy types to Python types for JSON serialization
            result_serializable = {}
            for key, value in result.items():
                if hasattr(value, 'tolist'):  # numpy array
                    result_serializable[key] = value.tolist()
                elif key in ['equity_curve', 'drawdown_curve', 'daily_returns', 'daily_results']:
                    result_serializable[key] = value
                else:
                    result_serializable[key] = value
            
            with open(output_path, 'w') as f:
                json.dump(result_serializable, f, indent=2, default=str)
            print(f"\n✓ Results saved to {output_path}")
        
        # Generate plots if requested
        if args.plot:
            if not HAS_MATPLOTLIB:
                print("⚠️  Matplotlib not installed. Install with: pip install matplotlib")
            else:
                try:
                    plot_path = Path(args.output).with_suffix('.png') if args.output else Path("results/backtest_plot.png")
                    plot_path.parent.mkdir(parents=True, exist_ok=True)
                    plot_backtest_results(result, str(plot_path))
                    print(f"✓ Plots saved to {plot_path}")
                except Exception as e:
                    print(f"⚠️  Could not generate plots: {e}")
    
    print(f"\n{'='*70}")
    print("BACKTEST COMPLETE")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
