#!/usr/bin/env python3
"""
Comprehensive Trading System Evaluation.

Runs all analysis modules and generates a complete performance report:
- Portfolio performance metrics
- LLM decision quality analysis
- Risk metrics (CVaR, tail risk)
- Market regime analysis
- Behavioral pattern detection
"""

import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent / ".."))

from analysis.decision_analyzer import DecisionAnalyzer
from risk.performance_metrics import calculate_all_metrics
from risk.cvar import calculate_portfolio_cvar
from data.fetch_market_data import fetch_historical_data


def load_portfolio_data():
    """Load current portfolio state."""
    try:
        with open("data/portfolio_state.json") as f:
            return json.load(f)
    except FileNotFoundError:
        return None


def load_recent_results(days=30):
    """Load recent daily results for analysis."""
    results_dir = Path("results/daily")
    if not results_dir.exists():
        return []
    
    files = sorted(results_dir.glob("*.json"))[-days:]
    results = []
    
    for file in files:
        try:
            with open(file) as f:
                data = json.load(f)
                results.append(data)
        except:
            continue
    
    return results


def _get_benchmark_return(start_date: str, end_date: str, benchmark: str = "SPY") -> Optional[float]:
    """Fetch benchmark buy-and-hold return for a date range."""
    try:
        data = fetch_historical_data([benchmark], start=start_date, end=end_date)
        if benchmark in data and not data[benchmark].empty:
            closes = data[benchmark]["Close"].values.flatten()
            if len(closes) >= 2:
                start_price = float(closes[0])
                end_price = float(closes[-1])
                return (end_price / start_price) - 1
    except Exception:
        pass
    return None


def calculate_performance_trends(results):
    """Calculate performance trends over time."""
    if not results:
        return {}
    
    trends = {
        "portfolio_values": [],
        "daily_returns": [],
        "cash_levels": [],
        "position_counts": [],
    }
    
    for r in results:
        if "portfolio_after" in r:
            after = r["portfolio_after"]
            trends["portfolio_values"].append(after.get("total_value", 0))
            trends["cash_levels"].append(after.get("cash", 0))
            trends["position_counts"].append(after.get("num_positions", 0))
            
            # Calculate daily return if we have previous value
            if len(trends["portfolio_values"]) > 1:
                prev = trends["portfolio_values"][-2]
                curr = trends["portfolio_values"][-1]
                if prev > 0:
                    trends["daily_returns"].append((curr - prev) / prev)
    
    return trends


def generate_comprehensive_report():
    """Generate comprehensive evaluation report."""
    
    print("="*70)
    print("COMPREHENSIVE TRADING SYSTEM EVALUATION")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    # 1. Portfolio Status
    portfolio = load_portfolio_data()
    if portfolio:
        print("\n📊 PORTFOLIO STATUS")
        print("-" * 40)
        print(f"Total Value: €{portfolio['total_value']:,.2f}")
        print(f"Cash: €{portfolio['cash']:,.2f} ({portfolio['cash']/portfolio['total_value']*100:.1f}%)")
        print(f"Realized P&L: €{portfolio['total_realized_pnl']:,.2f}")
        print(f"Positions: {len(portfolio['positions'])}")
    
    # 2. Recent Performance Trends
    results = load_recent_results(days=30)
    trends = calculate_performance_trends(results)
    
    if trends.get("portfolio_values"):
        print("\n📈 PERFORMANCE TRENDS (30 days)")
        print("-" * 40)
        values = trends["portfolio_values"]
        if len(values) >= 2:
            change = (values[-1] - values[0]) / values[0] * 100
            print(f"Period Return: {change:+.2f}%")
            print(f"Highest Value: €{max(values):,.2f}")
            print(f"Lowest Value: €{min(values):,.2f}")
        
        if trends.get("daily_returns"):
            import numpy as np
            returns = trends["daily_returns"]
            print(f"Volatility (ann): {np.std(returns) * np.sqrt(252) * 100:.1f}%")
    
    # 3. LLM Decision Quality
    print("\n🤖 LLM DECISION QUALITY")
    print("-" * 40)
    
    analyzer = DecisionAnalyzer()
    decisions = analyzer.load_decisions(days=30)
    
    if decisions:
        print(f"Decisions Analyzed: {len(decisions)} days")
        
        # Quick stats
        total_trades = sum(len(d.get("trades", [])) for d in decisions)
        print(f"Total Trades: {total_trades}")
        print(f"Avg Trades/Day: {total_trades/len(decisions):.1f}")
        
        # Detailed outcomes
        outcomes = analyzer.analyze_outcomes(decisions)
        print(f"\nWin Rate: {outcomes['win_rate']*100:.1f}%")
        print(f"Buy Accuracy: {outcomes['buy_accuracy']*100:.1f}%")
        print(f"Sell Accuracy: {outcomes['sell_accuracy']*100:.1f}%")
    else:
        print("No decision data available yet.")
    
    # 4. Risk Assessment
    print("\n⚠️  RISK ASSESSMENT")
    print("-" * 40)
    
    if trends.get("daily_returns"):
        import numpy as np
        returns = np.array(trends["daily_returns"])
        
        # CVaR estimation
        var_95 = np.percentile(returns, 5)
        cvar_95 = returns[returns <= var_95].mean() if len(returns[returns <= var_95]) > 0 else var_95
        
        print(f"VaR (95%): {var_95*100:.2f}%")
        print(f"CVaR (95%): {cvar_95*100:.2f}%")
        print(f"Max Drawdown (est): {np.min(returns)*100:.2f}%")
    
    # 5. System Health
    print("\n🔧 SYSTEM HEALTH")
    print("-" * 40)
    
    # Check data freshness
    from data.fetch_market_data import fetch_current_prices
    try:
        test_prices = fetch_current_prices(["SPY"])
        if "SPY" in test_prices and test_prices["SPY"] > 0:
            print("✓ Data feed: Operational")
        else:
            print("⚠ Data feed: Issues detected")
    except Exception as e:
        print(f"✗ Data feed: Error - {e}")
    
    # Check file structure
    required_files = [
        "src/data/fetch_market_data.py",
        "src/portfolio/portfolio.py",
        "src/llm/trading_agent.py",
        "config/universe.json"
    ]
    
    missing = [f for f in required_files if not Path(f).exists()]
    if missing:
        print(f"⚠ Missing files: {', '.join(missing)}")
    else:
        print("✓ Core files: Present")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    if portfolio:
        total_return = (portfolio['total_value'] - 10000) / 10000 * 100
        print(f"Total Return: {total_return:+.2f}%")
        
        # Compare to benchmark if we have date range from results
        if results:
            dates = [r.get('date') for r in results if r.get('date')]
            if dates:
                bench_return = _get_benchmark_return(min(dates), max(dates))
                if bench_return is not None:
                    alpha = total_return - bench_return * 100
                    print(f"vs Buy & Hold (SPY): {'+' if alpha >= 0 else ''}{alpha:.2f}%")
    
    print("\n✓ Evaluation complete.")
    print("="*70)


def main():
    """Run comprehensive evaluation."""
    generate_comprehensive_report()
    
    # Save to file
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Redirect output to file
    import io
    old_stdout = sys.stdout
    sys.stdout = buffer = io.StringIO()
    
    generate_comprehensive_report()
    
    output = buffer.getvalue()
    sys.stdout = old_stdout
    
    # Save report
    report_file = output_dir / f"comprehensive_evaluation_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(report_file, 'w') as f:
        f.write(output)
    
    print(f"\n📄 Full report saved to: {report_file}")


if __name__ == "__main__":
    main()
