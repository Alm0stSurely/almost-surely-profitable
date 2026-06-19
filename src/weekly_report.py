#!/usr/bin/env python3
"""
Weekly report generator with performance metrics.
Run every Friday after market close to generate weekly performance report.
"""

import sys
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reporting import ReportGenerator
from portfolio.portfolio import Portfolio
from data.fetch_market_data import fetch_current_prices, fetch_historical_data
from risk.performance_metrics import calculate_all_metrics, PerformanceMetrics
from risk.cvar import calculate_portfolio_cvar, tail_risk_analysis


def calculate_weekly_returns(week_results):
    """Calculate daily returns from week results."""
    returns = []
    for i in range(1, len(week_results)):
        prev_value = week_results[i-1].get('portfolio_after', {}).get('total_value', 0)
        curr_value = week_results[i].get('portfolio_after', {}).get('total_value', 0)
        if prev_value > 0:
            daily_return = (curr_value - prev_value) / prev_value
            returns.append(daily_return)
    return np.array(returns)


def fetch_benchmark_returns(start_date, end_date, benchmarks=None):
    """Fetch benchmark returns for comparison.
    
    Args:
        start_date: Start date string
        end_date: End date string
        benchmarks: List of benchmark tickers, or single string. Defaults to ['SPY', 'CAC.PA'].
    
    Returns:
        Dict mapping benchmark ticker to returns array, or None for failed fetches.
    """
    if benchmarks is None:
        benchmarks = ['SPY', 'CAC.PA']
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]
    
    result = {}
    try:
        data = fetch_historical_data(benchmarks, period="1mo")
        for benchmark in benchmarks:
            if benchmark in data:
                df = data[benchmark]
                if 'Close' in df.columns and len(df) >= 2:
                    closes = df['Close'].values
                    returns = np.diff(closes) / closes[:-1]
                    result[benchmark] = returns
                else:
                    print(f"  ⚠ Not enough data for benchmark {benchmark}")
            else:
                print(f"  ⚠ Could not fetch benchmark {benchmark}")
    except Exception as e:
        print(f"  ⚠ Could not fetch benchmarks: {e}")
    
    return result if result else None


def generate_weekly_report():
    """Generate and save weekly performance report."""
    print("="*70)
    print(f"WEEKLY REPORT — {datetime.now().strftime('%Y-%m-%d')}")
    print("="*70)
    
    # Initialize report generator
    rg = ReportGenerator()
    
    # Load portfolio for current state
    portfolio = Portfolio(data_dir="data")
    
    # Update with current prices
    if portfolio.positions:
        current_prices = fetch_current_prices(list(portfolio.positions.keys()))
        portfolio.update_prices(current_prices)
    
    # Load this week's daily results
    today = datetime.now()
    week_start = today - timedelta(days=today.weekday())  # Monday
    
    week_results = rg.load_daily_results(
        start_date=week_start.strftime('%Y-%m-%d'),
        end_date=today.strftime('%Y-%m-%d')
    )
    
    print(f"\n📊 Week of {week_start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}")
    print(f"   Trading days: {len(week_results)}")
    
    # Portfolio summary
    summary = portfolio.get_summary()
    print(f"\n💰 Portfolio Summary:")
    print(f"   Cash: €{summary['cash']:.2f}")
    print(f"   Positions Value: €{summary['positions_value']:.2f}")
    print(f"   Total Value: €{summary['total_value']:.2f}")
    print(f"   Total Return: {summary['total_return_pct']:.2f}%")
    print(f"   Realized P&L: €{summary['total_realized_pnl']:.2f}")
    print(f"   Unrealized P&L: €{summary['total_unrealized_pnl']:.2f}")
    
    # Weekly performance
    start_value = week_results[0].get('portfolio_before', {}).get('total_value', summary['total_value']) if week_results else summary['total_value']
    end_value = summary['total_value']
    weekly_return = ((end_value - start_value) / start_value * 100) if start_value else 0
    
    if len(week_results) >= 1:
        print(f"\n📈 Weekly Performance:")
        print(f"   Start of week: €{start_value:.2f}")
        print(f"   End of week: €{end_value:.2f}")
        print(f"   Weekly return: {weekly_return:+.2f}%")
    
    # Calculate performance metrics
    portfolio_returns = calculate_weekly_returns(week_results)
    benchmark_returns = fetch_benchmark_returns(
        week_start.strftime('%Y-%m-%d'),
        today.strftime('%Y-%m-%d')
    )
    
    if len(portfolio_returns) >= 2:
        print(f"\n📊 Performance Metrics (Week):")
        
        # Calculate metrics against each benchmark
        spy_returns = benchmark_returns.get('SPY') if benchmark_returns else None
        metrics = calculate_all_metrics(portfolio_returns, spy_returns)
        
        print(f"   Sharpe Ratio: {metrics.sharpe_ratio:.2f}")
        print(f"   Sortino Ratio: {metrics.sortino_ratio:.2f}")
        
        if metrics.beta is not None:
            print(f"   Beta (vs SPY): {metrics.beta:.2f}")
            print(f"   Alpha (vs SPY): {metrics.alpha:.2%}")
        
        print(f"   Max Drawdown: {metrics.max_drawdown:.2%}")
        print(f"   Volatility: {metrics.volatility:.2%}")
        
        if metrics.information_ratio is not None:
            print(f"   Information Ratio (vs SPY): {metrics.information_ratio:.2f}")
        
        # Compare against CAC.PA if available
        cac_returns = benchmark_returns.get('CAC.PA') if benchmark_returns else None
        if cac_returns is not None and len(cac_returns) == len(portfolio_returns):
            cac_metrics = calculate_all_metrics(portfolio_returns, cac_returns)
            if cac_metrics.beta is not None:
                print(f"   Beta (vs CAC.PA): {cac_metrics.beta:.2f}")
                print(f"   Alpha (vs CAC.PA): {cac_metrics.alpha:.2%}")
        elif cac_returns is not None:
            print(f"   ⚠ CAC.PA data length mismatch, skipping comparison")
        
        # Risk metrics (CVaR)
        cvar_result = calculate_portfolio_cvar(
            {'portfolio': portfolio_returns},
            {'portfolio': 1.0}
        )
        print(f"   CVaR 95%: {cvar_result.cvar_95:.2%}")
        print(f"   VaR 95%: {cvar_result.var_95:.2%}")
        
        # Tail risk analysis (vs SPY as primary benchmark)
        tail = tail_risk_analysis(portfolio_returns, spy_returns)
        print(f"   Skewness: {tail.get('skewness', 0):.2f}")
        print(f"   Kurtosis: {tail.get('kurtosis', 0):.2f}")
    
    # Positions
    if summary['positions']:
        print(f"\n📋 Current Positions:")
        for pos in summary['positions']:
            print(f"   {pos['ticker']}: {pos['quantity']:.2f} shares @ €{pos['current_price']:.2f}")
            print(f"      Value: €{pos['market_value']:.2f} | P&L: {pos['unrealized_pnl_pct']:+.2f}%")
    
    # Trades this week
    all_trades = []
    for day_result in week_results:
        trades = day_result.get('executed_trades', [])
        for trade in trades:
            if trade.get('status') == 'executed':
                all_trades.append({
                    'date': day_result.get('date', 'unknown'),
                    **trade
                })
    
    if all_trades:
        print(f"\n🔄 Trades This Week ({len(all_trades)}):")
        for trade in all_trades:
            print(f"   {trade['date']}: {trade['action'].upper()} {trade['ticker']} @ €{trade['price']:.2f}")
            
            # Calculate P&L if sell
            if trade['action'] == 'sell':
                # Find original buy price
                realized = trade.get('realized_pnl', 0)
                print(f"      Realized P&L: €{realized:.2f}")
    else:
        print(f"\n🔄 No trades this week")
    
    # Save report
    report_file = f"results/weekly-{today.strftime('%G-W%V')}.md"
    Path("results").mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(f"# Weekly Report — Week {today.strftime('%G-W%V')}\n\n")
        f.write(f"**Period:** {week_start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}\n\n")
        
        # Portfolio Summary
        f.write(f"## Portfolio Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Cash | €{summary['cash']:.2f} |\n")
        f.write(f"| Positions Value | €{summary['positions_value']:.2f} |\n")
        f.write(f"| **Total Value** | **€{summary['total_value']:.2f}** |\n")
        f.write(f"| Total Return | {summary['total_return_pct']:.2f}% |\n")
        f.write(f"| Realized P&L | €{summary['total_realized_pnl']:.2f} |\n")
        f.write(f"| Unrealized P&L | €{summary['total_unrealized_pnl']:.2f} |\n")
        f.write(f"| Number of Positions | {len(summary['positions'])} |\n")
        
        # Weekly Performance
        f.write(f"\n## Weekly Performance\n\n")
        if len(week_results) >= 1:
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| Start of Week | €{start_value:.2f} |\n")
            f.write(f"| End of Week | €{end_value:.2f} |\n")
            f.write(f"| Weekly Return | {weekly_return:+.2f}% |\n")
            f.write(f"| Trading Days | {len(week_results)} |\n")
        
        # Performance Metrics
        if len(portfolio_returns) >= 2:
            f.write(f"\n## Performance Metrics\n\n")
            f.write(f"| Metric | Value | Interpretation |\n")
            f.write(f"|--------|-------|----------------|\n")
            f.write(f"| Sharpe Ratio | {metrics.sharpe_ratio:.2f} | {'Good' if metrics.sharpe_ratio > 1 else 'Poor'} |\n")
            f.write(f"| Sortino Ratio | {metrics.sortino_ratio:.2f} | {'Good' if metrics.sortino_ratio > 1 else 'Poor'} |\n")
            
            if metrics.beta is not None:
                beta_interp = "Neutral" if 0.9 < metrics.beta < 1.1 else ("Defensive" if metrics.beta < 0.9 else "Aggressive")
                f.write(f"| Beta (vs SPY) | {metrics.beta:.2f} | {beta_interp} |\n")
                alpha_interp = "Outperform" if metrics.alpha and metrics.alpha > 0 else "Underperform"
                f.write(f"| Alpha | {metrics.alpha:.2%} | {alpha_interp} |\n")
            
            f.write(f"| Max Drawdown | {metrics.max_drawdown:.2%} | {'High' if metrics.max_drawdown < -0.1 else 'Moderate' if metrics.max_drawdown < -0.05 else 'Low'} |\n")
            f.write(f"| Volatility | {metrics.volatility:.2%} | {'High' if metrics.volatility > 0.3 else 'Moderate' if metrics.volatility > 0.15 else 'Low'} |\n")
            
            if metrics.information_ratio is not None:
                ir_interp = "Good" if metrics.information_ratio > 0.5 else "Neutral"
                f.write(f"| Information Ratio | {metrics.information_ratio:.2f} | {ir_interp} |\n")
            
            # Risk metrics
            f.write(f"\n## Risk Metrics\n\n")
            f.write(f"| Metric | Value |\n")
            f.write(f"|--------|-------|\n")
            f.write(f"| CVaR 95% | {cvar_result.cvar_95:.2%} |\n")
            f.write(f"| VaR 95% | {cvar_result.var_95:.2%} |\n")
            f.write(f"| Skewness | {tail.get('skewness', 0):.2f} |\n")
            f.write(f"| Kurtosis | {tail.get('kurtosis', 0):.2f} |\n")
            
            # CAC.PA comparison
            if cac_returns is not None and len(cac_returns) == len(portfolio_returns):
                f.write(f"\n## Benchmark Comparison: CAC 40\n\n")
                f.write(f"| Metric | Value | Interpretation |\n")
                f.write(f"|--------|-------|----------------|\n")
                if cac_metrics.beta is not None:
                    cac_beta_interp = "Neutral" if 0.9 < cac_metrics.beta < 1.1 else ("Defensive" if cac_metrics.beta < 0.9 else "Aggressive")
                    f.write(f"| Beta (vs CAC.PA) | {cac_metrics.beta:.2f} | {cac_beta_interp} |\n")
                    cac_alpha_interp = "Outperform" if cac_metrics.alpha and cac_metrics.alpha > 0 else "Underperform"
                    f.write(f"| Alpha (vs CAC.PA) | {cac_metrics.alpha:.2%} | {cac_alpha_interp} |\n")
                if cac_metrics.information_ratio is not None:
                    cac_ir_interp = "Good" if cac_metrics.information_ratio > 0.5 else "Neutral"
                    f.write(f"| Information Ratio (vs CAC.PA) | {cac_metrics.information_ratio:.2f} | {cac_ir_interp} |\n")
        
        # Positions
        f.write(f"\n## Positions\n\n")
        f.write(f"| Ticker | Quantity | Price | Value | P&L % | P&L € |\n")
        f.write(f"|--------|----------|-------|-------|-------|-------|\n")
        for pos in summary['positions']:
            f.write(f"| {pos['ticker']} | {pos['quantity']:.2f} | €{pos['current_price']:.2f} | ")
            f.write(f"€{pos['market_value']:.2f} | {pos['unrealized_pnl_pct']:+.2f}% | €{pos['unrealized_pnl']:+.2f} |\n")
        
        # Trades
        f.write(f"\n## Trades This Week\n\n")
        if all_trades:
            f.write(f"| Date | Action | Ticker | Price | P&L |\n")
            f.write(f"|------|--------|--------|-------|-----|\n")
            for trade in all_trades:
                pnl_str = f"€{trade.get('realized_pnl', 0):.2f}" if trade['action'] == 'sell' else "—"
                f.write(f"| {trade['date']} | {trade['action'].upper()} | {trade['ticker']} | €{trade['price']:.2f} | {pnl_str} |\n")
        else:
            f.write(f"No trades executed this week.\n")
    
    print(f"\n✅ Report saved to {report_file}")
    print("="*70)


if __name__ == "__main__":
    try:
        generate_weekly_report()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
