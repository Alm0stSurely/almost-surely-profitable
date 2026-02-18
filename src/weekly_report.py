#!/usr/bin/env python3
"""
Weekly report generator.
Run every Friday after market close to generate weekly performance report.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from reporting import ReportGenerator
from portfolio.portfolio import Portfolio
from data.fetch_market_data import fetch_current_prices


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
    
    # Weekly performance (if we have start of week data)
    start_value = week_results[0].get('portfolio_before', {}).get('total_value', summary['total_value']) if week_results else summary['total_value']
    end_value = summary['total_value']
    weekly_return = ((end_value - start_value) / start_value * 100) if start_value else 0
    if len(week_results) >= 1:
        print(f"\n📈 Weekly Performance:")
        print(f"   Start of week: €{start_value:.2f}")
        print(f"   End of week: €{end_value:.2f}")
        print(f"   Weekly return: {weekly_return:+.2f}%")
    
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
    else:
        print(f"\n🔄 No trades this week")
    
    # Save report
    report_file = f"results/weekly-{today.strftime('%Y-W%W')}.md"
    Path("results").mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        f.write(f"# Weekly Report — Week {today.strftime('%Y-W%W')}\n\n")
        f.write(f"**Period:** {week_start.strftime('%Y-%m-%d')} to {today.strftime('%Y-%m-%d')}\n\n")
        f.write(f"## Portfolio Summary\n\n")
        f.write(f"| Metric | Value |\n")
        f.write(f"|--------|-------|\n")
        f.write(f"| Cash | €{summary['cash']:.2f} |\n")
        f.write(f"| Positions Value | €{summary['positions_value']:.2f} |\n")
        f.write(f"| **Total Value** | **€{summary['total_value']:.2f}** |\n")
        f.write(f"| Total Return | {summary['total_return_pct']:.2f}% |\n")
        f.write(f"| Realized P&L | €{summary['total_realized_pnl']:.2f} |\n")
        f.write(f"| Unrealized P&L | €{summary['total_unrealized_pnl']:.2f} |\n")
        f.write(f"\n## Weekly Performance\n\n")
        if len(week_results) >= 1:
            f.write(f"- Start of week: €{start_value:.2f}\n")
            f.write(f"- End of week: €{end_value:.2f}\n")
            f.write(f"- Weekly return: {weekly_return:+.2f}%\n")
        f.write(f"\n## Positions\n\n")
        for pos in summary['positions']:
            f.write(f"- **{pos['ticker']}**: {pos['quantity']:.2f} shares @ €{pos['current_price']:.2f}\n")
            f.write(f"  - Value: €{pos['market_value']:.2f}\n")
            f.write(f"  - P&L: {pos['unrealized_pnl_pct']:+.2f}% (€{pos['unrealized_pnl']:+.2f})\n")
    
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
