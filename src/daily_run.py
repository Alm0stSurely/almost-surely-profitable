#!/usr/bin/env python3
"""
Daily trading pipeline.
Orchestrates data fetching, indicator calculation, LLM decision, and order execution.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from data.fetch_market_data import fetch_historical_data, fetch_current_prices, ALL_TICKERS
from data.indicators import analyze_market_data
from portfolio.portfolio import Portfolio
from llm.trading_agent import TradingAgent
from risk.cvar import calculate_portfolio_cvar, tail_risk_analysis


def setup_directories():
    """Create necessary directories."""
    Path("data").mkdir(exist_ok=True)
    Path("results/daily").mkdir(parents=True, exist_ok=True)


def run_daily_pipeline(dry_run: bool = False):
    """
    Execute the complete daily trading pipeline.
    
    Args:
        dry_run: If True, don't execute actual trades (for testing)
    """
    print("="*70)
    print(f"DAILY TRADING RUN — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    setup_directories()
    
    # Step 1: Fetch market data
    print("\n[1/7] Fetching market data...")
    market_data_raw = fetch_historical_data(ALL_TICKERS, period="30d")
    print(f"  ✓ Fetched data for {len(market_data_raw)} assets")
    
    # Step 2: Calculate indicators
    print("\n[2/7] Calculating technical indicators...")
    market_analysis = analyze_market_data(market_data_raw)
    print(f"  ✓ Analysis complete for {len(market_analysis['assets'])} assets")
    
    # Step 3: Load portfolio
    print("\n[3/7] Loading portfolio...")
    portfolio = Portfolio(data_dir="data")
    
    # Update current prices in portfolio
    current_prices = fetch_current_prices(list(portfolio.positions.keys()))
    portfolio.update_prices(current_prices)
    portfolio.save_state()
    
    portfolio.print_summary()
    
    # Calculate portfolio risk metrics (CVaR)
    print("\n[3.5/7] Calculating risk metrics (CVaR)...")
    portfolio_summary = portfolio.get_summary()
    
    # Build position returns from historical data for CVaR calculation
    position_returns = {}
    portfolio_weights = {}
    total_value = portfolio_summary['total_value']
    
    for position in portfolio_summary.get('positions', []):
        ticker = position['ticker']
        if ticker in market_analysis['assets']:
            # Get historical returns for this position
            hist_data = market_analysis['assets'][ticker]
            if 'returns' in hist_data:
                position_returns[ticker] = np.array(hist_data['returns'])
                # Weight = position value / total portfolio value
                portfolio_weights[ticker] = position['market_value'] / total_value
    
    if position_returns and len(position_returns) > 0:
        cvar_result = calculate_portfolio_cvar(position_returns, portfolio_weights)
        tail_risk = tail_risk_analysis(
            np.array([sum(position_returns[t] * portfolio_weights[t] 
                         for t in position_returns if t in portfolio_weights)]),
            benchmark_returns=None
        )
        
        print(f"  CVaR 95%: {cvar_result.cvar_95:.2%}")
        print(f"  VaR 95%: {cvar_result.var_95:.2%}")
        print(f"  Max Drawdown: {tail_risk.get('max_drawdown', 0):.2%}")
        
        # Add risk metrics to portfolio summary
        portfolio_summary['risk_metrics'] = {
            'cvar_95': cvar_result.cvar_95,
            'cvar_99': cvar_result.cvar_99,
            'var_95': cvar_result.var_95,
            'var_99': cvar_result.var_99,
            'max_drawdown': tail_risk.get('max_drawdown', 0),
            'sortino_ratio': tail_risk.get('sortino_ratio', 0),
            'skewness': tail_risk.get('skewness', 0),
            'kurtosis': tail_risk.get('kurtosis', 0)
        }
    
    # Step 4: Get LLM decision
    print("\n[4/7] Getting trading decision from LLM...")
    agent = TradingAgent()
    portfolio_summary = portfolio.get_summary()
    
    decision = agent.get_trading_decision(market_analysis, portfolio_summary)
    
    print(f"\n  Reasoning: {decision['reasoning'][:200]}...")
    print(f"  Actions: {len(decision['actions'])}")
    
    # Step 5: Execute orders
    print("\n[5/7] Executing orders...")
    executed_trades = []
    
    if dry_run:
        print("  (DRY RUN - no actual trades executed)")
    
    for action in decision['actions']:
        ticker = action['ticker']
        action_type = action['action']
        pct = action.get('pct', 0)
        
        # Get current price
        current_price = current_prices.get(ticker)
        if not current_price:
            # Try to get from market data
            if ticker in market_analysis['assets']:
                current_price = market_analysis['assets'][ticker]['latest']['price']
        
        if not current_price:
            print(f"  ✗ {ticker}: No price available")
            continue
        
        if dry_run:
            print(f"  [DRY] {ticker}: {action_type} {pct}% @ €{current_price:.2f}")
            executed_trades.append({
                'ticker': ticker,
                'action': action_type,
                'pct': pct,
                'price': current_price,
                'status': 'dry_run'
            })
        else:
            if action_type == 'buy':
                success = portfolio.buy(ticker, pct, current_price)
                if success:
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'buy',
                        'pct': pct,
                        'price': current_price,
                        'status': 'executed'
                    })
            elif action_type == 'sell':
                success = portfolio.sell(ticker, current_price)
                if success:
                    executed_trades.append({
                        'ticker': ticker,
                        'action': 'sell',
                        'pct': pct,
                        'price': current_price,
                        'status': 'executed'
                    })
            elif action_type == 'hold':
                print(f"  • {ticker}: HOLD")
    
    # Step 6: Save portfolio state
    print("\n[6/7] Saving portfolio state...")
    portfolio.save_state()
    print("  ✓ State saved")
    
    # Step 7: Log results
    print("\n[7/7] Logging results...")
    
    result = {
        'date': datetime.now().strftime('%Y-%m-%d'),
        'timestamp': datetime.now().isoformat(),
        'dry_run': dry_run,
        'market_summary': {
            'assets_analyzed': len(market_analysis['assets']),
            'analysis_date': market_analysis['analysis_date']
        },
        'portfolio_before': {
            'cash': portfolio_summary['cash'],
            'total_value': portfolio_summary['total_value'],
            'positions': len(portfolio_summary['positions'])
        },
        'decision': {
            'reasoning': decision['reasoning'],
            'actions': decision['actions'],
            'error': decision.get('error', False)
        },
        'executed_trades': executed_trades,
        'portfolio_after': portfolio.get_summary() if not dry_run else portfolio_summary
    }
    
    # Save to daily results
    result_file = f"results/daily/{datetime.now().strftime('%Y-%m-%d')}.json"
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"  ✓ Results saved to {result_file}")
    
    # Final summary
    print("\n" + "="*70)
    print("DAILY RUN COMPLETE")
    print("="*70)
    print(f"Total Value: €{portfolio.get_summary()['total_value']:.2f}")
    print(f"Total Return: {portfolio.get_summary()['total_return_pct']:.2f}%")
    print(f"Trades Executed: {len([t for t in executed_trades if t.get('status') == 'executed'])}")
    print("="*70)
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Daily trading pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing actual trades"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode (minimal data)"
    )
    
    args = parser.parse_args()
    
    try:
        result = run_daily_pipeline(dry_run=args.dry_run)
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
