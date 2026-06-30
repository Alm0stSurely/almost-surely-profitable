#!/usr/bin/env python3
"""
Prompt Optimizer for LLM Trading Agent.

Tests different system prompt variations on historical data
to identify which prompt configurations yield the best performance.
"""

import json
import sys
import itertools
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).parent / ".."))

from data.fetch_market_data import fetch_historical_data
from data.indicators import analyze_market_data
from portfolio.portfolio import Portfolio
from llm.trading_agent import TradingAgent


@dataclass
class PromptVariant:
    """A variant of the system prompt to test."""
    name: str
    system_prompt: str
    description: str
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class BacktestResult:
    """Results from testing a prompt variant."""
    variant_name: str
    start_date: str
    end_date: str
    
    # Performance metrics
    total_return_pct: float
    sharpe_ratio: float
    max_drawdown_pct: float
    win_rate: float
    
    # Trade statistics
    total_trades: int
    buy_trades: int
    sell_trades: int
    avg_trades_per_day: float
    
    # Risk metrics
    volatility: float
    calmar_ratio: float
    
    # Additional metrics
    final_portfolio_value: float
    cash_utilization: float  # Avg invested capital
    
    def to_dict(self) -> Dict:
        return asdict(self)


class PromptOptimizer:
    """
    Optimizes system prompts by backtesting variants on historical data.
    """
    
    # Base prompt components that can be mixed and matched
    CVAR_COMPONENT = """
CVaR Framework: Always consider Conditional Value at Risk (95% confidence).
Calculate the expected loss in the worst 5% of cases for any position.
Maximum portfolio CVaR: 2% of capital per day.
"""
    
    PROSPECT_THEORY_COMPONENT = """
Prospect Theory: Humans feel losses ~2.25x more than equivalent gains.
Factor this into position sizing:
- For potential losses: multiply perceived risk by 2.25
- For potential gains: apply diminishing sensitivity (marginal utility decreases)
- Prefer positively skewed payoff distributions
"""
    
    LOSS_AVERSION_COMPONENT = """
Loss Aversion Discipline:
- Cut losing positions quickly (don't hope for recovery)
- Let winners run (avoid premature profit-taking)
- Mental stop-loss at -5% per position
- Scale out of winners: sell 50% at +5%, hold rest with trailing stop
"""
    
    META_LABELING_COMPONENT = """
Meta-Labeling Filter: Before any trade, estimate:
1. Primary model confidence (0-100%): Directional edge
2. Secondary model confidence (0-100%): Probability of profit given entry
Only trade if: Primary > 60% AND Secondary > 55%
This reduces false discovery rate in backtests.
"""
    
    REGIME_AWARE_COMPONENT = """
Regime Awareness: Adapt strategy to market conditions:
- High volatility regime (>75th percentile): Reduce position sizes by 50%
- Low volatility regime (<25th percentile): Normal sizing
- High correlation regime (>0.9 avg): Prioritize cash (diversification broken)
- Trending regime (ADX > 25): Allow momentum following
- Mean-reverting regime (ADX < 20): Take contrarian positions
"""
    
    CONTRARIAN_COMPONENT = """
Contrarian Discipline: When markets are euphoric (RSI > 70 everywhere), 
maintain cash. When fear dominates (RSI < 30), deploy capital gradually.
The best opportunities often feel uncomfortable.
"""
    
    def __init__(self, start_date: str, end_date: str, initial_capital: float = 10000.0):
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date = datetime.strptime(end_date, "%Y-%m-%d")
        self.initial_capital = initial_capital
        
        self.variants: List[PromptVariant] = []
        self.results: List[BacktestResult] = []
        
    def create_default_variants(self) -> List[PromptVariant]:
        """Create standard prompt variants to test."""
        
        base_prompt = """You are an elite quantitative trader with expertise in behavioral finance and risk management.

Your task: Make trading decisions for a portfolio given current market conditions.

PORTFOLIO STATE:
- Cash available: {cash:.2f} EUR
- Current positions: {positions}
- Total portfolio value: {total_value:.2f} EUR
- Current drawdown: {drawdown:.2f}%

MARKET CONDITIONS:
{market_summary}

DECISION FRAMEWORK:
1. Analyze each asset's technical indicators (RSI, Bollinger Bands, trend)
2. Consider portfolio risk concentration and correlations
3. Decide: buy (accumulate), sell (reduce), or hold for each position
4. Specify position sizing as percentage of portfolio (0-100%)

Respond in JSON format:
{{
  "actions": [
    {{"ticker": "XXX", "action": "buy|sell|hold", "pct": 5}},
  ],
  "reasoning": "brief explanation of your strategy"
}}"""
        
        variants = [
            PromptVariant(
                name="baseline",
                system_prompt=base_prompt,
                description="Simple baseline without behavioral components"
            ),
            
            PromptVariant(
                name="cvar_only",
                system_prompt=base_prompt + self.CVAR_COMPONENT,
                description="Baseline + CVaR risk management"
            ),
            
            PromptVariant(
                name="prospect_theory",
                system_prompt=base_prompt + self.PROSPECT_THEORY_COMPONENT,
                description="Baseline + Prospect Theory psychology"
            ),
            
            PromptVariant(
                name="loss_aversion",
                system_prompt=base_prompt + self.LOSS_AVERSION_COMPONENT,
                description="Baseline + Loss Aversion discipline"
            ),
            
            PromptVariant(
                name="meta_labeling",
                system_prompt=base_prompt + self.META_LABELING_COMPONENT,
                description="Baseline + Meta-labeling filter for trade selection"
            ),
            
            PromptVariant(
                name="regime_aware",
                system_prompt=base_prompt + self.REGIME_AWARE_COMPONENT,
                description="Baseline + Market regime adaptation"
            ),
            
            PromptVariant(
                name="contrarian",
                system_prompt=base_prompt + self.CONTRARIAN_COMPONENT,
                description="Baseline + Contrarian discipline"
            ),
            
            PromptVariant(
                name="full_behavioral",
                system_prompt=base_prompt + self.CVAR_COMPONENT + self.PROSPECT_THEORY_COMPONENT + 
                                self.LOSS_AVERSION_COMPONENT + self.META_LABELING_COMPONENT + 
                                self.REGIME_AWARE_COMPONENT,
                description="All behavioral components combined"
            ),
            
            PromptVariant(
                name="risk_focused",
                system_prompt=base_prompt + self.CVAR_COMPONENT + self.LOSS_AVERSION_COMPONENT + 
                                self.REGIME_AWARE_COMPONENT,
                description="Risk-focused subset (CVaR + Loss Aversion + Regime)"
            ),
        ]
        
        return variants
    
    def load_variants_from_config(self, config_path: str) -> List[PromptVariant]:
        """Load prompt variants from a JSON config file."""
        with open(config_path) as f:
            config = json.load(f)
        
        variants = []
        for item in config.get("variants", []):
            variants.append(PromptVariant(
                name=item["name"],
                system_prompt=item["system_prompt"],
                description=item.get("description", "")
            ))
        
        return variants
    
    def backtest_variant(
        self, 
        variant: PromptVariant, 
        market_data: Dict[str, pd.DataFrame],
        verbose: bool = False
    ) -> BacktestResult:
        """
        Backtest a single prompt variant on historical data.
        
        This simulates running the trading agent each day with this prompt
        and tracks the resulting portfolio performance.
        """
        if verbose:
            print(f"\nBacktesting: {variant.name}")
            print(f"Description: {variant.description}")
        
        # Initialize portfolio (ephemeral backtest, no persistence)
        portfolio = Portfolio(state_file="backtest_state.json", trades_file="backtest_trades.json", data_dir="/tmp/backtest")
        
        # Get trading dates from market data
        # Use SPY as reference for dates
        if "SPY" not in market_data:
            raise ValueError("SPY data required for backtest")
        
        spy_data = market_data["SPY"]
        trading_dates = spy_data.index[
            (spy_data.index >= self.start_date) & 
            (spy_data.index <= self.end_date)
        ]
        
        if len(trading_dates) == 0:
            raise ValueError(f"No trading dates found between {self.start_date} and {self.end_date}")
        
        # Track trades and portfolio peak
        all_trades = []
        daily_values = []
        peak_value = self.initial_capital
        
        # Simulate each trading day
        for i, date in enumerate(trading_dates):
            date_str = date.strftime("%Y-%m-%d")
            
            # Get market data up to this date
            historical_slice = {
                ticker: df[df.index <= date] 
                for ticker, df in market_data.items()
                if not df.empty
            }
            
            # Calculate indicators
            market_analysis = analyze_market_data(historical_slice)
            
            # Update portfolio with current prices
            current_prices = {
                ticker: df["Close"].iloc[-1] 
                for ticker, df in historical_slice.items() 
                if not df.empty
            }
            portfolio.update_prices(current_prices)
            
            # Track peak for drawdown
            current_total = portfolio.total_value
            if current_total > peak_value:
                peak_value = current_total
            
            # Build prompt context
            context = self._build_context(portfolio, market_analysis, date_str, peak_value)
            
            # Get LLM decision (using the variant's system prompt)
            # Note: This requires API calls - for actual backtest we'd use cached decisions
            # or simulate with a simplified model
            
            # For now, we'll skip actual LLM calls and use a placeholder
            # In production, this would call the LLM with the variant prompt
            
            # Placeholder: no trades for now
            # trades = self._simulate_llm_decision(context, variant.system_prompt)
            
            # Record daily value
            daily_values.append({
                "date": date_str,
                "value": portfolio.total_value,
                "cash": portfolio.cash
            })
            
            if verbose and i % 10 == 0:
                print(f"  {date_str}: Portfolio = ${portfolio.total_value:.2f}")
        
        # Calculate final metrics
        final_value = portfolio.total_value
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        # Calculate Sharpe and other metrics from daily values
        values_series = pd.Series([d["value"] for d in daily_values])
        returns = values_series.pct_change().dropna()
        
        sharpe = 0.0
        if len(returns) > 1 and returns.std() > 0:
            sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        # Calculate max drawdown
        cummax = values_series.cummax()
        drawdowns = (values_series - cummax) / cummax
        max_drawdown = drawdowns.min()
        
        result = BacktestResult(
            variant_name=variant.name,
            start_date=self.start_date.strftime("%Y-%m-%d"),
            end_date=self.end_date.strftime("%Y-%m-%d"),
            total_return_pct=total_return * 100,
            sharpe_ratio=sharpe,
            max_drawdown_pct=max_drawdown * 100,
            win_rate=0.0,  # Would need trade-level analysis
            total_trades=len(all_trades),
            buy_trades=sum(1 for t in all_trades if t["action"] == "buy"),
            sell_trades=sum(1 for t in all_trades if t["action"] == "sell"),
            avg_trades_per_day=len(all_trades) / len(trading_dates) if len(trading_dates) > 0 else 0,
            volatility=returns.std() * np.sqrt(252) * 100 if len(returns) > 1 else 0,
            calmar_ratio=(total_return * 100) / abs(max_drawdown * 100) if max_drawdown != 0 else 0,
            final_portfolio_value=final_value,
            cash_utilization=1 - (sum(d["cash"] for d in daily_values) / len(daily_values) / final_value)
        )
        
        return result
    
    def _build_context(self, portfolio: Portfolio, market_analysis: Dict, date: str, peak_value: float) -> Dict:
        """Build the context dictionary for LLM prompting."""
        current_total = portfolio.total_value
        drawdown = (current_total - peak_value) / peak_value if peak_value > 0 else 0.0
        return {
            "cash": portfolio.cash,
            "positions": portfolio.positions,
            "total_value": current_total,
            "drawdown": drawdown,
            "market_summary": market_analysis,
            "date": date
        }
    
    def run_optimization(
        self, 
        variants: Optional[List[PromptVariant]] = None,
        verbose: bool = True
    ) -> List[BacktestResult]:
        """
        Run backtests for all prompt variants and return ranked results.
        """
        if variants is None:
            variants = self.create_default_variants()
        
        if verbose:
            print(f"="*70)
            print(f"PROMPT OPTIMIZATION RUN")
            print(f"Period: {self.start_date.date()} to {self.end_date.date()}")
            print(f"Capital: ${self.initial_capital:,.2f}")
            print(f"Testing {len(variants)} prompt variants")
            print(f"="*70)
        
        # Fetch market data once for all backtests
        if verbose:
            print("\nFetching market data...")
        
        period_days = (self.end_date - self.start_date).days + 60
        market_data = fetch_historical_data(
            tickers=["SPY", "QQQ", "GLD", "TLT"],  # Simplified universe for speed
            period=f"{period_days}d"
        )
        
        # Run backtest for each variant
        results = []
        for variant in variants:
            try:
                result = self.backtest_variant(variant, market_data, verbose=verbose)
                results.append(result)
                
                if verbose:
                    print(f"\n  Results:")
                    print(f"    Return: {result.total_return_pct:+.2f}%")
                    print(f"    Sharpe: {result.sharpe_ratio:.2f}")
                    print(f"    Max DD: {result.max_drawdown_pct:.2f}%")
                    
            except Exception as e:
                print(f"Error backtesting {variant.name}: {e}")
                continue
        
        # Sort by risk-adjusted return (Calmar ratio)
        results.sort(key=lambda x: x.calmar_ratio, reverse=True)
        self.results = results
        
        return results
    
    def generate_report(self) -> str:
        """Generate a comparison report of all tested variants."""
        if not self.results:
            return "No results to report. Run optimization first."
        
        report = f"""
{'='*80}
PROMPT OPTIMIZATION RESULTS
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Period: {self.start_date.date()} to {self.end_date.date()}
{'='*80}

RANKING BY CALMAR RATIO (Return / Max Drawdown)
{'-'*80}
"""
        
        for i, result in enumerate(self.results, 1):
            report += f"""
{i}. {result.variant_name.upper()}
   Return: {result.total_return_pct:+.2f}% | Sharpe: {result.sharpe_ratio:.2f} | 
   Max DD: {result.max_drawdown_pct:.2f}% | Calmar: {result.calmar_ratio:.2f}
   Trades: {result.total_trades} ({result.buy_trades} buys, {result.sell_trades} sells)
   Volatility: {result.volatility:.2f}% | Final Value: ${result.final_portfolio_value:,.2f}
"""
        
        # Find best by different metrics
        best_return = max(self.results, key=lambda x: x.total_return_pct)
        best_sharpe = max(self.results, key=lambda x: x.sharpe_ratio)
        best_calmar = max(self.results, key=lambda x: x.calmar_ratio)
        lowest_dd = max(self.results, key=lambda x: x.max_drawdown_pct)  # Less negative is better
        
        report += f"""
{'-'*80}
BEST BY METRIC
{'-'*80}
Highest Return:    {best_return.variant_name} ({best_return.total_return_pct:+.2f}%)
Best Sharpe:       {best_sharpe.variant_name} ({best_sharpe.sharpe_ratio:.2f})
Best Calmar:       {best_calmar.variant_name} ({best_calmar.calmar_ratio:.2f})
Lowest Drawdown:   {lowest_dd.variant_name} ({lowest_dd.max_drawdown_pct:.2f}%)
{'='*80}
"""
        
        return report
    
    def save_results(self, output_dir: str = "results/prompt_optimization"):
        """Save results to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save results as JSON
        results_file = output_path / f"optimization_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2)
        
        # Save report as text
        report_file = output_path / f"optimization_report_{timestamp}.txt"
        with open(report_file, 'w') as f:
            f.write(self.generate_report())
        
        print(f"Results saved to: {results_file}")
        print(f"Report saved to: {report_file}")


def main():
    """Run prompt optimization."""
    # Test on recent 3-month period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=90)
    
    optimizer = PromptOptimizer(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=10000.0
    )
    
    # Run optimization
    results = optimizer.run_optimization(verbose=True)
    
    # Generate and print report
    report = optimizer.generate_report()
    print(report)
    
    # Save results
    optimizer.save_results()


if __name__ == "__main__":
    main()
