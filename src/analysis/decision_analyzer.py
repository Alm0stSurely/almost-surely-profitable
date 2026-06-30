#!/usr/bin/env python3
"""
LLM Decision Quality Analyzer - Enhanced Version.

Analyzes the quality of LLM trading decisions by comparing predictions
with actual market outcomes. Tracks hit rate, risk-adjusted returns,
and behavioral biases in LLM decisions.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

try:
    import numpy as np
    import pandas as pd
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    np = None
    pd = None

sys.path.insert(0, str(Path(__file__).parent / ".."))

from data.fetch_market_data import fetch_historical_data, fetch_current_prices


class DecisionAnalyzer:
    """Analyze quality of LLM trading decisions with improved accuracy."""
    
    def __init__(self, results_dir: str = "results/daily"):
        self.results_dir = Path(results_dir)
        self.analysis_cache = {}
    
    def load_decisions(self, days: int = 30) -> List[Dict]:
        """Load historical decisions from daily results."""
        decisions = []
        
        if not self.results_dir.exists():
            print(f"Warning: Results directory {self.results_dir} not found")
            return decisions
        
        # Get files from last N days
        files = sorted(self.results_dir.glob("*.json"))[-days:]
        print(f"Loading {len(files)} daily result files...")
        
        for file in files:
            try:
                with open(file) as f:
                    data = json.load(f)
                
                # Extract date from filename or data
                date_str = data.get("date", file.stem)
                
                # Only process if we have both decision and executed trades
                if "decision" in data and "executed_trades" in data:
                    trades = data.get("executed_trades", [])
                    
                    if trades:  # Only include days with actual trades
                        decisions.append({
                            "date": date_str,
                            "timestamp": data.get("timestamp"),
                            "actions": data["decision"].get("actions", []),
                            "trades": trades,
                            "reasoning": data["decision"].get("reasoning", ""),
                            "portfolio_before": data.get("portfolio_before", {}),
                            "portfolio_after": data.get("portfolio_after", {})
                        })
            except Exception as e:
                print(f"Warning: Could not load {file}: {e}")
                continue
        
        print(f"Loaded {len(decisions)} decision records with trades")
        return decisions
    
    def analyze_outcomes(self, decisions: List[Dict], 
                        forward_days: int = 5) -> Dict:
        """
        Analyze outcomes of decisions vs forward market performance.
        
        Args:
            decisions: List of decision records
            forward_days: How many days forward to check performance
        """
        if not HAS_DEPS:
            print("Warning: numpy/pandas not available, skipping analysis")
            return self._empty_metrics()
        
        outcomes = {
            "buys": [],
            "sells": [],
            "holds": [],
        }
        
        print(f"\nAnalyzing outcomes with {forward_days}-day forward window...")
        
        for i, decision in enumerate(decisions):
            date = decision["date"]
            
            for trade in decision.get("trades", []):
                ticker = trade["ticker"]
                action = trade["action"]
                price = trade.get("price", 0)
                
                if price == 0 or not price:
                    continue
                
                # Calculate forward return
                forward_return = self._get_forward_return(
                    ticker, date, price, forward_days
                )
                
                record = {
                    "ticker": ticker,
                    "date": date,
                    "entry_price": price,
                    "forward_return": forward_return,
                    "success": False
                }
                
                if action == "buy":
                    # Buy is successful if price goes up
                    record["success"] = forward_return > 0
                    outcomes["buys"].append(record)
                elif action == "sell":
                    # Sell is successful if price goes down (we avoided loss)
                    record["success"] = forward_return < 0
                    outcomes["sells"].append(record)
        
        print(f"  Buy decisions: {len(outcomes['buys'])}")
        print(f"  Sell decisions: {len(outcomes['sells'])}")
        
        return self._calculate_metrics(outcomes)
    
    def _get_forward_return(self, ticker: str, date: str, 
                           entry_price: float, days: int) -> float:
        """
        Calculate forward return for a decision.
        
        Looks up the price N trading days after the decision date
        and calculates the return.
        """
        try:
            # Fetch historical data - get enough days to cover forward window
            # Add buffer for weekends/holidays
            buffer_days = int(days * 1.5) + 10
            data = fetch_historical_data([ticker], period=f"{buffer_days}d")
            
            if ticker not in data or data[ticker].empty:
                return 0.0
            
            df = data[ticker].copy()
            
            # Ensure index is datetime (timezone-aware or naive)
            if not isinstance(df.index, pd.DatetimeIndex):
                df.index = pd.to_datetime(df.index)
            
            # Parse entry date - make it timezone-naive for comparison
            entry_date = pd.to_datetime(date)
            if entry_date.tzinfo is not None:
                entry_date = entry_date.tz_localize(None)
            
            # Make index timezone-naive for comparison
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Find the entry date or next available trading day
            mask = df.index >= entry_date
            if not mask.any():
                return 0.0
            
            # Get data from entry date onwards
            future_data = df[mask].sort_index()
            
            if len(future_data) < 2:  # Need at least entry + 1 day
                return 0.0
            
            # Get price at decision time (entry)
            entry_day_price = future_data["Close"].iloc[0]
            
            # Get price N days forward (or last available if not enough data)
            forward_idx = min(days, len(future_data) - 1)
            exit_price = future_data["Close"].iloc[forward_idx]
            
            # Calculate return
            return_pct = (exit_price - entry_day_price) / entry_day_price
            
            return return_pct
            
        except Exception as e:
            # Silent fail - return 0 for this calculation
            return 0.0
    
    def _calculate_metrics(self, outcomes: Dict) -> Dict:
        """Calculate aggregate metrics from outcomes."""
        if not HAS_DEPS:
            return self._empty_metrics()
        
        metrics = {
            "buy_accuracy": 0.0,
            "sell_accuracy": 0.0,
            "avg_forward_return_buy": 0.0,
            "avg_forward_return_sell": 0.0,
            "sharpe_of_decisions": 0.0,
            "total_decisions": 0,
            "win_rate": 0.0,
        }
        
        # Buy metrics
        if outcomes["buys"]:
            buy_returns = [r["forward_return"] for r in outcomes["buys"]]
            buy_successes = sum(1 for r in outcomes["buys"] if r["success"])
            
            metrics["buy_accuracy"] = buy_successes / len(outcomes["buys"])
            metrics["avg_forward_return_buy"] = np.mean(buy_returns)
            metrics["buy_count"] = len(outcomes["buys"])
        else:
            metrics["buy_count"] = 0
        
        # Sell metrics
        if outcomes["sells"]:
            sell_returns = [r["forward_return"] for r in outcomes["sells"]]
            sell_successes = sum(1 for r in outcomes["sells"] if r["success"])
            
            metrics["sell_accuracy"] = sell_successes / len(outcomes["sells"])
            # For sells, negative return means we avoided loss (good)
            # So we negate to show the "saved" return
            metrics["avg_forward_return_sell"] = -np.mean(sell_returns)
            metrics["sell_count"] = len(outcomes["sells"])
        else:
            metrics["sell_count"] = 0
        
        # Overall metrics
        all_decisions = outcomes["buys"] + outcomes["sells"]
        if all_decisions:
            metrics["total_decisions"] = len(all_decisions)
            wins = sum(1 for d in all_decisions if d["success"])
            metrics["win_rate"] = wins / len(all_decisions)
            
            # Calculate pseudo-Sharpe of decision returns
            returns = []
            for r in all_decisions:
                # For sells, invert the return (selling before a drop is good)
                ret = r["forward_return"]
                if r in outcomes["sells"]:
                    ret = -ret
                returns.append(ret)
            
            if len(returns) > 1 and np.std(returns) > 0:
                metrics["sharpe_of_decisions"] = np.mean(returns) / np.std(returns)
            else:
                metrics["sharpe_of_decisions"] = 0.0
        
        return metrics
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics when dependencies not available."""
        return {
            "buy_accuracy": 0.0,
            "sell_accuracy": 0.0,
            "avg_forward_return_buy": 0.0,
            "avg_forward_return_sell": 0.0,
            "sharpe_of_decisions": 0.0,
            "total_decisions": 0,
            "win_rate": 0.0,
            "buy_count": 0,
            "sell_count": 0,
        }
    
    def analyze_behavioral_patterns(self, decisions: List[Dict]) -> Dict:
        """Analyze behavioral patterns in LLM decisions."""
        patterns = {
            "herding_tendency": 0.0,
            "loss_aversion_score": 0.0,
            "overconfidence_check": 0.0,
            "diversification_score": 0.0,
            "avg_trades_per_day": 0.0,
            "unique_assets_traded": 0,
        }
        
        if not decisions:
            return patterns
        
        # Count unique tickers and total trades
        all_tickers = set()
        total_trades = 0
        
        for decision in decisions:
            for trade in decision.get("trades", []):
                all_tickers.add(trade["ticker"])
                total_trades += 1
        
        # Average trades per day
        patterns["avg_trades_per_day"] = total_trades / len(decisions) if decisions else 0
        patterns["unique_assets_traded"] = len(all_tickers)
        
        # Overconfidence: too many trades per day
        if patterns["avg_trades_per_day"] <= 2:
            patterns["overconfidence_check"] = 1.0
        elif patterns["avg_trades_per_day"] <= 4:
            patterns["overconfidence_check"] = 0.7
        else:
            patterns["overconfidence_check"] = 0.4
        
        # Diversification: trades spread across many assets
        patterns["diversification_score"] = min(len(all_tickers) / 10, 1.0)
        
        # Loss aversion: analyze if losses are cut quickly
        # This would require analyzing holding periods of losing positions
        # Simplified: check if sell decisions happen more often after losses
        patterns["loss_aversion_score"] = self._calculate_loss_aversion(decisions)
        
        return patterns
    
    def _calculate_loss_aversion(self, decisions: List[Dict]) -> float:
        """
        Calculate loss aversion score based on how quickly losses are cut.
        Higher score = better loss aversion (cutting losses quickly).
        """
        # Simplified heuristic: check ratio of sell to buy actions
        # In a declining market, more sells = better loss aversion
        sells = 0
        buys = 0
        
        for decision in decisions:
            for trade in decision.get("trades", []):
                if trade["action"] == "sell":
                    sells += 1
                elif trade["action"] == "buy":
                    buys += 1
        
        total = sells + buys
        if total == 0:
            return 0.5  # Neutral
        
        # In a bear market, more sells is good (cutting losses)
        # Score ranges from 0 (never sells) to 1 (always sells)
        return min(sells / total * 2, 1.0) if sells > 0 else 0.0
    
    def generate_report(self, days: int = 30) -> str:
        """Generate comprehensive analysis report."""
        decisions = self.load_decisions(days)
        
        if not decisions:
            return "No decision data available for analysis."
        
        # Analyze with different forward windows
        outcomes_5d = self.analyze_outcomes(decisions, forward_days=5)
        outcomes_1d = self.analyze_outcomes(decisions, forward_days=1)
        
        behavioral = self.analyze_behavioral_patterns(decisions)
        
        report = f"""
{'='*70}
LLM DECISION QUALITY ANALYSIS (Enhanced)
Analysis Period: Last {len(decisions)} trading days
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

TRADE STATISTICS
----------------
Total Trading Days Analyzed: {len(decisions)}
Total Trades Executed: {outcomes_5d.get('buy_count', 0) + outcomes_5d.get('sell_count', 0)}
  - Buy trades: {outcomes_5d.get('buy_count', 0)}
  - Sell trades: {outcomes_5d.get('sell_count', 0)}
Avg Trades per Day: {behavioral['avg_trades_per_day']:.2f}
Unique Assets Traded: {behavioral['unique_assets_traded']}

PERFORMANCE METRICS (5-Day Forward)
-----------------------------------
Overall Win Rate: {outcomes_5d['win_rate']*100:.1f}%

Buy Decisions:
  - Count: {outcomes_5d.get('buy_count', 0)}
  - Accuracy (price went up): {outcomes_5d['buy_accuracy']*100:.1f}%
  - Avg 5D Forward Return: {outcomes_5d['avg_forward_return_buy']*100:+.2f}%

Sell Decisions:
  - Count: {outcomes_5d.get('sell_count', 0)}
  - Accuracy (price went down): {outcomes_5d['sell_accuracy']*100:.1f}%
  - Avg 5D Return Avoided: {outcomes_5d['avg_forward_return_sell']*100:+.2f}%

Decision Sharpe Ratio: {outcomes_5d['sharpe_of_decisions']:.3f}

PERFORMANCE METRICS (1-Day Forward)
-----------------------------------
Overall Win Rate (1D): {outcomes_1d['win_rate']*100:.1f}%
Buy Accuracy (1D): {outcomes_1d['buy_accuracy']*100:.1f}%
Sell Accuracy (1D): {outcomes_1d['sell_accuracy']*100:.1f}%

BEHAVIORAL ANALYSIS
-------------------
Overconfidence Check: {behavioral['overconfidence_check']:.1f}/1.0
  (Lower trading frequency = better discipline)

Diversification Score: {behavioral['diversification_score']:.1f}/1.0
  (Number of unique assets traded)

Loss Aversion Score: {behavioral['loss_aversion_score']:.2f}/1.0
  (Ratio of sells in declining market)

ASSESSMENT
----------
"""
        
        # Add qualitative assessment
        win_rate = outcomes_5d['win_rate']
        sharpe = outcomes_5d['sharpe_of_decisions']
        
        if win_rate > 0.55:
            report += "✓ LLM shows predictive skill (win rate > 55%)\n"
        elif win_rate > 0.45:
            report += "~ LLM performance is near random (45-55% win rate)\n"
        elif win_rate > 0:
            report += "⚠ LLM underperforming (win rate < 45%)\n"
        else:
            report += "⚠ No valid win rate calculated (insufficient data)\n"
        
        if sharpe > 0.5:
            report += "✓ Positive risk-adjusted returns from decisions\n"
        elif sharpe > 0:
            report += "~ Marginal risk-adjusted returns\n"
        elif sharpe < 0:
            report += "⚠ Negative risk-adjusted returns - review strategy\n"
        
        if behavioral['overconfidence_check'] < 0.5:
            report += "⚠ High trading frequency detected - potential overtrading\n"
        
        report += f"\n{'='*70}\n"
        
        return report


def main():
    """Run decision analysis."""
    print("Starting LLM Decision Quality Analysis...\n")
    
    analyzer = DecisionAnalyzer()
    report = analyzer.generate_report(days=30)
    print(report)
    
    # Save report
    output_dir = Path("results/analysis")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"decision_analysis_{datetime.now().strftime('%Y%m%d')}.txt"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"Report saved to: {output_file}")


if __name__ == "__main__":
    main()
