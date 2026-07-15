#!/usr/bin/env python3
"""
Decision Memory Module.

Stores and analyzes past trading decisions to enable learning from history.
Provides the LLM with context about its own decision patterns and outcomes.
"""

import json
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent / ".."))

import numpy as np
import pandas as pd


@dataclass
class DecisionRecord:
    """A single trading decision with context and outcome."""
    date: str
    ticker: str
    action: str  # buy, sell, hold
    quantity: float
    price: float
    portfolio_value_before: float
    portfolio_value_after: float
    
    # Market context at decision time
    rsi: Optional[float] = None
    bollinger_position: Optional[float] = None
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    volatility: Optional[float] = None
    
    # Decision rationale
    reasoning: str = ""
    
    # Outcome tracking (filled in later)
    exit_price: Optional[float] = None
    holding_period_days: Optional[int] = None
    pnl_pct: Optional[float] = None
    max_drawdown_during_hold: Optional[float] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DecisionRecord':
        return cls(**data)


class DecisionMemory:
    """
    Memory system for trading decisions.
    
    Stores decisions, tracks their outcomes, and provides insights
    for future decision-making.
    """
    
    def __init__(self, memory_file: str = "data/decision_memory.json"):
        self.memory_file = Path(memory_file)
        self.decisions: List[DecisionRecord] = []
        self._load_memory()
    
    def _load_memory(self):
        """Load existing decision memory from disk."""
        if self.memory_file.exists():
            try:
                with open(self.memory_file) as f:
                    data = json.load(f)
                    self.decisions = [DecisionRecord.from_dict(d) for d in data]
                print(f"Loaded {len(self.decisions)} decisions from memory")
            except Exception as e:
                print(f"Warning: Could not load decision memory: {e}")
                self.decisions = []
        else:
            self.decisions = []
    
    def save_memory(self):
        """Save decision memory to disk."""
        self.memory_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.memory_file, 'w') as f:
            json.dump([d.to_dict() for d in self.decisions], f, indent=2, default=str)
        print(f"Saved {len(self.decisions)} decisions to memory")
    
    def add_decision(self, decision: DecisionRecord):
        """Add a new decision to memory."""
        self.decisions.append(decision)
        # Auto-save every 10 decisions
        if len(self.decisions) % 10 == 0:
            self.save_memory()
    
    def update_outcomes(self, ticker: str, current_price: float, date: str):
        """
        Update outcomes for open positions.
        
        Call this periodically to fill in exit prices and P&L for
        decisions that have been closed.
        """
        for decision in self.decisions:
            if (decision.ticker == ticker and 
                decision.exit_price is None and 
                decision.action in ["buy", "sell"]):
                
                # For now, simple logic: if we have a buy, and later a sell
                # In production, this would track actual position closes
                pass
    
    def get_decision_summary(self, days: int = 30) -> Dict:
        """Generate summary statistics of recent decisions."""
        cutoff_date = (datetime.now() - timedelta(days=days)).date()
        recent_decisions = [
            d for d in self.decisions
            if datetime.strptime(d.date, "%Y-%m-%d").date() >= cutoff_date
        ]
        
        if not recent_decisions:
            return {
                "period_days": days,
                "total_decisions": 0,
                "message": "No decisions in this period"
            }
        
        # Count by action
        action_counts = defaultdict(int)
        for d in recent_decisions:
            action_counts[d.action] += 1
        
        # Calculate win rate for completed trades
        completed = [d for d in recent_decisions if d.pnl_pct is not None]
        winners = [d for d in completed if d.pnl_pct > 0]
        
        win_rate = len(winners) / len(completed) if completed else 0
        
        # Average P&L
        avg_pnl = np.mean([d.pnl_pct for d in completed]) if completed else 0
        
        # Average holding period
        holding_periods = [d.holding_period_days for d in completed if d.holding_period_days is not None]
        avg_hold = np.mean(holding_periods) if holding_periods else 0
        
        return {
            "period_days": days,
            "total_decisions": len(recent_decisions),
            "action_breakdown": dict(action_counts),
            "completed_trades": len(completed),
            "win_rate": win_rate,
            "avg_pnl_pct": avg_pnl,
            "avg_holding_days": avg_hold,
            "best_trade": max([d.pnl_pct for d in completed]) if completed else None,
            "worst_trade": min([d.pnl_pct for d in completed]) if completed else None,
        }
    
    def get_pattern_analysis(self) -> Dict:
        """
        Analyze patterns in decision making.
        
        Identifies:
        - Which indicators correlate with success
        - Optimal holding periods
        - Behavioral biases (herding, overtrading, etc.)
        """
        completed = [d for d in self.decisions if d.pnl_pct is not None]
        
        if len(completed) < 10:
            return {
                "status": "insufficient_data",
                "message": f"Need at least 10 completed trades, have {len(completed)}"
            }
        
        analysis = {
            "status": "ok",
            "total_analyzed": len(completed),
            "winners": len([d for d in completed if d.pnl_pct > 0]),
            "losers": len([d for d in completed if d.pnl_pct <= 0]),
        }
        
        # Analyze RSI correlation with success
        rsi_data = [(d.rsi, d.pnl_pct) for d in completed if d.rsi is not None]
        if len(rsi_data) > 5:
            rsis, pnls = zip(*rsi_data)
            analysis["rsi_correlation"] = np.corrcoef(rsis, pnls)[0, 1] if len(rsis) > 1 else 0
        
        # Analyze Bollinger position correlation
        bb_data = [(d.bollinger_position, d.pnl_pct) for d in completed if d.bollinger_position is not None]
        if len(bb_data) > 5:
            bbs, pnls = zip(*bb_data)
            analysis["bollinger_correlation"] = np.corrcoef(bbs, pnls)[0, 1] if len(bbs) > 1 else 0
        
        # Optimal holding period
        hold_data = [(d.holding_period_days, d.pnl_pct) for d in completed if d.holding_period_days]
        if hold_data:
            holds, pnls = zip(*hold_data)
            # Simple binning by holding period
            short_term = [p for h, p in hold_data if h <= 5]
            medium_term = [p for h, p in hold_data if 5 < h <= 20]
            long_term = [p for h, p in hold_data if h > 20]
            
            analysis["holding_period_performance"] = {
                "short_term_5d": np.mean(short_term) if short_term else None,
                "medium_term_5_20d": np.mean(medium_term) if medium_term else None,
                "long_term_20d_plus": np.mean(long_term) if long_term else None,
            }
        
        # Behavioral indicators
        recent_dates = sorted(set([d.date for d in self.decisions[-20:]]))  # Last 20 unique dates
        trades_per_day = len([d for d in self.decisions if d.date in recent_dates]) / len(recent_dates) if recent_dates else 0
        
        analysis["behavioral_indicators"] = {
            "avg_trades_per_day": trades_per_day,
            "overtrading_flag": trades_per_day > 3,  # More than 3 trades/day is suspicious
            "recent_concentration": len(set([d.ticker for d in self.decisions[-10:]])) if len(self.decisions) >= 10 else 0,
        }
        
        return analysis
    
    def generate_lessons_learned(self) -> List[str]:
        """
        Generate actionable lessons from decision history.
        
        Returns a list of insights that can be added to the LLM prompt.
        """
        lessons = []
        
        summary = self.get_decision_summary(days=90)
        patterns = self.get_pattern_analysis()
        
        if summary["total_decisions"] == 0:
            return ["No trading history yet. Focus on building a track record."]
        
        # Win rate lesson
        if summary.get("win_rate"):
            if summary["win_rate"] < 0.4:
                lessons.append(f"⚠️ Recent win rate is {summary['win_rate']:.1%} — below random. Review entry criteria.")
            elif summary["win_rate"] > 0.55:
                lessons.append(f"✓ Recent win rate is {summary['win_rate']:.1%} — strategy showing edge. Maintain discipline.")
        
        # Overtrading lesson
        if summary["total_decisions"] > 60:  # More than 2 trades/day average
            lessons.append("⚠️ High trade frequency detected. Consider fewer, higher-conviction trades.")
        
        # P&L lesson
        if summary.get("avg_pnl_pct"):
            if summary["avg_pnl_pct"] < -1:
                lessons.append(f"⚠️ Average loss per trade: {summary['avg_pnl_pct']:.2f}%. Review position sizing and stop losses.")
            elif summary["avg_pnl_pct"] > 1:
                lessons.append(f"✓ Average gain per trade: {summary['avg_pnl_pct']:.2f}%. Good risk/reward balance.")
        
        # Pattern-based lessons
        if patterns.get("status") == "ok":
            # RSI correlation
            if "rsi_correlation" in patterns:
                corr = patterns["rsi_correlation"]
                if corr < -0.3:
                    lessons.append("📊 Lower RSI entries tend to perform better (mean reversion working).")
                elif corr > 0.3:
                    lessons.append("📊 Higher RSI entries tend to perform better (momentum working).")
            
            # Bollinger correlation
            if "bollinger_correlation" in patterns:
                corr = patterns["bollinger_correlation"]
                if corr < -0.3:
                    lessons.append("📊 Entries near lower Bollinger band perform better (oversold bounces).")
            
            # Holding period
            if "holding_period_performance" in patterns:
                hp = patterns["holding_period_performance"]
                if hp["short_term_5d"] and hp["long_term_20d_plus"]:
                    if hp["short_term_5d"] > hp["long_term_20d_plus"]:
                        lessons.append("📊 Short-term holds (≤5d) outperform longer holds. Consider quicker profit-taking.")
                    else:
                        lessons.append("📊 Longer holds (>20d) outperform short-term. Let winners run.")
        
        # Behavioral lessons
        if patterns.get("behavioral_indicators", {}).get("overtrading_flag"):
            lessons.append("⚠️ Overtrading detected. Implement a cooling-off period between trades.")
        
        if not lessons:
            lessons.append("📈 Building track record. Focus on consistent execution of strategy.")
        
        return lessons
    
    def get_memory_context_for_llm(self) -> str:
        """
        Generate a context string for the LLM that summarizes
        lessons from past decisions.
        
        This can be appended to the system prompt.
        """
        summary = self.get_decision_summary(days=30)
        lessons = self.generate_lessons_learned()
        
        context = f"""
YOUR RECENT TRADING TRACK RECORD (Last {summary.get('period_days', 30)} days):
- Total decisions: {summary.get('total_decisions', 0)}
- Completed trades with outcomes: {summary.get('completed_trades', 0)}
- Win rate: {summary.get('win_rate', 0):.1%}
- Average P&L per trade: {summary.get('avg_pnl_pct', 0):+.2f}%

KEY LESSONS FROM YOUR HISTORY:
"""
        for lesson in lessons:
            context += f"\n{lesson}"
        
        context += "\n\nApply these insights to today's decisions. Learn from patterns, but avoid overfitting to recent noise."
        
        return context
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Export all decisions to a pandas DataFrame for analysis."""
        if not self.decisions:
            return pd.DataFrame()
        
        data = [d.to_dict() for d in self.decisions]
        return pd.DataFrame(data)
    
    def get_similar_decisions(
        self, 
        ticker: str, 
        rsi: float, 
        bollinger: float,
        n: int = 5
    ) -> List[DecisionRecord]:
        """
        Find similar past decisions based on market conditions.
        
        Useful for: "What happened last time I bought X when RSI was around Y?"
        """
        # Calculate similarity score (lower is more similar)
        scored = []
        for d in self.decisions:
            if d.ticker != ticker or d.rsi is None or d.bollinger_position is None:
                continue
            
            rsi_diff = abs(d.rsi - rsi) / 100  # Normalized
            bb_diff = abs(d.bollinger_position - bollinger)
            
            similarity = rsi_diff + bb_diff
            scored.append((similarity, d))
        
        # Sort by similarity and return top N
        scored.sort(key=lambda x: x[0])
        return [d for _, d in scored[:n]]


def main():
    """Demo the decision memory system."""
    memory = DecisionMemory()
    
    print("="*70)
    print("DECISION MEMORY SYSTEM")
    print("="*70)
    
    # Show summary
    print("\n📊 RECENT DECISION SUMMARY (30 days):")
    summary = memory.get_decision_summary(days=30)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Show pattern analysis
    print("\n🔍 PATTERN ANALYSIS:")
    patterns = memory.get_pattern_analysis()
    for key, value in patterns.items():
        if isinstance(value, dict):
            print(f"  {key}:")
            for k, v in value.items():
                print(f"    {k}: {v}")
        else:
            print(f"  {key}: {value}")
    
    # Show lessons
    print("\n📚 LESSONS LEARNED:")
    lessons = memory.generate_lessons_learned()
    for lesson in lessons:
        print(f"  {lesson}")
    
    # Show LLM context
    print("\n🤖 CONTEXT FOR LLM:")
    context = memory.get_memory_context_for_llm()
    print(context)


if __name__ == "__main__":
    main()
