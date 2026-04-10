#!/usr/bin/env python3
"""
Research Demo: Decision Memory & Prompt Optimization

Demonstrates the new research modules:
1. Decision Memory - Learning from past trades
2. Prompt Optimizer - Testing prompt variants
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from analysis.decision_memory import DecisionMemory, DecisionRecord
from llm.prompt_optimizer import PromptOptimizer, PromptVariant
from datetime import datetime, timedelta


def demo_decision_memory():
    """Demonstrate the decision memory system."""
    print("="*80)
    print("DEMO: DECISION MEMORY SYSTEM")
    print("="*80)
    
    memory = DecisionMemory()
    
    # Simulate adding some decisions (in production, these come from actual trades)
    print("\n1. Adding sample decisions to memory...")
    
    sample_decisions = [
        DecisionRecord(
            date="2026-04-01",
            ticker="RMS.PA",
            action="buy",
            quantity=0.5,
            price=1663.48,
            portfolio_value_before=9578.85,
            portfolio_value_after=9576.50,
            rsi=38.1,
            bollinger_position=0.30,
            reasoning="Mean reversion setup: RSI oversold, Bollinger near lower band",
            pnl_pct=5.95,  # Filled in later
            holding_period_days=9
        ),
        DecisionRecord(
            date="2026-04-03",
            ticker="SLV",
            action="sell",
            quantity=10.0,
            price=65.80,
            portfolio_value_before=9576.56,
            portfolio_value_after=9576.56,
            rsi=38.5,
            bollinger_position=0.37,
            reasoning="Risk management: High volatility (60%), cut loss before deeper drawdown",
            pnl_pct=-0.1,
            holding_period_days=1
        ),
        DecisionRecord(
            date="2026-04-08",
            ticker="DBA",
            action="buy",
            quantity=23.88,
            price=26.87,
            portfolio_value_before=9719.84,
            portfolio_value_after=9719.84,
            rsi=48.7,
            bollinger_position=0.42,
            reasoning="Diversification play: Low vol commodity, negative corr to SPY",
            pnl_pct=0.04,
            holding_period_days=2
        ),
    ]
    
    for decision in sample_decisions:
        memory.add_decision(decision)
    
    memory.save_memory()
    
    # Show summary
    print("\n2. Decision Summary (Last 30 days):")
    summary = memory.get_decision_summary(days=30)
    for key, value in summary.items():
        if isinstance(value, dict):
            print(f"   {key}:")
            for k, v in value.items():
                print(f"      {k}: {v}")
        else:
            print(f"   {key}: {value}")
    
    # Show lessons
    print("\n3. Lessons Learned:")
    lessons = memory.generate_lessons_learned()
    for lesson in lessons:
        print(f"   {lesson}")
    
    # Show LLM context
    print("\n4. Context for LLM Prompt:")
    context = memory.get_memory_context_for_llm()
    print(context)


def demo_prompt_optimizer():
    """Demonstrate the prompt optimizer."""
    print("\n" + "="*80)
    print("DEMO: PROMPT OPTIMIZER")
    print("="*80)
    
    # Create optimizer for recent period
    end_date = datetime.now()
    start_date = end_date - timedelta(days=30)  # Short period for demo
    
    optimizer = PromptOptimizer(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        initial_capital=10000.0
    )
    
    # Show available variants
    print("\n1. Available Prompt Variants:")
    variants = optimizer.create_default_variants()
    for i, variant in enumerate(variants, 1):
        print(f"   {i}. {variant.name}: {variant.description}")
    
    # Note about backtesting
    print("\n2. Backtesting Note:")
    print("   Full backtesting requires:")
    print("   - Historical market data for the period")
    print("   - LLM API calls for each trading day (or cached decisions)")
    print("   - Position tracking and P&L calculation")
    print("   ")
    print("   To run full optimization:")
    print("   $ python src/llm/prompt_optimizer.py")
    
    # Show what metrics would be calculated
    print("\n3. Metrics Calculated for Each Variant:")
    metrics = [
        "Total Return (%)",
        "Sharpe Ratio",
        "Maximum Drawdown (%)",
        "Win Rate (%)",
        "Calmar Ratio (Return / Max DD)",
        "Volatility (annualized)",
        "Total Trades",
        "Cash Utilization"
    ]
    for metric in metrics:
        print(f"   - {metric}")


def integration_example():
    """Show how to integrate with daily_run.py."""
    print("\n" + "="*80)
    print("INTEGRATION: USING IN DAILY RUN")
    print("="*80)
    
    code_example = '''
# In daily_run.py, add to the LLM context:

from analysis.decision_memory import DecisionMemory

# Initialize memory
memory = DecisionMemory()

# Get lessons learned
lessons_context = memory.get_memory_context_for_llm()

# Append to system prompt
enhanced_prompt = system_prompt + "\\n\\n" + lessons_context

# Use enhanced_prompt for LLM call
response = trading_agent.get_decision(market_data, enhanced_prompt)

# After executing trades, record the decision
from analysis.decision_memory import DecisionRecord

decision = DecisionRecord(
    date=today,
    ticker=trade["ticker"],
    action=trade["action"],
    quantity=trade["quantity"],
    price=trade["price"],
    # ... other fields
)
memory.add_decision(decision)
'''
    
    print("\nCode snippet for integration:")
    print(code_example)


def main():
    """Run all demos."""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*20 + "RESEARCH MODULES DEMO" + " "*37 + "║")
    print("║" + " "*15 + "Decision Memory & Prompt Optimization" + " "*26 + "║")
    print("╚" + "="*78 + "╝")
    
    demo_decision_memory()
    demo_prompt_optimizer()
    integration_example()
    
    print("\n" + "="*80)
    print("NEXT STEPS")
    print("="*80)
    print("""
1. Run decision memory on historical data:
   $ python src/analysis/decision_memory.py

2. Run prompt optimization (requires historical data + LLM calls):
   $ python src/llm/prompt_optimizer.py

3. Integrate into daily_run.py for live learning

4. Analyze results and refine prompts based on:
   - Which behavioral components improve performance
   - Optimal trade frequency
   - Indicator thresholds that work best
""")


if __name__ == "__main__":
    main()
