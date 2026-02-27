"""
LLM trading agent module.
Integrates with LLM API to get trading decisions based on market state and portfolio.
"""

import os
import json
import logging
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import requests

# Try to load dotenv, but don't fail if not installed
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # Use environment variables directly

# Import risk metrics module
try:
    from ..risk import calculate_portfolio_risk_metrics, get_risk_summary_for_llm
    RISK_MODULE_AVAILABLE = True
except ImportError:
    RISK_MODULE_AVAILABLE = False

logger = logging.getLogger(__name__)

# System prompt with risk management principles inspired by prospect theory
SYSTEM_PROMPT = """You are a sophisticated quantitative trading agent operating in a paper trading environment with 10,000 EUR initial capital.

Your decisions must follow these principles inspired by Prospect Theory and Behavioral Finance:

1. LOSS AVERSION: Losses are psychologically ~2.25x more painful than equivalent gains are pleasurable. Protect against downside first.

2. RISK SENSITIVITY: Use a CVaR (Conditional Value at Risk) mindset - focus on tail risks, not just variance.

3. MOMENTUM & MEAN REVERSION: 
   - RSI < 30: Potential oversold (mean reversion opportunity)
   - RSI > 70: Potential overbought (momentum may continue but risk increases)
   - Consider Bollinger Band position (0=lower band, 1=upper band)

4. POSITION SIZING:
   - Maximum 25% of portfolio in any single position
   - Keep 10-30% cash buffer for opportunities
   - Scale in/out gradually rather than all at once

5. DIVERSIFICATION:
   - Monitor correlations between positions
   - Avoid concentration in highly correlated assets
   - Consider geographic/sector diversification

6. STOP LOSS MENTALITY:
   - If drawdown > 5% on a position, consider reducing
   - If portfolio drawdown > 3% in a day, get defensive

OUTPUT FORMAT:
Respond with a JSON object containing:
{
  "actions": [
    {"ticker": "SPY", "action": "buy", "pct": 15},
    {"ticker": "MC.PA", "action": "sell", "pct": 100},
    {"ticker": "GLD", "action": "hold"}
  ],
  "reasoning": "Brief explanation of your decision process, referencing specific indicators and risk considerations"
}

ACTIONS:
- "buy" with "pct": percentage of available cash to deploy
- "sell" with "pct": percentage of position to sell (use 100 for full exit)
- "hold": no action

Remember: You are risk-aware, not risk-seeking. Preserve capital first, grow second."""


class TradingAgent:
    """
    LLM-powered trading agent.
    Fetches decisions from LLM API based on market context.
    """
    
    def __init__(
        self,
        api_key: Optional[str] = None,
        api_url: Optional[str] = None,
        model: str = None,
        history_file: str = "data/decision_history.json"
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.api_url = api_url or os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.model = model or os.getenv("LLM_MODEL", "kimi")
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        
        if not self.api_key:
            logger.warning("No LLM API key configured!")
    
    def load_recent_decisions(self, days: int = 5) -> List[Dict]:
        """Load recent trading decisions from history."""
        if not self.history_file.exists():
            return []
        
        try:
            with open(self.history_file, 'r') as f:
                all_decisions = json.load(f)
            
            # Filter to recent days
            cutoff = datetime.now() - timedelta(days=days)
            recent = [
                d for d in all_decisions
                if datetime.fromisoformat(d['timestamp']) > cutoff
            ]
            return recent[-5:]  # Last 5 decisions max
        except Exception as e:
            logger.error(f"Error loading decision history: {e}")
            return []
    
    def save_decision(self, decision: Dict) -> None:
        """Save decision to history file."""
        decisions = []
        if self.history_file.exists():
            try:
                with open(self.history_file, 'r') as f:
                    decisions = json.load(f)
            except:
                decisions = []
        
        decisions.append(decision)
        
        # Keep only last 100 decisions
        decisions = decisions[-100:]
        
        with open(self.history_file, 'w') as f:
            json.dump(decisions, f, indent=2)
    
    def build_prompt(
        self,
        market_data: Dict,
        portfolio_summary: Dict,
        recent_decisions: Optional[List[Dict]] = None
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            market_data: Dict with asset indicators and correlations
            portfolio_summary: Current portfolio state
            recent_decisions: List of recent trading decisions
        
        Returns:
            Formatted prompt string
        """
        prompt_parts = []
        
        # Market state
        prompt_parts.append("=== MARKET STATE ===\n")
        
        assets = market_data.get('assets', {})
        for ticker, data in assets.items():
            latest = data.get('latest', {})
            prompt_parts.append(f"\n{ticker}:")
            prompt_parts.append(f"  Price: €{latest.get('price', 0):.2f}")
            prompt_parts.append(f"  SMA20: €{latest.get('sma_20', 0):.2f} | SMA50: €{latest.get('sma_50', 0):.2f}")
            prompt_parts.append(f"  RSI(14): {latest.get('rsi_14', 50):.1f}")
            prompt_parts.append(f"  Bollinger Position: {latest.get('bb_position', 0.5):.2f}")
            prompt_parts.append(f"  Volatility (ann): {latest.get('volatility_annual', 0)*100:.1f}%")
            prompt_parts.append(f"  Drawdown: {latest.get('drawdown', 0)*100:.2f}%")
            prompt_parts.append(f"  Daily Return: {latest.get('daily_return', 0)*100:.2f}%")
        
        # Correlations
        correlations = market_data.get('correlations', {})
        if not correlations.empty:
            prompt_parts.append("\n=== CORRELATIONS (20-day returns) ===")
            prompt_parts.append(correlations.to_string())
        
        # Portfolio state
        prompt_parts.append("\n\n=== PORTFOLIO STATE ===")
        prompt_parts.append(f"Cash: €{portfolio_summary.get('cash', 0):.2f}")
        prompt_parts.append(f"Total Value: €{portfolio_summary.get('total_value', 0):.2f}")
        prompt_parts.append(f"Total Return: {portfolio_summary.get('total_return_pct', 0):.2f}%")
        prompt_parts.append(f"Total P&L: €{portfolio_summary.get('total_pnl', 0):+.2f}")
        
        # Risk metrics (CVaR)
        risk_metrics = portfolio_summary.get('risk_metrics', {})
        if risk_metrics:
            prompt_parts.append("\n=== RISK METRICS (Tail Risk Analysis) ===")
            prompt_parts.append(f"CVaR 95% (Expected Shortfall): {risk_metrics.get('cvar_95', 0)*100:.2f}%")
            prompt_parts.append(f"VaR 95%: {risk_metrics.get('var_95', 0)*100:.2f}%")
            prompt_parts.append(f"Max Drawdown: {risk_metrics.get('max_drawdown', 0)*100:.2f}%")
            prompt_parts.append(f"Sortino Ratio: {risk_metrics.get('sortino_ratio', 0):.2f}")
            prompt_parts.append(f"Return Skewness: {risk_metrics.get('skewness', 0):.2f}")
            prompt_parts.append(f"Return Kurtosis: {risk_metrics.get('kurtosis', 0):.2f}")
            prompt_parts.append("\nNote: CVaR measures expected loss in worst 5% of cases. Lower is safer.")
        
        positions = portfolio_summary.get('positions', [])
        if positions:
            prompt_parts.append("\nCurrent Positions:")
            for pos in positions:
                prompt_parts.append(
                    f"  {pos['ticker']}: {pos['quantity']:.4f} shares, "
                    f"avg €{pos['avg_price']:.2f}, current €{pos['current_price']:.2f}, "
                    f"P&L {pos['unrealized_pnl_pct']:+.2f}%"
                )
        else:
            prompt_parts.append("\nNo current positions (all cash)")
        
        # Risk metrics (if available)
        if RISK_MODULE_AVAILABLE and 'historical_prices' in market_data:
            try:
                prices_dict = market_data['historical_prices']
                weights = {pos['ticker']: pos['market_value'] for pos in positions}
                risk_metrics = calculate_portfolio_risk_metrics(prices_dict, weights)
                prompt_parts.append("\n\n=== PORTFOLIO RISK METRICS ===")
                prompt_parts.append(get_risk_summary_for_llm(risk_metrics))
            except Exception as e:
                logger.warning(f"Could not calculate risk metrics: {e}")
        
        # Recent decisions
        if recent_decisions:
            prompt_parts.append("\n\n=== RECENT DECISIONS (last 5 days) ===")
            for d in recent_decisions:
                prompt_parts.append(f"\n{d['timestamp'][:10]}:")
                prompt_parts.append(f"  Reasoning: {d.get('reasoning', 'N/A')[:200]}...")
                actions = d.get('actions', [])
                for a in actions:
                    prompt_parts.append(f"  - {a['ticker']}: {a['action']} {a.get('pct', 0)}%")
        
        prompt_parts.append("\n\n=== YOUR DECISION ===")
        prompt_parts.append("Based on the above market state and portfolio, what actions should we take?")
        prompt_parts.append("Respond with the JSON format specified in your instructions.")
        
        return "\n".join(prompt_parts)
    
    def call_llm(self, prompt: str) -> Optional[str]:
        """
        Call the LLM API.
        
        Args:
            prompt: The formatted prompt
        
        Returns:
            LLM response text or None if error
        """
        if not self.api_key:
            logger.error("No API key configured!")
            return None
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": os.getenv("LLM_USER_AGENT", "python-trading-agent/1.0")
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 16000,
        }
        
        try:
            logger.info("Calling LLM API...")
            response = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=180
            )
            response.raise_for_status()
            
            result = response.json()
            msg = result["choices"][0]["message"]
            content = msg.get("content", "") or ""
            # Kimi may put the answer in reasoning_content instead of content
            if not content.strip() and msg.get("reasoning_content"):
                content = msg["reasoning_content"]
                logger.info("Using reasoning_content as content (Kimi quirk)")
            logger.info(f"LLM content length: {len(content) if content else 0}")
            logger.info(f"LLM content preview: {repr(content[:200]) if content else None}")
            logger.info("✓ LLM response received")
            return content
            
        except Exception as e:
            logger.error(f"Error calling LLM API: {e}")
            return None
    
    def parse_response(self, response: str) -> Dict:
        """
        Parse LLM response to extract actions.
        
        Args:
            response: Raw LLM response
        
        Returns:
            Parsed decision dict with actions and reasoning
        """
        try:
            # Try to extract JSON from the response
            # Response may be: pure JSON, markdown-wrapped JSON, or free text with JSON embedded
            json_str = None
            
            if "```json" in response:
                json_str = response.split("```json")[1].split("```")[0].strip()
            elif "```" in response:
                json_str = response.split("```")[1].split("```")[0].strip()
            
            # Try to find JSON object with "actions" key in the text
            if json_str is None:
                import re
                # Find the last JSON-like block containing "actions"
                matches = list(re.finditer(r'{[^{}]*"actions"\s*:', response))
                if matches:
                    start = matches[-1].start()
                    # Find matching closing brace
                    depth = 0
                    for i in range(start, len(response)):
                        if response[i] == "{":
                            depth += 1
                        elif response[i] == "}":
                            depth -= 1
                            if depth == 0:
                                json_str = response[start:i+1]
                                break
            
            if json_str is None:
                json_str = response.strip()
            
            parsed = json.loads(json_str)
            
            # Validate structure
            if "actions" not in parsed:
                raise ValueError("Missing 'actions' key")
            
            for action in parsed["actions"]:
                if "ticker" not in action or "action" not in action:
                    raise ValueError("Invalid action format")
                if action["action"] not in ["buy", "sell", "hold"]:
                    raise ValueError(f"Invalid action: {action['action']}")
            
            return parsed
            
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            logger.error(f"Response: {response[:500]}")
            # Return hold all on error
            return {
                "actions": [],
                "reasoning": f"Error parsing response: {e}. Defaulting to hold all positions.",
                "error": True
            }
    
    def get_trading_decision(
        self,
        market_data: Dict,
        portfolio_summary: Dict
    ) -> Dict:
        """
        Get trading decision from LLM.
        
        Args:
            market_data: Market indicators and correlations
            portfolio_summary: Current portfolio state
        
        Returns:
            Decision dict with actions and metadata
        """
        # Load recent decisions for context
        recent_decisions = self.load_recent_decisions(days=5)
        
        # Build prompt
        prompt = self.build_prompt(market_data, portfolio_summary, recent_decisions)
        
        # Call LLM
        response = self.call_llm(prompt)
        
        if response is None:
            # API error - hold all
            decision = {
                "timestamp": datetime.now().isoformat(),
                "actions": [],
                "reasoning": "LLM API error. Holding all positions.",
                "error": True
            }
        else:
            # Parse response
            parsed = self.parse_response(response)
            decision = {
                "timestamp": datetime.now().isoformat(),
                "actions": parsed.get("actions", []),
                "reasoning": parsed.get("reasoning", "No reasoning provided"),
                "error": parsed.get("error", False),
                "raw_response": response[:1000]  # Truncate for storage
            }
        
        # Save to history
        self.save_decision(decision)
        
        return decision


if __name__ == "__main__":
    # Quick test
    print("Testing TradingAgent...")
    
    agent = TradingAgent()
    
    # Mock data
    market_data = {
        "assets": {
            "SPY": {
                "latest": {
                    "price": 400.0,
                    "sma_20": 395.0,
                    "sma_50": 390.0,
                    "rsi_14": 55.0,
                    "bb_position": 0.6,
                    "volatility_annual": 0.15,
                    "drawdown": -0.02,
                    "daily_return": 0.005
                }
            }
        },
        "correlations": None
    }
    
    portfolio = {
        "cash": 10000.0,
        "total_value": 10000.0,
        "total_return_pct": 0.0,
        "total_pnl": 0.0,
        "positions": []
    }
    
    if os.getenv("LLM_API_KEY"):
        decision = agent.get_trading_decision(market_data, portfolio)
        print("\nDecision:")
        print(json.dumps(decision, indent=2))
    else:
        print("No API key configured - skipping LLM call")
