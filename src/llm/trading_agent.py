"""
LLM trading agent module.
Integrates with LLM API to get trading decisions based on market state and portfolio.
"""

import os
import json
import logging
import time
import random
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from pathlib import Path
import requests
try:
    import pandas as pd
    import numpy as np
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False

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

# Import Deflated Sharpe Ratio module
try:
    from ..backtest.deflated_sharpe import DeflatedSharpeRatio
    import numpy as np
    DSR_MODULE_AVAILABLE = True
except ImportError:
    DSR_MODULE_AVAILABLE = False

logger = logging.getLogger(__name__)

# System prompt with risk management principles inspired by prospect theory
SYSTEM_PROMPT = """You are a sophisticated quantitative trading agent operating in a paper trading environment with 10,000 EUR initial capital.

Your decisions must follow these principles inspired by Prospect Theory, Behavioral Finance, and Advances in Financial Machine Learning (Lopez de Prado):

1. LOSS AVERSION: Losses are psychologically ~2.25x more painful than equivalent gains are pleasurable. Protect against downside first.

2. RISK SENSITIVITY: Use a CVaR (Conditional Value at Risk) mindset - focus on tail risks, not just variance. Question any "high Sharpe ratio" strategy - it may be a false discovery from multiple testing.

3. MOMENTUM & MEAN REVERSION: 
   - RSI < 30: Potential oversold (mean reversion opportunity)
   - RSI > 70: Potential overbought (momentum may continue but risk increases)
   - Consider Bollinger Band position (0=lower band, 1=upper band)

4. DEFLATED SHARPE RATIO MINDSET:
   - The standard Sharpe ratio overstates true performance due to:
     a) Non-normality (skewness, kurtosis) of returns
     b) Multiple testing bias (testing many strategies, some will appear good by chance)
   - Be skeptical of strategies with short track records (< 1 year)
   - Prefer strategies with positive skew and low excess kurtosis
   - If current portfolio shows high kurtosis (> 3), reduce risk

5. POSITION SIZING & CASH TARGETS:
   - Maximum 25% of portfolio in any single position
   - Cash buffer targets by volatility regime:
     * HIGH volatility: 30-50% cash (capital preservation)
     * NORMAL volatility: 15-30% cash (balanced)
     * LOW volatility: 10-20% cash (fully deployed)
   - If current cash is above the regime target, you are under-invested.
     Find opportunities to deploy capital gradually, unless the Market Regime Analysis explicitly recommends defense.
   - Scale in/out gradually rather than all at once
   - When adding a new position, consider: "Would this pass a Deflated Sharpe Ratio test?"

6. DIVERSIFICATION:
   - Monitor correlations between positions
   - Avoid concentration in highly correlated assets
   - Consider geographic/sector diversification
   - Remember: during crises, correlations tend to 1.0

7. STOP LOSS MENTALITY:
   - If drawdown > 5% on a single position, consider reducing
   - If portfolio drops > 3% in a single trading day, get defensive
   - If total portfolio drawdown from inception exceeds 5-7%, reduce risk and reassess
   - A -3% total inception drawdown is not a daily crash; it requires caution, not maximum defense
   - Cut losses quickly, let winners run (with trailing stops)

8. META-LABELING PRINCIPLE:
   - Primary model predicts direction (up/down)
   - Secondary model (meta-labeling) predicts probability of being right
   - Only trade when you have both: directional edge AND high confidence
   - Current confidence proxy: RSI extremes + Bollinger Band position

9. MARKET REGIME ADAPTATION:
   - Volatility Regime: In HIGH volatility, prioritize capital preservation, tighten stops, reduce position sizes. In LOW volatility, you can be more aggressive but maintain discipline.
   - Trend Regime: In TRENDING markets, favor momentum strategies (trend following). In MEAN-REVERTING markets, favor contrarian strategies (buy oversold, sell overbought).
   - Correlation Regime: In HIGH correlation environments, diversification benefits are limited — reduce overall equity exposure. In LOW correlation, opportunities for risk reduction through diversification are good.
   - The Market Regime Analysis section provides specific recommendations — follow them unless you have strong evidence to the contrary.

OUTPUT FORMAT:
Respond with a JSON object containing:
{
  "actions": [
    {"ticker": "SPY", "action": "buy", "pct": 15},
    {"ticker": "MC.PA", "action": "sell", "pct": 100},
    {"ticker": "GLD", "action": "hold"}
  ],
  "reasoning": "Brief explanation of your decision process, referencing specific indicators and risk considerations. Mention if you're applying loss aversion, CVaR thinking, or DSR skepticism."
}

ACTIONS:
- "buy" with "pct": percentage of available cash to deploy
- "sell" with "pct": percentage of position to sell (use 100 for full exit)
- "hold": no action

Remember: You are risk-aware, not risk-seeking. Preserve capital first, grow second. Question every "too good to be true" signal."""


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
        history_file: str = "data/decision_history.json",
        max_retries: Optional[int] = None,
        retry_backoff_factor: Optional[float] = None,
        retry_jitter: Optional[float] = None
    ):
        self.api_key = api_key or os.getenv("LLM_API_KEY")
        self.api_url = api_url or os.getenv("LLM_API_URL", "https://api.openai.com/v1/chat/completions")
        self.model = model or os.getenv("LLM_MODEL", "kimi")
        self.history_file = Path(history_file)
        self.history_file.parent.mkdir(exist_ok=True)
        self.max_retries = max_retries if max_retries is not None else int(os.getenv("LLM_MAX_RETRIES", "3"))
        self.retry_backoff_factor = retry_backoff_factor if retry_backoff_factor is not None else float(os.getenv("LLM_RETRY_BACKOFF_FACTOR", "1.0"))
        self.retry_jitter = retry_jitter if retry_jitter is not None else float(os.getenv("LLM_RETRY_JITTER", "0.0"))

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
        recent_decisions: Optional[List[Dict]] = None,
        cooldown_status: Optional[Dict] = None
    ) -> str:
        """
        Build the complete prompt for the LLM.
        
        Args:
            market_data: Dict with asset indicators and correlations
            portfolio_summary: Current portfolio state
            recent_decisions: List of recent trading decisions
            cooldown_status: Dict with position cooldown guardrail status
        
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
        has_correlations = False
        if hasattr(correlations, 'empty'):
            has_correlations = not correlations.empty
        elif isinstance(correlations, dict):
            has_correlations = len(correlations) > 0
        
        if has_correlations:
            prompt_parts.append("\n=== CORRELATIONS (20-day returns) ===")
            if hasattr(correlations, 'to_string'):
                prompt_parts.append(correlations.to_string())
            elif isinstance(correlations, dict):
                for pair, corr in correlations.items():
                    prompt_parts.append(f"  {pair}: {corr:.3f}")
        
        # Market Regime Analysis
        regime = market_data.get('regime')
        if regime and regime.get('formatted'):
            prompt_parts.append(regime['formatted'])
        
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
        
        # Deflated Sharpe Ratio analysis
        if DSR_MODULE_AVAILABLE and 'historical_prices' in market_data:
            try:
                prices_dict = market_data['historical_prices']
                weights = {pos['ticker']: pos['market_value'] for pos in positions}
                
                # Calculate portfolio returns
                portfolio_returns = self._calculate_portfolio_returns(prices_dict, weights)
                
                if len(portfolio_returns) >= 30:  # Need sufficient data
                    dsr_calc = DeflatedSharpeRatio(n_trials=10)  # Assume 10 strategies tested
                    dsr_metrics = dsr_calc.calculate(portfolio_returns)
                    
                    prompt_parts.append("\n\n=== DEFLATED SHARPE RATIO ANALYSIS ===")
                    prompt_parts.append(f"Standard Sharpe: {dsr_metrics.sharpe_ratio:.3f}")
                    prompt_parts.append(f"Deflated Sharpe: {dsr_metrics.deflated_sharpe:.3f}")
                    prompt_parts.append(f"Statistical Significance: {'YES' if dsr_metrics.is_significant else 'NO'}")
                    prompt_parts.append(f"Skewness: {dsr_metrics.skewness:.2f} (positive = good)")
                    prompt_parts.append(f"Excess Kurtosis: {dsr_metrics.kurtosis - 3:.2f} (lower = better)")
                    prompt_parts.append(f"Observations: {dsr_metrics.n_observations} days")
                    prompt_parts.append("\nInterpretation:")
                    if not dsr_metrics.is_significant:
                        prompt_parts.append("  ⚠️ Current performance may be due to chance (multiple testing bias)")
                        prompt_parts.append("  → Reduce risk until statistical significance improves")
                    if dsr_metrics.kurtosis > 4:
                        prompt_parts.append("  ⚠️ High kurtosis = fat tails = tail risk")
                        prompt_parts.append("  → Consider reducing position sizes")
                    if dsr_metrics.skewness < 0:
                        prompt_parts.append("  ⚠️ Negative skew = more frequent small gains, rare large losses")
                        prompt_parts.append("  → Typical of trend-following; ensure stop-losses are tight")
            except Exception as e:
                logger.warning(f"Could not calculate DSR metrics: {e}")
        
        # Recent decisions
        if recent_decisions:
            prompt_parts.append("\n\n=== RECENT DECISIONS (last 5 days) ===")
            for d in recent_decisions:
                prompt_parts.append(f"\n{d['timestamp'][:10]}:")
                prompt_parts.append(f"  Reasoning: {d.get('reasoning', 'N/A')[:200]}...")
                actions = d.get('actions', [])
                for a in actions:
                    prompt_parts.append(f"  - {a['ticker']}: {a['action']} {a.get('pct', 0)}%")
        
        # Cooldown guardrails
        if cooldown_status:
            prompt_parts.append("\n\n=== COOLDOWN GUARDRAILS ===")
            prompt_parts.append(f"Weekly trades used: {cooldown_status.get('trades_this_week', 0)}/{cooldown_status.get('weekly_cap', 2)}")
            
            active_entries = cooldown_status.get('active_entries', {})
            if active_entries:
                prompt_parts.append("\nActive positions (holding period):")
                for ticker, info in active_entries.items():
                    hold_days = info.get('hold_days', 0)
                    min_hold = cooldown_status.get('config', {}).get('min_hold_days', 5)
                    status = "✓ can sell" if hold_days >= min_hold else f"✗ hold {min_hold - hold_days:.1f} more days"
                    prompt_parts.append(f"  {ticker}: held {hold_days:.1f} days — {status}")
            
            recent_exits = cooldown_status.get('recent_exits', {})
            if recent_exits:
                prompt_parts.append("\nRecent exits (flip cooldown):")
                for ticker, info in recent_exits.items():
                    days_since = info.get('days_since_exit', 0)
                    flip_days = cooldown_status.get('config', {}).get('flip_cooldown_days', 10)
                    status = "✓ can re-buy" if days_since >= flip_days else f"✗ wait {flip_days - days_since:.1f} more days"
                    prompt_parts.append(f"  {ticker}: exited {days_since:.1f} days ago — {status}")
            
            if cooldown_status.get('trades_this_week', 0) >= cooldown_status.get('weekly_cap', 2):
                prompt_parts.append("\n⚠️ WEEKLY TRADE CAP REACHED — No new buys or sells allowed (except stop-loss overrides).")
                prompt_parts.append("Recommend HOLD for all positions. Do not suggest any new trades.")
            else:
                remaining = cooldown_status.get('weekly_cap', 2) - cooldown_status.get('trades_this_week', 0)
                prompt_parts.append(f"\nYou have {remaining} trade(s) remaining this week. Use them wisely.")
        
        prompt_parts.append("\n\n=== YOUR DECISION ===")
        prompt_parts.append("Based on the above market state and portfolio, what actions should we take?")
        prompt_parts.append("Respond with the JSON format specified in your instructions.")
        
        return "\n".join(prompt_parts)
    
    def _calculate_portfolio_returns(
        self,
        prices_dict: Dict[str, pd.DataFrame],
        weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Calculate portfolio returns from individual asset prices and weights.
        
        Args:
            prices_dict: Dict of ticker -> DataFrame with price data
            weights: Dict of ticker -> weight in portfolio
        
        Returns:
            Array of portfolio daily returns
        """
        import pandas as pd
        
        # Normalize weights to sum to 1
        total_weight = sum(weights.values())
        if total_weight == 0:
            return np.array([])
        
        normalized_weights = {k: v / total_weight for k, v in weights.items()}
        
        # Calculate returns for each asset
        returns_df = pd.DataFrame()
        for ticker, prices in prices_dict.items():
            if ticker in normalized_weights and normalized_weights[ticker] > 0:
                if 'Close' in prices.columns:
                    returns = prices['Close'].pct_change().dropna()
                    returns_df[ticker] = returns
        
        if returns_df.empty:
            return np.array([])
        
        # Align dates and calculate weighted portfolio returns
        returns_df = returns_df.dropna()
        portfolio_returns = pd.Series(0.0, index=returns_df.index)
        
        for ticker in returns_df.columns:
            if ticker in normalized_weights:
                portfolio_returns += returns_df[ticker] * normalized_weights[ticker]
        
        return portfolio_returns.values
    
    def call_llm(self, prompt: str) -> Optional[str]:
        """
        Call the LLM API with exponential-backoff retries for transient errors.

        Retryable conditions: network errors (requests.exceptions.RequestException)
        and HTTP status codes 429 (rate limit), 502, 503, 504 (server errors).
        Non-retryable 4xx errors fail immediately.

        Args:
            prompt: The formatted prompt

        Returns:
            LLM response text or None if all attempts fail
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
            "temperature": float(os.getenv("LLM_TEMPERATURE", "0.3")),
            "max_tokens": 16000,
        }

        last_exception = None
        for attempt in range(self.max_retries + 1):
            try:
                logger.info(f"Calling LLM API (attempt {attempt + 1}/{self.max_retries + 1})...")
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

            except requests.exceptions.HTTPError as e:
                last_exception = e
                status_code = e.response.status_code if e.response is not None else None
                if status_code in (429, 502, 503, 504) and attempt < self.max_retries:
                    wait_time = self.retry_backoff_factor * (2 ** attempt)
                    if self.retry_jitter > 0:
                        wait_time += random.uniform(0, wait_time * self.retry_jitter)
                    logger.warning(
                        f"LLM API returned {status_code} (attempt {attempt + 1}/{self.max_retries + 1}); "
                        f"retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error calling LLM API: {e}")
                return None

            except requests.exceptions.RequestException as e:
                last_exception = e
                if attempt < self.max_retries:
                    wait_time = self.retry_backoff_factor * (2 ** attempt)
                    if self.retry_jitter > 0:
                        wait_time += random.uniform(0, wait_time * self.retry_jitter)
                    logger.warning(
                        f"LLM API request failed (attempt {attempt + 1}/{self.max_retries + 1}): {e}; "
                        f"retrying in {wait_time:.1f}s..."
                    )
                    time.sleep(wait_time)
                    continue
                logger.error(f"Error calling LLM API: {e}")
                return None

            except Exception as e:
                logger.error(f"Error calling LLM API: {e}")
                return None

        # All retries exhausted
        logger.error(f"LLM API failed after {self.max_retries + 1} attempts: {last_exception}")
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
        portfolio_summary: Dict,
        cooldown_status: Optional[Dict] = None
    ) -> Dict:
        """
        Get trading decision from LLM.
        
        Args:
            market_data: Market indicators and correlations
            portfolio_summary: Current portfolio state
            cooldown_status: Optional position cooldown guardrail status
        
        Returns:
            Decision dict with actions and metadata
        """
        # Load recent decisions for context
        recent_decisions = self.load_recent_decisions(days=5)
        
        # Build prompt
        prompt = self.build_prompt(market_data, portfolio_summary, recent_decisions, cooldown_status)
        
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
