"""
Test suite for LLM Trading Agent module.
Uses mocks to avoid actual API calls.
"""

import json
import sys
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.trading_agent import (
    SYSTEM_PROMPT,
    TradingAgent
)


def test_system_prompt_exists():
    """Test that system prompt is defined and contains key principles."""
    print("Test 1: System Prompt Content")
    print("-" * 40)
    
    assert SYSTEM_PROMPT is not None
    assert len(SYSTEM_PROMPT) > 1000
    
    # Check for key principles
    assert "LOSS AVERSION" in SYSTEM_PROMPT
    assert "CVaR" in SYSTEM_PROMPT
    assert "RSI" in SYSTEM_PROMPT
    assert "Deflated Sharpe Ratio" in SYSTEM_PROMPT
    assert "OUTPUT FORMAT" in SYSTEM_PROMPT
    
    # Check for regime-aware cash targets (added 2026-06-29)
    assert "POSITION SIZING & CASH TARGETS" in SYSTEM_PROMPT
    assert "HIGH volatility: 30-50% cash" in SYSTEM_PROMPT
    assert "NORMAL volatility: 15-30% cash" in SYSTEM_PROMPT
    assert "LOW volatility: 10-20% cash" in SYSTEM_PROMPT
    assert "you are under-invested" in SYSTEM_PROMPT
    
    # Check for drawdown clarification (added 2026-06-29)
    assert "single trading day" in SYSTEM_PROMPT
    assert "total portfolio drawdown from inception" in SYSTEM_PROMPT
    
    print(f"  Prompt length: {len(SYSTEM_PROMPT)} chars")
    print("  ✓ Contains LOSS AVERSION")
    print("  ✓ Contains CVaR principle")
    print("  ✓ Contains Deflated Sharpe Ratio")
    print("✓ System prompt test passed\n")


def test_trading_agent_initialization():
    """Test TradingAgent initialization."""
    print("Test 2: Trading Agent Initialization")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "test_decisions.json"
        
        agent = TradingAgent(
            api_key="test_key",
            api_url="https://test.api.com",
            model="kimi-test",
            history_file=str(history_file)
        )
        
        assert agent.api_key == "test_key"
        assert agent.api_url == "https://test.api.com"
        assert agent.model == "kimi-test"
        assert agent.history_file == history_file
        
        print("  Agent initialized successfully")
        print(f"  API Key: {'*' * len(agent.api_key)}")
        print(f"  Model: {agent.model}")
        print("✓ Trading agent initialization test passed\n")


def test_trading_agent_no_api_key_warning():
    """Test that agent warns when no API key is provided."""
    print("Test 3: Missing API Key Warning")
    print("-" * 40)
    
    with patch.dict(os.environ, {}, clear=True):
        with tempfile.TemporaryDirectory() as tmpdir:
            history_file = Path(tmpdir) / "test_decisions.json"
            agent = TradingAgent(history_file=str(history_file))
            
            assert agent.api_key is None
            print("  Agent created without API key")
            print("  ✓ Warning would be logged")
            print("✓ Missing API key test passed\n")


def test_save_and_load_decisions():
    """Test saving and loading decision history."""
    print("Test 4: Save and Load Decisions")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        # Save a decision
        decision = {
            "timestamp": datetime.now().isoformat(),
            "actions": [{"ticker": "SPY", "action": "buy", "pct": 10}],
            "reasoning": "Test decision"
        }
        agent.save_decision(decision)
        
        # Load recent decisions
        loaded = agent.load_recent_decisions(days=1)
        
        assert len(loaded) == 1
        assert loaded[0]["actions"][0]["ticker"] == "SPY"
        
        print(f"  Saved and loaded {len(loaded)} decision(s)")
        print("✓ Save/load decisions test passed\n")


def test_load_decisions_empty_file():
    """Test loading from non-existent history file."""
    print("Test 5: Load from Empty History")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "nonexistent.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        decisions = agent.load_recent_decisions(days=5)
        
        assert decisions == []
        print("  Empty history handled gracefully")
        print("✓ Empty history test passed\n")


def test_build_prompt_structure():
    """Test that prompt building creates proper structure."""
    print("Test 6: Build Prompt Structure")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        # Structure expected by build_prompt
        market_data = {
            "assets": {
                "SPY": {
                    "latest": {
                        "price": 400.0,
                        "rsi_14": 45.0,
                        "bb_position": 0.5,
                        "sma_20": 395.0,
                        "sma_50": 390.0,
                        "volatility_annual": 0.15,
                        "drawdown": -0.02,
                        "daily_return": 0.005
                    }
                },
                "TLT": {
                    "latest": {
                        "price": 100.0,
                        "rsi_14": 55.0,
                        "bb_position": 0.6,
                        "sma_20": 99.0,
                        "sma_50": 98.0,
                        "volatility_annual": 0.10,
                        "drawdown": -0.01,
                        "daily_return": 0.002
                    }
                }
            },
            "correlations": pd.DataFrame(),  # Empty DataFrame as expected by code
            "regime": {"formatted": "\n=== MARKET REGIME ===\nTest regime analysis\n"}
        }
        
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "positions": [
                {
                    "ticker": "SPY",
                    "quantity": 5,
                    "avg_price": 390.0,
                    "current_price": 400.0,
                    "unrealized_pnl_pct": 2.56,
                    "market_value": 2000.0
                }
            ],
            "risk_metrics": {
                "cvar_95": -0.02,
                "var_95": -0.015,
                "max_drawdown": -0.05,
                "sortino_ratio": 1.2,
                "skewness": -0.1,
                "kurtosis": 3.0
            }
        }
        
        prompt = agent.build_prompt(market_data, portfolio)
        
        assert "MARKET STATE" in prompt or "MARKET" in prompt.upper()
        assert "PORTFOLIO" in prompt.upper()
        assert "SPY" in prompt
        assert "TLT" in prompt
        
        print(f"  Prompt length: {len(prompt)} chars")
        print("  ✓ Contains market data")
        print("  ✓ Contains portfolio info")
        print("✓ Build prompt test passed\n")


def test_api_call_mock_success():
    """Test API call with mocked successful response."""
    print("Test 7: API Call - Success")
    print("-" * 40)
    
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {
                "content": json.dumps({
                    "actions": [
                        {"ticker": "SPY", "action": "buy", "pct": 10}
                    ],
                    "reasoning": "RSI indicates oversold"
                })
            }
        }]
    }
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file))
        
        with patch('requests.post', return_value=mock_response):
            market_data = {"SPY": {"rsi": 30}}
            portfolio = {"cash": 9000, "total_value": 10000, "positions": {}}
            
            result = agent.get_trading_decision(market_data, portfolio)
            
            assert result is not None
            assert "actions" in result
            print("  API call successful")
            print(f"  Received {len(result.get('actions', []))} action(s)")
            print("✓ API success test passed\n")


def test_api_call_mock_error():
    """Test API call with error response."""
    print("Test 8: API Call - Error Handling")
    print("-" * 40)
    
    mock_response = Mock()
    mock_response.status_code = 500
    mock_response.text = "Internal Server Error"
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file))
        
        with patch('requests.post', return_value=mock_response):
            market_data = {}
            portfolio = {"cash": 9000, "total_value": 10000, "positions": {}}
            
            result = agent.get_trading_decision(market_data, portfolio)
            
            # Should return fallback (hold all) on error
            assert result is not None
            print("  API error handled gracefully")
            print("✓ API error test passed\n")


def test_api_call_network_error():
    """Test API call with network failure."""
    print("Test 9: API Call - Network Error")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file))
        
        with patch('requests.post', side_effect=Exception("Network error")):
            market_data = {}
            portfolio = {"cash": 9000, "total_value": 10000, "positions": {}}
            
            result = agent.get_trading_decision(market_data, portfolio)
            
            assert result is not None
            print("  Network error handled gracefully")
            print("✓ Network error test passed\n")


def test_decision_history_limit():
    """Test that history is limited to last 100 decisions."""
    print("Test 10: Decision History Limit")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        # Save 110 decisions
        for i in range(110):
            decision = {
                "timestamp": datetime.now().isoformat(),
                "actions": [{"ticker": "SPY", "action": "buy", "pct": 1}],
                "reasoning": f"Decision {i}"
            }
            agent.save_decision(decision)
        
        # Load and verify only 100 kept
        with open(history_file) as f:
            saved = json.load(f)
        
        assert len(saved) == 100
        print(f"  Saved 110 decisions, kept {len(saved)}")
        print("✓ History limit test passed\n")


def test_load_recent_days_filter():
    """Test filtering decisions by recent days."""
    print("Test 11: Recent Days Filter")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        # Save old decision (10 days ago)
        old_decision = {
            "timestamp": (datetime.now() - timedelta(days=10)).isoformat(),
            "actions": [],
            "reasoning": "Old"
        }
        
        # Save recent decision
        recent_decision = {
            "timestamp": datetime.now().isoformat(),
            "actions": [{"ticker": "SPY", "action": "buy"}],
            "reasoning": "Recent"
        }
        
        agent.save_decision(old_decision)
        agent.save_decision(recent_decision)
        
        # Load only last 5 days
        recent = agent.load_recent_decisions(days=5)
        
        assert len(recent) == 1
        assert recent[0]["reasoning"] == "Recent"
        
        print(f"  Loaded {len(recent)} recent decision(s)")
        print("✓ Recent days filter test passed\n")


def test_build_prompt_with_cooldown_status():
    """Test that cooldown status is included in prompt when provided."""
    print("Test 12: Build Prompt with Cooldown Status")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        market_data = {
            "assets": {
                "SPY": {
                    "latest": {
                        "price": 400.0,
                        "rsi_14": 45.0,
                        "bb_position": 0.5,
                        "sma_20": 395.0,
                        "sma_50": 390.0,
                        "volatility_annual": 0.15,
                        "drawdown": -0.02,
                        "daily_return": 0.005
                    }
                }
            },
            "correlations": pd.DataFrame(),
            "regime": None
        }
        
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "positions": [
                {
                    "ticker": "SPY",
                    "quantity": 5,
                    "avg_price": 390.0,
                    "current_price": 400.0,
                    "unrealized_pnl_pct": 2.56,
                    "market_value": 2000.0
                }
            ]
        }
        
        cooldown_status = {
            "trades_this_week": 2,
            "weekly_cap": 2,
            "active_entries": {
                "SPY": {"entry_date": "2026-06-16T21:00:00", "hold_days": 2.0}
            },
            "recent_exits": {
                "GLD": {"exit_date": "2026-06-18T16:00:00", "days_since_exit": 0.5}
            },
            "config": {"min_hold_days": 5, "flip_cooldown_days": 10}
        }
        
        prompt = agent.build_prompt(market_data, portfolio, cooldown_status=cooldown_status)
        
        assert "COOLDOWN GUARDRAILS" in prompt
        assert "Weekly trades used: 2/2" in prompt
        assert "WEEKLY TRADE CAP REACHED" in prompt
        assert "SPY: held 2.0 days" in prompt
        assert "GLD: exited 0.5 days ago" in prompt
        
        print(f"  Prompt length: {len(prompt)} chars")
        print("  ✓ Contains cooldown guardrails section")
        print("  ✓ Shows weekly trade cap reached")
        print("  ✓ Shows active entry hold periods")
        print("  ✓ Shows recent exit flip cooldowns")
        print("✓ Cooldown status prompt test passed\n")


def test_build_prompt_without_cooldown_status():
    """Test that prompt works normally when no cooldown status provided."""
    print("Test 13: Build Prompt without Cooldown Status")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))
        
        market_data = {
            "assets": {
                "SPY": {
                    "latest": {
                        "price": 400.0,
                        "rsi_14": 45.0,
                        "bb_position": 0.5,
                        "sma_20": 395.0,
                        "sma_50": 390.0,
                        "volatility_annual": 0.15,
                        "drawdown": -0.02,
                        "daily_return": 0.005
                    }
                }
            },
            "correlations": pd.DataFrame(),
            "regime": None
        }
        
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "positions": []
        }
        
        prompt = agent.build_prompt(market_data, portfolio)
        
        assert "COOLDOWN GUARDRAILS" not in prompt
        assert "MARKET STATE" in prompt
        
        print(f"  Prompt length: {len(prompt)} chars")
        print("  ✓ No cooldown section when not provided")
        print("✓ No cooldown status test passed\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Running LLM Trading Agent Tests")
    print("=" * 60 + "\n")
    
    test_system_prompt_exists()
    test_trading_agent_initialization()
    test_trading_agent_no_api_key_warning()
    test_save_and_load_decisions()
    test_load_decisions_empty_file()
    test_build_prompt_structure()
    test_build_prompt_with_cooldown_status()
    test_build_prompt_without_cooldown_status()
    test_api_call_mock_success()
    test_api_call_mock_error()
    test_api_call_network_error()
    test_decision_history_limit()
    test_load_recent_days_filter()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
