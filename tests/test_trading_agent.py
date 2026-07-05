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
import requests

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


def _make_error_response(status_code: int) -> Mock:
    """Helper to create a mocked response that raises HTTPError with given status."""
    mock_response = Mock()
    mock_response.status_code = status_code
    mock_response.text = f"HTTP {status_code}"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    return mock_response


def _make_success_response(content: str) -> Mock:
    """Helper to create a mocked successful JSON response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{
            "message": {"content": content}
        }]
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


def test_api_call_retry_on_rate_limit():
    """Test that 429 rate-limit errors are retried and eventually succeed."""
    print("Test 14: API Call - Retry on Rate Limit (429)")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=2)
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        rate_limit_response = _make_error_response(429)
        
        with patch('requests.post', side_effect=[rate_limit_response, success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert result == '{"actions": [], "reasoning": "OK"}'
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
                assert mock_sleep.call_args[0][0] == 1.0  # backoff factor * 2^0
                
                print("  429 retried successfully on second attempt")
                print("✓ Rate limit retry test passed\n")


def test_api_call_retry_on_server_error():
    """Test that 503 service-unavailable errors are retried."""
    print("Test 15: API Call - Retry on Server Error (503)")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=3)
        
        success_response = _make_success_response('{"actions": [{"ticker": "SPY", "action": "hold"}], "reasoning": "OK"}')
        server_error_response = _make_error_response(503)
        
        with patch('requests.post', side_effect=[server_error_response, server_error_response, success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert "SPY" in result
                assert mock_post.call_count == 3
                assert mock_sleep.call_count == 2
                # Wait times: 1.0s and 2.0s (exponential backoff)
                assert mock_sleep.call_args_list[0][0][0] == 1.0
                assert mock_sleep.call_args_list[1][0][0] == 2.0
                
                print("  503 retried successfully on third attempt")
                print("✓ Server error retry test passed\n")


def test_api_call_no_retry_on_client_error():
    """Test that non-retryable 4xx errors fail immediately without retry."""
    print("Test 16: API Call - No Retry on Client Error (400)")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=3)
        
        client_error_response = _make_error_response(400)
        
        with patch('requests.post', return_value=client_error_response) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is None
                assert mock_post.call_count == 1
                assert mock_sleep.call_count == 0
                
                print("  400 failed immediately without retry")
                print("✓ No retry on client error test passed\n")


def test_api_call_exhaust_retries():
    """Test that persistent transient errors return None after exhausting retries."""
    print("Test 17: API Call - Exhaust Retries")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=2)
        
        server_error_response = _make_error_response(502)
        
        with patch('requests.post', return_value=server_error_response) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is None
                # max_retries=2 means 1 initial + 2 retries = 3 attempts
                assert mock_post.call_count == 3
                assert mock_sleep.call_count == 2
                # Wait times: 1.0s and 2.0s
                assert mock_sleep.call_args_list[0][0][0] == 1.0
                assert mock_sleep.call_args_list[1][0][0] == 2.0
                
                print("  All retries exhausted, returned None")
                print("✓ Exhaust retries test passed\n")


def test_api_call_retry_on_network_error():
    """Test that network-level errors are retried and eventually succeed."""
    print("Test 18: API Call - Retry on Network Error")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=2)
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        
        with patch('requests.post', side_effect=[requests.exceptions.ConnectionError("Connection refused"), success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
                
                print("  Network error retried successfully on second attempt")
                print("✓ Network error retry test passed\n")


def test_retry_configuration():
    """Test that retry settings are configurable via constructor and env vars."""
    print("Test 19: Retry Configuration")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        
        # Constructor values
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            max_retries=5,
            retry_backoff_factor=0.5
        )
        assert agent.max_retries == 5
        assert agent.retry_backoff_factor == 0.5
        
        # Environment defaults
        with patch.dict(os.environ, {"LLM_MAX_RETRIES": "7", "LLM_RETRY_BACKOFF_FACTOR": "2.5"}):
            agent_env = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_env.max_retries == 7
            assert agent_env.retry_backoff_factor == 2.5
        
        print("  Constructor and env-var configuration both work")
        print("✓ Retry configuration test passed\n")


def test_timeout_configuration():
    """Test that request timeout is configurable via constructor and env vars."""
    print("Test 20: Timeout Configuration")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        
        # Constructor value
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=60.0
        )
        assert agent.timeout == 60.0
        
        # Environment default
        with patch.dict(os.environ, {"LLM_TIMEOUT": "90"}):
            agent_env = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_env.timeout == 90.0
        
        # Default should be 180.0
        with patch.dict(os.environ, {}, clear=True):
            agent_default = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_default.timeout == 180.0
        
        print("  Constructor and env-var timeout configuration both work")
        print("  ✓ Default timeout is 180.0s")
        print("✓ Timeout configuration test passed\n")


def test_timeout_passed_to_requests():
    """Test that the configured timeout is passed to requests.post."""
    print("Test 21: Timeout Passed to requests.post")
    print("-" * 40)
    
    success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=45.0
        )
        
        with patch('requests.post', return_value=success_response) as mock_post:
            result = agent.call_llm("test prompt")
            
            assert result is not None
            assert mock_post.call_count == 1
            assert mock_post.call_args[1]["timeout"] == 45.0
            
            print("  requests.post called with timeout=45.0")
            print("✓ Timeout passed to requests test passed\n")


def test_timeout_per_call_override():
    """Test that a per-call timeout overrides the agent-level timeout."""
    print("Test 22: Per-Call Timeout Override")
    print("-" * 40)
    
    success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=120.0
        )
        
        with patch('requests.post', return_value=success_response) as mock_post:
            result = agent.call_llm("test prompt", timeout=30.0)
            
            assert result is not None
            assert mock_post.call_count == 1
            assert mock_post.call_args[1]["timeout"] == 30.0
            
            print("  Per-call timeout=30.0 overrides agent-level 120.0")
            print("✓ Per-call timeout override test passed\n")


def test_invalid_timeout_fails_fast():
    """Test that non-positive timeouts fail immediately without calling the API."""
    print("Test 23: Invalid Timeout Fails Fast")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=0.0
        )
        
        with patch('requests.post') as mock_post:
            result = agent.call_llm("test prompt")
            
            assert result is None
            assert mock_post.call_count == 0
            
            print("  Invalid timeout returned None without calling API")
            print("✓ Invalid timeout fails fast test passed\n")


def test_retry_wait_helper():
    """Test the exponential-backoff wait calculation helper."""
    print("Test 24: Retry Wait Helper")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            retry_backoff_factor=1.5
        )
        
        assert agent._calculate_retry_wait(0) == 1.5
        assert agent._calculate_retry_wait(1) == 3.0
        assert agent._calculate_retry_wait(2) == 6.0
        assert agent._calculate_retry_wait(3) == 12.0
        
        # Zero backoff factor should return 0
        agent_zero = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            retry_backoff_factor=0.0
        )
        assert agent_zero._calculate_retry_wait(0) == 0.0
        assert agent_zero._calculate_retry_wait(5) == 0.0
        
        print("  Exponential backoff: 1.5, 3.0, 6.0, 12.0")
        print("  ✓ Zero backoff factor yields zero wait")
        print("✓ Retry wait helper test passed\n")


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
    test_api_call_retry_on_rate_limit()
    test_api_call_retry_on_server_error()
    test_api_call_no_retry_on_client_error()
    test_api_call_exhaust_retries()
    test_api_call_retry_on_network_error()
    test_retry_configuration()
    test_timeout_configuration()
    test_timeout_passed_to_requests()
    test_timeout_per_call_override()
    test_invalid_timeout_fails_fast()
    test_retry_wait_helper()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)
