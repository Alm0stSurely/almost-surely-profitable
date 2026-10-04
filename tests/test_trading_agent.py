"""
Test suite for LLM Trading Agent module.
Uses mocks to avoid actual API calls.
"""

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pandas as pd
import pytest
import requests

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import llm.trading_agent as trading_agent_module
from llm.trading_agent import SYSTEM_PROMPT, TradingAgent


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
    assert "capital is being dragged" in SYSTEM_PROMPT
    assert "Default to deploying available cash" in SYSTEM_PROMPT
    
    # Check for drawdown clarification (added 2026-06-29)
    assert "single trading day" in SYSTEM_PROMPT
    assert "total portfolio drawdown from inception" in SYSTEM_PROMPT
    
    # Check for sell discipline / let winners run (added 2026-07-09)
    assert "SELL DISCIPLINE" in SYSTEM_PROMPT
    assert "LET WINNERS RUN" in SYSTEM_PROMPT
    assert "Do NOT sell a position merely because it is showing a small profit" in SYSTEM_PROMPT
    assert "confirmed technical reversal" in SYSTEM_PROMPT
    assert "When in doubt, default to HOLD" in SYSTEM_PROMPT

    # Check for stop-override policy (added 2026-09-25, TLT conflict formalization)
    assert "STOP-OVERRIDE POLICY" in SYSTEM_PROMPT
    assert "breached stop" in SYSTEM_PROMPT
    assert "hard exit threshold" in SYSTEM_PROMPT
    assert "re-justified at every daily session" in SYSTEM_PROMPT
    assert "never widen an existing override threshold" in SYSTEM_PROMPT
    # Anti-drift anchoring (added 2026-09-28, TLT -7% -> -8% threshold drift under first policy-era override)
    # NB: assert within-line fragments — the prompt literal wraps lines, so a phrase
    # spanning a line break + indentation is not a contiguous substring.
    assert "at least as tight as the" in SYSTEM_PROMPT
    assert "tightest level previously named" in SYSTEM_PROMPT
    assert "never loosen" in SYSTEM_PROMPT

    print(f"  Prompt length: {len(SYSTEM_PROMPT)} chars")
    print("  ✓ Contains LOSS AVERSION")
    print("  ✓ Contains CVaR principle")
    print("  ✓ Contains Deflated Sharpe Ratio")
    print("  ✓ Contains SELL DISCIPLINE / LET WINNERS RUN")
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


def test_save_decision_sanitizes_non_finite_values():
    """Test that non-finite floats in a decision are sanitized before saving.

    ``json.loads`` accepts non-standard ``NaN``/``Infinity`` tokens by default.
    If a parsed LLM response contained such values, the default ``json.dump``
    would persist them and break strict downstream consumers. The safe
    serializer must replace them with ``null``.
    """
    print("Test 4b: Save Decision Sanitizes Non-Finite Values")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        decision = {
            "timestamp": datetime.now().isoformat(),
            "actions": [
                {"ticker": "SPY", "action": "buy", "pct": float("nan")},
                {"ticker": "TLT", "action": "sell", "pct": float("inf")},
            ],
            "reasoning": "Degenerate LLM output",
        }
        agent.save_decision(decision)

        raw_text = history_file.read_text()
        # Strict JSON must not contain non-standard tokens
        assert "NaN" not in raw_text
        assert "Infinity" not in raw_text

        loaded = json.loads(raw_text)
        assert len(loaded) == 1
        assert loaded[0]["actions"][0]["pct"] is None
        assert loaded[0]["actions"][1]["pct"] is None

        # Loading through the agent should also work
        recent = agent.load_recent_decisions(days=1)
        assert recent[0]["actions"][0]["pct"] is None

        print("  Non-finite action percentages sanitized to null")
        print("✓ Save decision sanitization test passed\n")


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
    """Test that history is bounded but keeps well over a year of trading days."""
    print("Test 10: Decision History Limit")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        # Save 510 decisions
        for i in range(510):
            decision = {
                "timestamp": datetime.now().isoformat(),
                "actions": [{"ticker": "SPY", "action": "buy", "pct": 1}],
                "reasoning": f"Decision {i}"
            }
            agent.save_decision(decision)

        # Load and verify only the bound is kept
        with open(history_file) as f:
            saved = json.load(f)

        assert len(saved) == 500
        # The bound must exceed one year of trading days (~260) so research
        # analyses never silently lose history (regression: the previous
        # bound of 100 discarded decisions within ~5 months).
        assert len(saved) > 260
        print(f"  Saved 510 decisions, kept {len(saved)}")
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




def test_jitter_configuration():
    """Test that retry jitter is configurable via constructor and env vars."""
    print("Test 20: Jitter Configuration")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        
        # Constructor value
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            retry_jitter=0.25
        )
        assert agent.retry_jitter == 0.25
        
        # Environment default
        with patch.dict(os.environ, {"LLM_RETRY_JITTER": "0.5"}):
            agent_env = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_env.retry_jitter == 0.5
        
        # Default should be 0.0 (no jitter)
        with patch.dict(os.environ, {}, clear=True):
            agent_default = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_default.retry_jitter == 0.0
        
        print("  Constructor and env-var jitter configuration both work")
        print("  ✓ Default jitter is 0.0 (backward-compatible)")
        print("✓ Jitter configuration test passed\n")


def test_jitter_applied_on_http_retry():
    """Jitter increases wait time by up to the configured fraction."""
    print("Test 21: Jitter Applied on HTTP Retry")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            max_retries=1,
            retry_backoff_factor=1.0,
            retry_jitter=0.25
        )
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        rate_limit_response = _make_error_response(429)
        
        with patch('requests.post', side_effect=[rate_limit_response, success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
                
                waited = mock_sleep.call_args[0][0]
                # Base wait is 1.0s; jitter adds up to 0.25s
                assert 1.0 <= waited <= 1.25
                
                print(f"  Waited {waited:.3f}s (base 1.0s + jitter up to 25%)")
                print("✓ Jitter applied on HTTP retry test passed\n")


def test_jitter_applied_on_network_error():
    """Jitter is also applied on network-level retryable errors."""
    print("Test 22: Jitter Applied on Network Error")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            max_retries=1,
            retry_backoff_factor=2.0,
            retry_jitter=0.5
        )
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        
        with patch('requests.post', side_effect=[requests.exceptions.ConnectionError("Connection refused"), success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1
                
                waited = mock_sleep.call_args[0][0]
                # Base wait is 2.0s; jitter adds up to 1.0s
                assert 2.0 <= waited <= 3.0
                
                print(f"  Waited {waited:.3f}s (base 2.0s + jitter up to 50%)")
                print("✓ Jitter applied on network error test passed\n")


def test_no_jitter_exact_backoff():
    """With jitter disabled, backoff remains exactly exponential."""
    print("Test 23: No Jitter - Exact Exponential Backoff")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            max_retries=2,
            retry_backoff_factor=1.0,
            retry_jitter=0.0
        )
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        server_error_response = _make_error_response(503)
        
        with patch('requests.post', side_effect=[server_error_response, server_error_response, success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert mock_sleep.call_count == 2
                assert mock_sleep.call_args_list[0][0][0] == 1.0
                assert mock_sleep.call_args_list[1][0][0] == 2.0
                
                print("  Wait times exactly 1.0s and 2.0s with no jitter")
                print("✓ No jitter exact backoff test passed\n")


def _make_malformed_body_response() -> Mock:
    """Helper: 200 response whose body fails JSON decoding (truncated/garbage body)."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.side_effect = requests.exceptions.JSONDecodeError(
        "Expecting value", "<html>Gateway timeout</html>", 0
    )
    mock_response.raise_for_status.return_value = None
    return mock_response


def _make_envelope_response(envelope: dict) -> Mock:
    """Helper: 200 response with a valid-JSON envelope of arbitrary (possibly wrong) shape."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = envelope
    mock_response.raise_for_status.return_value = None
    return mock_response


def test_api_call_retry_on_malformed_json_body():
    """A 200 with an undecodable body is a transient transport failure: retried, then succeeds.

    A truncated/garbage body typically comes from a proxy or gateway that answered
    200 while the real payload never arrived. requests raises JSONDecodeError,
    which is a RequestException subclass, so the existing backoff retry applies.
    This pins the exact interaction with the parse-level degradation of the
    downstream parse path: a malformed *body* never reaches parse_response.
    """
    print("Test 23b: API Call - Retry on Malformed JSON Body")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=2)

        malformed_response = _make_malformed_body_response()
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')

        with patch('requests.post', side_effect=[malformed_response, success_response]) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")

                assert result is not None
                assert result == '{"actions": [], "reasoning": "OK"}'
                assert mock_post.call_count == 2
                assert mock_sleep.call_count == 1

                print("  Malformed body retried successfully on second attempt")
                print("✓ Malformed JSON body retry test passed\n")


def test_api_call_no_retry_on_envelope_shape_failure():
    """Valid JSON with a wrong shape (missing 'choices') fails after ONE attempt.

    Retrying a well-formed 200 whose envelope lacks the expected keys is
    unlikely to help — the provider will return the same shape. The failure
    must be cheap (no backoff loop) and loud (returns None, caller holds all).
    """
    print("Test 23c: API Call - No Retry on Envelope Shape Failure")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=3)

        wrong_shape = _make_envelope_response({"usage": {"total_tokens": 42}})

        with patch('requests.post', return_value=wrong_shape) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")

                assert result is None
                assert mock_post.call_count == 1
                assert mock_sleep.call_count == 0

                print("  Wrong-shape envelope failed immediately without retry")
                print("✓ No retry on envelope shape failure test passed\n")


def test_api_call_empty_choices_list_fails_single_attempt():
    """An empty choices list is a shape failure, not a transient one: single attempt, None."""
    print("Test 23d: API Call - Empty Choices List Fails Single Attempt")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=2)

        empty_choices = _make_envelope_response({"choices": []})

        with patch('requests.post', return_value=empty_choices) as mock_post:
            with patch('time.sleep', return_value=None) as mock_sleep:
                result = agent.call_llm("test prompt")

                assert result is None
                assert mock_post.call_count == 1
                assert mock_sleep.call_count == 0

                print("  Empty choices list returned None after one attempt")
                print("✓ Empty choices list test passed\n")


def test_api_call_reasoning_content_fallback():
    """Kimi may leave content empty and put the answer in reasoning_content."""
    print("Test 23e: API Call - Reasoning Content Fallback")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file))

        envelope = _make_envelope_response({
            "choices": [{
                "message": {
                    "content": "",
                    "reasoning_content": '{"actions": [], "reasoning": "from-reasoning"}'
                }
            }]
        })

        with patch('requests.post', return_value=envelope):
            result = agent.call_llm("test prompt")

            assert result == '{"actions": [], "reasoning": "from-reasoning"}'

            print("  Empty content recovered from reasoning_content")
            print("✓ Reasoning content fallback test passed\n")


def test_api_call_empty_content_holds_all_downstream():
    """A 200 with an empty message dict degrades to a hold-all decision downstream.

    End-to-end pin across the call_llm -> parse_response boundary (the PR #64
    interaction): empty content is returned as "" and parse_response maps it to
    error=True with zero actions — the global-absence fallback, never a crash
    and never a partial decision.
    """
    print("Test 23f: API Call - Empty Content Holds All Downstream")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file))

        empty_message = _make_envelope_response({"choices": [{"message": {}}]})

        with patch('requests.post', return_value=empty_message):
            content = agent.call_llm("test prompt")

        assert content == ""
        parsed = agent.parse_response(content)
        assert parsed.get("error") is True
        assert parsed.get("actions") == []

        print("  Empty content -> parse_response -> hold-all with error flag")
        print("✓ Empty content downstream hold-all test passed\n")


def test_api_call_retry_on_http_500_then_success():
    """A transient HTTP 500 is retried with backoff, then succeeds.

    Queued policy fix from the PR #66 audit notes: 500 was the only
    server-side transient status outside the retry set. Old behavior: a
    single 500 returned None immediately (the day's decision lost to one
    server hiccup). New behavior: same trajectory class as 502/503/504.
    """
    print("Test 23g: API Call - Retry on HTTP 500 Then Success")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file))

        responses = [
            _make_error_response(500),
            _make_success_response('{"actions": []}'),
        ]

        with patch('requests.post', side_effect=responses), patch('time.sleep', return_value=None):
            content = agent.call_llm("test prompt")

        assert content == '{"actions": []}'
        print("  500 retried, succeeded on second attempt")
        print("✓ HTTP 500 transient retry test passed\n")


def test_api_call_persistent_500_exhausts_retries():
    """A persistent HTTP 500 burns the full retry budget, then returns None.

    Pins the bounded side of the policy: retrying does not help a
    deterministic server fault, but the cost is capped at max_retries extra
    attempts and the terminal state (None -> hold-all downstream) is
    unchanged.
    """
    print("Test 23h: API Call - Persistent 500 Exhausts Retries")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test_key", history_file=str(history_file), max_retries=2)

        with patch('requests.post', side_effect=lambda *a, **k: _make_error_response(500)), \
             patch('time.sleep', return_value=None):
            content = agent.call_llm("test prompt")

        assert content is None
        print("  Persistent 500 exhausted retries and returned None")
        print("✓ Persistent 500 exhaustion test passed\n")


def test_timeout_configuration():
    """Test that request timeout is configurable via constructor and env vars."""
    print("Test 24: Timeout Configuration")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        
        # Constructor value
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=45.0
        )
        assert agent.timeout == 45.0
        
        # Environment default
        with patch.dict(os.environ, {"LLM_TIMEOUT": "300"}):
            agent_env = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_env.timeout == 300.0
        
        # Default should be 180.0
        with patch.dict(os.environ, {}, clear=True):
            agent_default = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent_default.timeout == 180.0
        
        print("  Constructor and env-var timeout configuration both work")
        print("  ✓ Default timeout is 180.0s (backward-compatible)")
        print("✓ Timeout configuration test passed\n")


def test_call_llm_uses_configured_timeout():
    """The configured agent timeout is passed to requests.post."""
    print("Test 25: call_llm Uses Configured Timeout")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=45.0
        )
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        
        with patch('requests.post', return_value=success_response) as mock_post:
            result = agent.call_llm("test prompt")
            
            assert result is not None
            assert mock_post.call_count == 1
            assert mock_post.call_args[1].get("timeout") == 45.0
            
            print("  requests.post received timeout=45.0")
            print("✓ call_llm uses configured timeout test passed\n")


def test_call_llm_timeout_override_argument():
    """A per-call timeout argument overrides the agent's default timeout."""
    print("Test 26: call_llm Timeout Override Argument")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(
            api_key="test_key",
            history_file=str(history_file),
            timeout=45.0
        )
        
        success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
        
        with patch('requests.post', return_value=success_response) as mock_post:
            result = agent.call_llm("test prompt", timeout=120.0)
            
            assert result is not None
            assert mock_post.call_count == 1
            assert mock_post.call_args[1].get("timeout") == 120.0
            
            print("  Per-call timeout=120.0 overrode configured timeout=45.0")
            print("✓ call_llm timeout override test passed\n")


def test_call_llm_env_timeout_used():
    """An LLM_TIMEOUT env var is used when no constructor timeout is provided."""
    print("Test 27: call_llm Env Timeout")
    print("-" * 40)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        
        with patch.dict(os.environ, {"LLM_TIMEOUT": "240"}):
            agent = TradingAgent(api_key="test_key", history_file=str(history_file))
            assert agent.timeout == 240.0
            
            success_response = _make_success_response('{"actions": [], "reasoning": "OK"}')
            
            with patch('requests.post', return_value=success_response) as mock_post:
                result = agent.call_llm("test prompt")
                
                assert result is not None
                assert mock_post.call_count == 1
                assert mock_post.call_args[1].get("timeout") == 240.0
                
                print("  Env timeout=240.0 applied to requests.post")
                print("✓ call_llm env timeout test passed\n")


def test_call_llm_invalid_timeout_returns_none():
    """Non-positive timeouts are rejected immediately without making a request."""
    print("Test 28: call_llm Invalid Timeout")
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
            
            print("  Non-positive timeout rejected without API call")
            print("✓ Invalid timeout test passed\n")


def test_safe_format_renders_non_finite_as_na():
    """_safe_format falls back to 'n/a' for non-finite or non-numeric inputs."""
    from llm.trading_agent import _safe_format

    assert _safe_format(1.2345, ".2f") == "1.23"
    assert _safe_format(float("nan"), ".2f") == "n/a"
    assert _safe_format(float("inf"), ".2f") == "n/a"
    assert _safe_format(float("-inf"), ".2f") == "n/a"
    assert _safe_format(None, ".2f") == "n/a"
    assert _safe_format("abc", ".2f") == "n/a"
    assert _safe_format("1.23", ".2f") == "1.23"


def test_build_prompt_non_finite_asset_fields():
    """Non-finite asset indicator values are rendered as n/a in the prompt."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {
            "assets": {
                "SPY": {
                    "latest": {
                        "price": float("nan"),
                        "rsi_14": float("inf"),
                        "bb_position": float("-inf"),
                        "sma_20": float("nan"),
                        "sma_50": float("nan"),
                        "volatility_annual": float("nan"),
                        "drawdown": float("inf"),
                        "daily_return": float("-inf"),
                    }
                }
            },
            "correlations": pd.DataFrame(),
            "regime": None,
        }
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "total_return_pct": 0.0,
            "total_pnl": 0.0,
            "positions": [],
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Price: €n/a" in prompt
        assert "RSI(14): n/a" in prompt
        assert "Bollinger Position: n/a" in prompt
        assert "Volatility (ann): n/a%" in prompt
        assert "Drawdown: n/a%" in prompt
        assert "Daily Return: n/a%" in prompt


def test_build_prompt_non_finite_portfolio_and_positions():
    """Non-finite portfolio totals and position fields are rendered as n/a."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": float("nan"),
            "total_value": float("inf"),
            "total_return_pct": float("-inf"),
            "total_pnl": float("nan"),
            "positions": [
                {
                    "ticker": "SPY",
                    "quantity": float("nan"),
                    "avg_price": float("inf"),
                    "current_price": float("-inf"),
                    "unrealized_pnl_pct": float("nan"),
                    "market_value": 2000.0,
                }
            ],
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Cash: €n/a" in prompt
        assert "Total Value: €n/a" in prompt
        assert "Total Return: n/a%" in prompt
        assert "Total P&L: €n/a" in prompt
        assert "SPY: n/a shares" in prompt
        assert "avg €n/a" in prompt
        assert "current €n/a" in prompt
        assert "P&L n/a%" in prompt


def test_build_prompt_non_finite_cooldown_status():
    """Non-finite cooldown counters and day values are rendered as n/a."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {"cash": 8000.0, "total_value": 10000.0, "positions": []}
        cooldown_status = {
            "trades_this_week": float("nan"),
            "weekly_cap": float("inf"),
            "active_entries": {
                "SPY": {"entry_date": "2026-06-16T21:00:00", "hold_days": float("nan")}
            },
            "recent_exits": {
                "GLD": {"exit_date": "2026-06-18T16:00:00", "days_since_exit": float("inf")}
            },
            "config": {"min_hold_days": 5, "flip_cooldown_days": 10},
        }

        prompt = agent.build_prompt(market_data, portfolio, cooldown_status=cooldown_status)

        assert "Weekly trades used: n/a/n/a" in prompt
        assert "SPY: held n/a days" in prompt
        assert "GLD: exited n/a days ago" in prompt
        assert "Weekly trade status unavailable." in prompt
        assert "WEEKLY TRADE CAP REACHED" not in prompt


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
    test_jitter_configuration()
    test_jitter_applied_on_http_retry()
    test_jitter_applied_on_network_error()
    test_no_jitter_exact_backoff()
    test_timeout_configuration()
    test_call_llm_uses_configured_timeout()
    test_call_llm_timeout_override_argument()
    test_call_llm_env_timeout_used()
    test_call_llm_invalid_timeout_returns_none()
    test_safe_format_renders_non_finite_as_na()
    test_build_prompt_non_finite_asset_fields()
    test_build_prompt_non_finite_portfolio_and_positions()
    test_build_prompt_non_finite_cooldown_status()
    test_save_decision_fallback_branch_rejects_non_finite()
    
    print("=" * 60)
    print("All tests passed! ✓")
    print("=" * 60)


def test_save_decision_fallback_branch_rejects_non_finite():
    """When the safe serializer is unavailable, the raw-json fallback must
    still refuse to persist NaN/Infinity tokens (allow_nan=False)."""
    print("Test: Save Decision Fallback Branch Rejects Non-Finite")
    print("-" * 40)

    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        decision = {
            "timestamp": datetime.now().isoformat(),
            "actions": [{"ticker": "SPY", "action": "buy", "pct": 10.0}],
            "reasoning": "finite decision",
        }

        original = trading_agent_module.JSON_SAFE_AVAILABLE
        try:
            trading_agent_module.JSON_SAFE_AVAILABLE = False

            # Finite decision still saves through the fallback path.
            agent.save_decision(decision)
            loaded = json.loads(history_file.read_text())
            assert loaded[0]["actions"][0]["pct"] == 10.0

            # Non-finite decision must raise instead of writing NaN tokens.
            bad_decision = {
                "timestamp": datetime.now().isoformat(),
                "actions": [{"ticker": "SPY", "action": "buy", "pct": float("nan")}],
                "reasoning": "degenerate",
            }
            with pytest.raises(ValueError):
                agent.save_decision(bad_decision)
        finally:
            trading_agent_module.JSON_SAFE_AVAILABLE = original

        print("  Fallback path: finite saves, non-finite raises ValueError")
        print("✓ Fallback strict-JSON contract test passed\n")


def test_safe_pct_validates_before_scaling():
    """_safe_pct scales finite ratios and renders absence/non-finite as n/a."""
    from llm.trading_agent import _safe_pct

    assert _safe_pct(-0.025, ".2f") == "-2.50"
    assert _safe_pct(0.0, ".2f") == "0.00"
    # None must never reach the multiplication (None * 100 raises TypeError).
    assert _safe_pct(None, ".2f") == "n/a"
    assert _safe_pct(float("nan"), ".2f") == "n/a"
    assert _safe_pct(float("inf"), ".2f") == "n/a"
    assert _safe_pct(float("-inf"), ".2f") == "n/a"
    assert _safe_pct("abc", ".2f") == "n/a"
    # bool ⊂ int in Python but is not a ratio measurement.
    assert _safe_pct(True, ".2f") == "n/a"


def test_build_prompt_absent_risk_metric_keys_render_na():
    """Partial risk_metrics block: absent keys render n/a, never fictitious 0.00.

    0.0 collides with a real Sortino / symmetric-skew / mesokurtic
    measurement, so a missing key must not launder into "0.00" in the LLM
    prompt (PR #65 sentinel-collision doctrine, consumer side).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "positions": [],
            "risk_metrics": {
                "cvar_95": -0.025,
                "var_95": -0.02,
                "max_drawdown": -0.10,
                # sortino_ratio / skewness / kurtosis keys absent
            },
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "CVaR 95% (Expected Shortfall): -2.50%" in prompt
        assert "VaR 95%: -2.00%" in prompt
        assert "Max Drawdown: -10.00%" in prompt
        assert "Sortino Ratio: n/a" in prompt
        assert "Return Skewness: n/a" in prompt
        assert "Return Kurtosis: n/a" in prompt
        assert "Sortino Ratio: 0.00" not in prompt
        assert "Return Skewness: 0.00" not in prompt
        assert "Return Kurtosis: 0.00" not in prompt


def test_build_prompt_none_scaled_ratio_renders_na():
    """None at a scaled ratio renders n/a% instead of raising TypeError."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "positions": [],
            "risk_metrics": {
                "cvar_95": None,
                "var_95": -0.02,
                "max_drawdown": -0.10,
                "sortino_ratio": None,
                "skewness": 0.10,
                "kurtosis": 3.0,
            },
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "CVaR 95% (Expected Shortfall): n/a%" in prompt
        assert "Sortino Ratio: n/a" in prompt
        # Sibling finite stats still render normally (per-element drop: one
        # malformed element must not void the n−1 valid ones).
        assert "VaR 95%: -2.00%" in prompt
        assert "Max Drawdown: -10.00%" in prompt
        assert "Return Skewness: 0.10" in prompt
        assert "Return Kurtosis: 3.00" in prompt


def test_build_prompt_finite_risk_metrics_render_unchanged():
    """Healthy path pinned: finite risk stats format exactly as before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": 8000.0,
            "total_value": 10000.0,
            "positions": [],
            "risk_metrics": {
                "cvar_95": -0.025,
                "var_95": -0.02,
                "max_drawdown": -0.10,
                "sortino_ratio": 1.23,
                "skewness": -0.45,
                "kurtosis": 4.56,
            },
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "CVaR 95% (Expected Shortfall): -2.50%" in prompt
        assert "VaR 95%: -2.00%" in prompt
        assert "Max Drawdown: -10.00%" in prompt
        assert "Sortino Ratio: 1.23" in prompt
        assert "Return Skewness: -0.45" in prompt
        assert "Return Kurtosis: 4.56" in prompt


def test_build_prompt_absent_portfolio_totals_render_na():
    """Partial portfolio block: absent totals render n/a, never fictitious 0.00.

    0.0 collides with a real breakeven portfolio (zero P&L, zero return), so
    a missing key must not launder into "Cash: €0.00" in the LLM prompt
    (PR #65 sentinel-collision doctrine, consumer side).
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": 8000.0,
            "positions": [],
            # total_value / total_return_pct / total_pnl keys absent
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Cash: €8000.00" in prompt
        assert "Total Value: €n/a" in prompt
        assert "Total Return: n/a%" in prompt
        assert "Total P&L: €n/a" in prompt
        assert "Total Value: €0.00" not in prompt
        assert "Total Return: 0.00%" not in prompt
        assert "Total P&L: €+0.00" not in prompt


def test_build_prompt_none_portfolio_total_renders_na():
    """None portfolio totals render n/a while finite siblings stay intact."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": 8000.0,
            "total_value": None,
            "total_return_pct": None,
            "total_pnl": None,
            "positions": [],
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Cash: €8000.00" in prompt
        assert "Total Value: €n/a" in prompt
        assert "Total Return: n/a%" in prompt
        assert "Total P&L: €n/a" in prompt


def test_build_prompt_finite_portfolio_totals_render_unchanged():
    """Healthy path pinned: finite portfolio totals format exactly as before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {
            "cash": 2725.69,
            "total_value": 10123.45,
            "total_return_pct": 1.23,
            "total_pnl": 123.45,
            "positions": [],
        }

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Cash: €2725.69" in prompt
        assert "Total Value: €10123.45" in prompt
        assert "Total Return: 1.23%" in prompt
        assert "Total P&L: €+123.45" in prompt


def test_build_prompt_absent_asset_indicators_render_na():
    """Empty per-asset `latest` dict: every indicator renders n/a, never a
    fictitious reading.

    Reachable in production: get_latest_indicators returns {} for an empty
    indicator frame, so all seven keys are absent at once. The numeric
    .get defaults used to launder that into \"Price: €0.00\", \"RSI(14):
    50.0\" (a real neutral-momentum reading), \"Bollinger Position: 0.50\"
    (a real mid-band reading) and \"Volatility (ann): 0.0%\" — every one a
    collision with a plausible real measurement (PR #65 sentinel-collision
    doctrine, consumer side). The € prefix sits outside _safe_format, so
    absence renders \"€n/a\"; the % suffix sits outside _safe_pct, so
    absence renders \"n/a%\" — both asserted literally.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        market_data = {"assets": {"SPY": {"latest": {}}}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {"cash": 8000.0, "positions": []}

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Price: €n/a" in prompt
        assert "SMA20: €n/a | SMA50: €n/a" in prompt
        assert "RSI(14): n/a" in prompt
        assert "Bollinger Position: n/a" in prompt
        assert "Volatility (ann): n/a%" in prompt
        assert "Drawdown: n/a%" in prompt
        assert "Daily Return: n/a%" in prompt
        # No fictitious readings survive anywhere in the asset block.
        assert "Price: €0.00" not in prompt
        assert "RSI(14): 50.0" not in prompt
        assert "Bollinger Position: 0.50" not in prompt
        assert "Volatility (ann): 0.0%" not in prompt
        assert "Drawdown: 0.00%" not in prompt
        assert "Daily Return: 0.00%" not in prompt


def test_build_prompt_none_asset_indicator_renders_na():
    """None scaled-ratio renders n/a% and cannot crash the prompt build.

    _safe_pct validates BEFORE scaling: on old code
    `latest.get('volatility_annual', 0) * 100` raised TypeError for a
    stored None, killing the whole prompt build (a lost trading day).
    Per-element drop: one malformed element must not void the n−1 valid
    ones, so finite siblings keep rendering.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        latest = {
            "price": 100.0,
            "sma_20": 99.0,
            "sma_50": 98.0,
            "rsi_14": 45.0,
            "bb_position": 0.3,
            "volatility_annual": None,
            "drawdown": None,
            "daily_return": 0.01,
        }
        market_data = {"assets": {"SPY": {"latest": latest}}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {"cash": 8000.0, "positions": []}

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Volatility (ann): n/a%" in prompt
        assert "Drawdown: n/a%" in prompt
        # Finite siblings intact.
        assert "Price: €100.00" in prompt
        assert "SMA20: €99.00 | SMA50: €98.00" in prompt
        assert "RSI(14): 45.0" in prompt
        assert "Bollinger Position: 0.30" in prompt
        assert "Daily Return: 1.00%" in prompt


def test_build_prompt_finite_asset_indicators_render_unchanged():
    """Healthy path pinned: finite asset indicators format exactly as before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        history_file = Path(tmpdir) / "decisions.json"
        agent = TradingAgent(api_key="test", history_file=str(history_file))

        latest = {
            "price": 592.35,
            "sma_20": 588.10,
            "sma_50": 575.20,
            "rsi_14": 61.7,
            "bb_position": 0.82,
            "volatility_annual": 0.152,
            "drawdown": -0.034,
            "daily_return": 0.012,
        }
        market_data = {"assets": {"SPY": {"latest": latest}}, "correlations": pd.DataFrame(), "regime": None}
        portfolio = {"cash": 8000.0, "positions": []}

        prompt = agent.build_prompt(market_data, portfolio)

        assert "Price: €592.35" in prompt
        assert "SMA20: €588.10 | SMA50: €575.20" in prompt
        assert "RSI(14): 61.7" in prompt
        assert "Bollinger Position: 0.82" in prompt
        assert "Volatility (ann): 15.2%" in prompt
        assert "Drawdown: -3.40%" in prompt
        assert "Daily Return: 1.20%" in prompt


# --- PR #73: cooldown/decision display block — absence renders n/a ---------


def _make_agent(tmpdir):
    history_file = Path(tmpdir) / "decisions.json"
    return TradingAgent(api_key="test", history_file=str(history_file))


def _bare_market_portfolio():
    return {"assets": {}, "correlations": pd.DataFrame()}, {"cash": 0.0, "positions": []}


def test_build_prompt_truncated_cooldown_counters_render_na():
    """A cooldown dict missing trades_this_week/weekly_cap must render
    "n/a/n/a", not the old "0/2" full-budget fiction."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        market_data, portfolio = _bare_market_portfolio()

        prompt = agent.build_prompt(market_data, portfolio, [], {"active_entries": {}})

        assert "Weekly trades used: n/a/n/a" in prompt
        assert "Weekly trades used: 0/2" not in prompt


def test_build_prompt_missing_hold_days_and_min_hold_render_na():
    """Missing hold_days / min_hold_days must render n/a and block the sell
    display (fail-closed), not fabricate "held 0.0 days"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        market_data, portfolio = _bare_market_portfolio()
        cooldown_status = {
            "trades_this_week": 1,
            "weekly_cap": 3,
            "active_entries": {"MC.PA": {"entry_date": "2026-09-01T00:00:00"}},
            "config": {},
        }

        prompt = agent.build_prompt(market_data, portfolio, [], cooldown_status)

        assert "MC.PA: held n/a days — ✗ hold n/a more days" in prompt
        assert "held 0.0 days" not in prompt


def test_build_prompt_finite_hold_days_missing_min_hold_blocks_sell():
    """Finite hold_days with an absent threshold must still block the sell
    display — an unknown threshold must not read as "can sell"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        market_data, portfolio = _bare_market_portfolio()
        cooldown_status = {
            "trades_this_week": 1,
            "weekly_cap": 3,
            "active_entries": {
                "MC.PA": {"entry_date": "2026-09-01T00:00:00", "hold_days": 12.5}
            },
            "config": {},
        }

        prompt = agent.build_prompt(market_data, portfolio, [], cooldown_status)

        assert "MC.PA: held 12.5 days — ✗ hold n/a more days" in prompt
        assert "✓ can sell" not in prompt


def test_build_prompt_missing_days_since_exit_and_flip_days_render_na():
    """Missing days_since_exit / flip_cooldown_days must render n/a and block
    the re-buy display, not fabricate "exited 0.0 days ago"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        market_data, portfolio = _bare_market_portfolio()
        cooldown_status = {
            "trades_this_week": 1,
            "weekly_cap": 3,
            "recent_exits": {"TLT": {"exit_date": "2026-09-15T00:00:00"}},
            "config": {},
        }

        prompt = agent.build_prompt(market_data, portfolio, [], cooldown_status)

        assert "TLT: exited n/a days ago — ✗ wait n/a more days" in prompt
        assert "exited 0.0 days ago" not in prompt


def test_build_prompt_action_without_pct_renders_no_suffix():
    """pct is optional by action contract (hold carries none): the display
    must omit the % suffix instead of laundering absence into "0%"."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        market_data, portfolio = _bare_market_portfolio()
        recent = [{
            "timestamp": "2026-10-02T21:30:00",
            "reasoning": "x",
            "actions": [{"ticker": "SPY", "action": "hold"}],
        }]

        prompt = agent.build_prompt(market_data, portfolio, recent)

        assert "- SPY: hold" in prompt
        assert "- SPY: hold 0%" not in prompt


def test_build_prompt_cooldown_and_history_healthy_path_unchanged():
    """Healthy path pinned: populated cooldown counters, thresholds, and
    finite action pcts format exactly as before."""
    with tempfile.TemporaryDirectory() as tmpdir:
        agent = _make_agent(tmpdir)
        market_data, portfolio = _bare_market_portfolio()
        cooldown_status = {
            "trades_this_week": 1,
            "weekly_cap": 3,
            "active_entries": {
                "MC.PA": {"entry_date": "2026-09-01T00:00:00", "hold_days": 12.5}
            },
            "recent_exits": {
                "TLT": {"exit_date": "2026-09-15T00:00:00", "days_since_exit": 4.0}
            },
            "config": {"min_hold_days": 5, "flip_cooldown_days": 10},
        }
        recent = [{
            "timestamp": "2026-10-02T21:30:00",
            "reasoning": "x",
            "actions": [
                {"ticker": "AI.PA", "action": "buy", "pct": 17},
                {"ticker": "SPY", "action": "hold"},
            ],
        }]

        prompt = agent.build_prompt(market_data, portfolio, recent, cooldown_status)

        assert "Weekly trades used: 1/3" in prompt
        assert "MC.PA: held 12.5 days — ✓ can sell" in prompt
        assert "TLT: exited 4.0 days ago — ✗ wait 6.0 more days" in prompt
        assert "- AI.PA: buy 17%" in prompt
        assert "- SPY: hold" in prompt
