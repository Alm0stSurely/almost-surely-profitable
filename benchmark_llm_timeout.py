"""
Benchmark: LLM request timeout and retry budget.

Computes the worst-case wall-clock budget for a TradingAgent call sequence
(initial request + retries) under different timeout and backoff settings.
The goal is to give operators a deterministic way to pick a timeout that
bounds the total time the daily pipeline can spend blocked on the LLM API.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from unittest.mock import Mock, patch
from llm.trading_agent import TradingAgent


def make_success_response():
    """Create a mocked successful JSON response."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "choices": [{"message": {"content": '{"actions": [], "reasoning": "OK"}'}}]
    }
    mock_response.raise_for_status.return_value = None
    return mock_response


def make_server_error_response():
    """Create a mocked 503 response."""
    import requests
    mock_response = Mock()
    mock_response.status_code = 503
    mock_response.text = "Service Unavailable"
    mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError(
        response=mock_response
    )
    return mock_response


def benchmark_worst_case_budget(timeout: float, max_retries: int, backoff_factor: float) -> float:
    """
    Estimate the worst-case wall-clock budget for one LLM call sequence.

    Assumes every request times out or returns a retryable error, then retries
    until exhausted. Total budget = timeout per attempt * (max_retries + 1)
                            + sum of backoff waits between attempts.
    """
    agent = TradingAgent(
        api_key="test_key",
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_factor=backoff_factor
    )

    # Sum of retry sleeps: backoff * (2^0 + 2^1 + ... + 2^(max_retries-1))
    # = backoff * (2^max_retries - 1)
    retry_wait_sum = backoff_factor * ((2 ** max_retries) - 1)
    request_time_sum = timeout * (max_retries + 1)
    return request_time_sum + retry_wait_sum


def benchmark_actual_call(timeout: float, max_retries: int, backoff_factor: float) -> float:
    """
    Measure actual time for a mocked sequence where every request times out.
    Uses mocked time.sleep to avoid wall-clock delays.
    """
    import requests
    agent = TradingAgent(
        api_key="test_key",
        timeout=timeout,
        max_retries=max_retries,
        retry_backoff_factor=backoff_factor
    )

    total_sleep = [0.0]

    def fake_sleep(seconds):
        total_sleep[0] += seconds

    with patch('requests.post', side_effect=requests.exceptions.Timeout("Request timed out")):
        with patch('time.sleep', side_effect=fake_sleep):
            start = time.perf_counter()
            agent.call_llm("test prompt")
            elapsed = time.perf_counter() - start

    return elapsed, total_sleep[0]


def main():
    print("=" * 70)
    print("LLM Timeout & Retry Budget Benchmark")
    print("=" * 70)
    print()

    configs = [
        {"timeout": 30, "max_retries": 3, "backoff_factor": 1.0},
        {"timeout": 60, "max_retries": 3, "backoff_factor": 1.0},
        {"timeout": 90, "max_retries": 3, "backoff_factor": 1.0},
        {"timeout": 180, "max_retries": 3, "backoff_factor": 1.0},
        {"timeout": 180, "max_retries": 5, "backoff_factor": 1.0},
        {"timeout": 180, "max_retries": 3, "backoff_factor": 2.0},
    ]

    print("Worst-case time budget (all attempts fail or time out):")
    print("-" * 70)
    print(f"{'Timeout (s)':<12} {'Retries':<8} {'Backoff':<9} {'Request Time':<14} {'Retry Sleep':<13} {'Total Budget':<13}")
    print("-" * 70)

    for cfg in configs:
        timeout = cfg["timeout"]
        max_retries = cfg["max_retries"]
        backoff = cfg["backoff_factor"]
        budget = benchmark_worst_case_budget(timeout, max_retries, backoff)
        request_time = timeout * (max_retries + 1)
        retry_sleep = budget - request_time
        print(
            f"{timeout:<12} {max_retries:<8} {backoff:<9.1f} "
            f"{request_time:<14.1f} {retry_sleep:<13.1f} {budget:<13.1f}"
        )

    print()
    print("Simulated call with mocked timeouts (no actual sleep):")
    print("-" * 70)
    print(f"{'Timeout (s)':<12} {'Retries':<8} {'Backoff':<9} {'Mocked Sleep':<14} {'Wall Time (μs)':<16}")
    print("-" * 70)

    for cfg in configs:
        elapsed, slept = benchmark_actual_call(
            cfg["timeout"], cfg["max_retries"], cfg["backoff_factor"]
        )
        print(
            f"{cfg['timeout']:<12} {cfg['max_retries']:<8} {cfg['backoff_factor']:<9.1f} "
            f"{slept:<14.1f} {elapsed * 1e6:<16.1f}"
        )

    print()
    print("Observation:")
    print("  The default config (180s timeout, 3 retries, 1.0s backoff) has a")
    print("  worst-case budget of ~727s (≈12 minutes). Operators running the")
    print("  daily pipeline under a scheduler should set LLM_TIMEOUT and")
    print("  LLM_MAX_RETRIES so that Total Budget < available scheduling window.")
    print("=" * 70)


if __name__ == "__main__":
    main()
