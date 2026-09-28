#!/usr/bin/env python3
"""
Benchmark: LLM retry resilience.

Simulates transient API failures and measures how the exponential-backoff
retry policy in src/llm/trading_agent.py improves the chance that a single
call_llm() invocation returns a usable response.

A retryable failure is defined as HTTP 429, 500, 502, 503, 504 or a
network-level requests exception. A non-retryable 4xx error is not retried.
(500 was added to the retry set on 2026-09-28: it is a transient
server-side fault in the same trajectory class as 502/503/504.)
"""

import random
import logging
from unittest.mock import Mock, patch

import requests

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
from llm.trading_agent import TradingAgent

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("llm.trading_agent")
logger.setLevel(logging.CRITICAL)


def _make_success_response():
    r = Mock()
    r.status_code = 200
    r.json.return_value = {"choices": [{"message": {"content": '{"actions": []}'}}]}
    r.raise_for_status.return_value = None
    return r


def _make_error_response(status_code):
    r = Mock()
    r.status_code = status_code
    r.text = f"HTTP {status_code}"
    r.raise_for_status.side_effect = requests.exceptions.HTTPError(response=r)
    return r


def simulate_call(transient_failure_rate, max_retries, n_runs=1000, status_code=503):
    """
    Simulate call_llm() with a fixed probability of transient failure on each
    HTTP request. Return the fraction of invocations that ultimately succeed.
    """
    agent = TradingAgent(api_key="test", max_retries=max_retries, retry_backoff_factor=0.0)
    successes = 0

    for _ in range(n_runs):
        responses = []
        for _ in range(max_retries + 1):
            if random.random() < transient_failure_rate:
                responses.append(_make_error_response(status_code))
            else:
                responses.append(_make_success_response())
                break

        with patch('requests.post', side_effect=responses), patch('time.sleep', return_value=None):
            result = agent.call_llm("test prompt")
            if result is not None:
                successes += 1

    return successes / n_runs


def main():
    random.seed(42)
    print("=" * 70)
    print("BENCHMARK: LLM API Retry Resilience")
    print("=" * 70)
    print()
    print("Transient failure rate per request -> success rate")
    print()

    for failure_rate in (0.0, 0.25, 0.50, 0.75, 0.90):
        no_retry = simulate_call(failure_rate, max_retries=0)
        with_retry = simulate_call(failure_rate, max_retries=3)
        theoretical = 1 - (failure_rate ** 4)  # 1 initial + 3 retries

        print(f"  Failure rate {failure_rate:>4.0%}:")
        print(f"    No retry   : {no_retry:.3f}  (expected: {1 - failure_rate:.3f})")
        print(f"    3 retries  : {with_retry:.3f}  (expected: {theoretical:.3f})")
        print()

    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print(
        "With 3 retries, a 50% transient failure rate is reduced to a ~7% "
        "chance of a failed daily trading session. A 75% transient failure "
        "rate still leaves ~68% of sessions successful."
    )
    print()

    # 500 equivalence: the status added to the retry set on 2026-09-28 must
    # behave identically to 503 under the same transient-failure model.
    for failure_rate in (0.25, 0.50, 0.75):
        s500 = simulate_call(failure_rate, max_retries=3, status_code=500)
        s503 = simulate_call(failure_rate, max_retries=3, status_code=503)
        match = "OK" if abs(s500 - s503) < 0.05 else "MISMATCH"
        print(f"  500 vs 503 @ {failure_rate:>4.0%} failure: {s500:.3f} vs {s503:.3f}  [{match}]")
    print()


if __name__ == "__main__":
    main()
