#!/usr/bin/env python3
"""
Benchmark: LLM retry jitter distribution.

Exponential backoff without jitter causes retry attempts from different callers
to cluster at the same instants, creating a thundering-herd effect when a shared
API is temporarily overloaded. Jitter spreads those retries over time.

This benchmark simulates 1000 independent retry schedules and reports the
spread of retry times for the first retry. With jitter, the retry times should
be distributed over [base, base * (1 + jitter)] instead of being a single
point mass.
"""

import random
import statistics
import logging
from unittest.mock import Mock, patch

import requests
import sys
from pathlib import Path

logging.disable(logging.CRITICAL)

sys.path.insert(0, str(Path(__file__).parent / "src"))
from llm.trading_agent import TradingAgent


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


def collect_retry_wait_times(jitter, n_runs=1000, base_factor=1.0, max_retries=1):
    """
    Simulate a 429 -> success sequence and record the first retry wait time
    for each run. Returns a list of wait times in seconds.
    """
    agent = TradingAgent(
        api_key="test",
        retry_backoff_factor=base_factor,
        retry_jitter=jitter,
        max_retries=max_retries
    )
    waits = []
    error_response = _make_error_response(429)
    success_response = _make_success_response()

    for _ in range(n_runs):
        with patch('requests.post', side_effect=[error_response, success_response]):
            with patch('time.sleep', return_value=None) as mock_sleep:
                agent.call_llm("test prompt")
                waits.append(mock_sleep.call_args[0][0])

    return waits


def main():
    random.seed(42)
    print("=" * 70)
    print("BENCHMARK: LLM API Retry Jitter Distribution")
    print("=" * 70)
    print()
    print("Base backoff for first retry: 1.0s")
    print()

    for jitter in (0.0, 0.125, 0.25, 0.5, 1.0):
        waits = collect_retry_wait_times(jitter=jitter)
        print(f"Jitter factor = {jitter:.3f}")
        print(f"  Min wait: {min(waits):.3f}s")
        print(f"  Max wait: {max(waits):.3f}s")
        print(f"  Mean wait: {statistics.mean(waits):.3f}s")
        print(f"  Std dev: {statistics.stdev(waits):.3f}s")

        # Bin wait times into 10 buckets to show spread
        max_wait = max(waits) if max(waits) > min(waits) else min(waits) + 1e-9
        n_buckets = 8
        bucket_width = (max_wait - min(waits)) / n_buckets if max_wait > min(waits) else 1.0
        buckets = [0] * n_buckets
        for w in waits:
            idx = min(int((w - min(waits)) / bucket_width), n_buckets - 1)
            buckets[idx] += 1

        print(f"  Histogram (buckets of ~{bucket_width:.3f}s):")
        for i, count in enumerate(buckets):
            lo = min(waits) + i * bucket_width
            bar = "#" * (count // (len(waits) // 40 + 1))
            print(f"    [{lo:.3f}s, {lo + bucket_width:.3f}s): {count:4d} {bar}")
        print()

    print("=" * 70)
    print("Interpretation")
    print("=" * 70)
    print(
        "Without jitter, every first retry happens at exactly 1.0s. With a "
        "jitter factor of 0.25, retries are spread uniformly over [1.0s, 1.25s], "
        "reducing the probability of synchronized retries hammering the API "
        "during a transient outage."
    )
    print()


if __name__ == "__main__":
    main()
