"""Utilities for loading and validating daily trading results."""
import json
from pathlib import Path
from typing import Dict, List


MIN_ASSETS_FOR_VALID_RUN = 5


def is_valid_daily_result(data: Dict) -> bool:
    """Return True if a daily result dict should be used for analysis.

    Filters out:
    - Dry-run results (explicit ``dry_run: true`` or ``_dry_run`` filename).
    - Test/placeholder artifacts (e.g. portfolio reset to 10 000 EUR with no
      positions and reasoning containing "HOLD for test").
    - Runs with too few assets analyzed (likely partial tests).
    """
    if data.get("dry_run", False):
        return False

    reasoning = (data.get("decision", {}).get("reasoning", "") or "").lower()
    if "test" in reasoning or "placeholder" in reasoning:
        return False

    market_summary = data.get("market_summary")
    if market_summary is not None:
        assets = market_summary.get("assets_analyzed", 0)
        if assets < MIN_ASSETS_FOR_VALID_RUN:
            return False

    return True


def load_valid_daily_results(
    results_dir: str = "results/daily",
    pattern: str = "*.json",
    skip_dry_run_files: bool = True,
) -> List[Dict]:
    """Load all valid daily results from ``results_dir``.

    Returns results sorted by filename (which is date-ordered for ISO dates).
    """
    path = Path(results_dir)
    if not path.exists():
        return []

    results: List[Dict] = []
    for file in sorted(path.glob(pattern)):
        if skip_dry_run_files and "_dry_run" in file.name:
            continue
        try:
            with open(file) as f:
                data = json.load(f)
            if is_valid_daily_result(data):
                results.append(data)
        except Exception:
            continue
    return results


def load_valid_daily_results_limited(
    results_dir: str = "results/daily",
    days: int = 30,
) -> List[Dict]:
    """Load the most recent ``days`` valid daily results."""
    all_results = load_valid_daily_results(results_dir)
    return all_results[-days:]
