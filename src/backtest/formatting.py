"""Shared finite-safe formatting helpers for backtest console output.

Extracted from ``backtest.backtest`` so that every console formatter in the
backtest package (per-strategy report, comparison tables) validates values
before scaling/formatting instead of growing per-module copies of the same
convention. Lightweight by design: stdlib + numpy only, no heavy imports, so
optional tooling (e.g. ``visualize.py``) can rely on it without pulling in
the LLM/data stack.
"""

import math

import numpy as np


def _fmt_finite(value: float, spec: str) -> str:
    """Format a numeric value if finite, else return 'n/a'.

    Public formatters may receive result dicts built from market data
    containing NaN ticks or degenerate price series; the formatter is the
    last guardrail before nan/inf tokens reach the console (and any
    captured CI logs).
    """
    if isinstance(value, bool):
        return "n/a"
    if isinstance(value, (int, np.integer)):
        return format(value, spec)
    if isinstance(value, (float, np.floating)):
        v = float(value)
        if math.isfinite(v):
            return format(v, spec)
    return "n/a"


def _fmt_pct(value: float, spec: str) -> str:
    """Format a fractional value as a percentage if finite, else return 'n/a'.

    Validates the raw value before scaling: ``None`` or a non-numeric input
    must not crash the report, and a non-finite fraction must not be
    multiplied into a non-finite percentage first.
    """
    if isinstance(value, (int, float, np.floating)) and not isinstance(value, bool):
        v = float(value)
        if math.isfinite(v):
            return format(v * 100, spec)
    return "n/a"
