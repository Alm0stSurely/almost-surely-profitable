"""Shared finite-safe formatting helpers for console and LLM-prompt output.

Extracted from ``backtest.formatting`` (itself extracted from
``backtest.backtest``) so that every formatter in the codebase validates
values before scaling/formatting instead of growing per-module copies of
the same convention. Lightweight by design: stdlib + numpy only, no heavy
imports, so optional tooling (e.g. ``backtest.visualize``) can rely on it
without pulling in the LLM/data stack.
"""

import math

import numpy as np


def _fmt_finite(value: float, spec: str) -> str:
    """Format a numeric value if finite, else return 'n/a'.

    Public formatters may receive aggregates built from market data
    containing NaN ticks or degenerate price series, and public dataclasses
    (e.g. ``RegimeState``) may be constructed by callers with non-finite
    fields; the formatter is the last guardrail before nan/inf tokens reach
    the console, captured CI logs, or the LLM prompt.

    ``bool`` is rejected explicitly even though it subclasses ``int`` —
    truthy regime flags must not render as ``1``/``0``. ``np.integer`` is
    accepted directly: integers are finite by construction, and numpy
    scalars support ``__format__``.
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
