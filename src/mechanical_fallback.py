"""
Mechanical fallback rulebook for LLM outage days.

Context (2026-11-17 .. 2026-11-20 outage, declared policy 2026-11-19):
when the live-arm LLM call fails, ``daily_run.py`` fail-opens to a
hold-all decision (``decision.error == True``, zero actions). That
fallback is deliberately silent — but silence became dangerous on day 3
of the Venice 402 outage: SPY sat ~0.9 pp above its 5% adaptive stop
with *no* pipeline-side check at all, and only the intraday monitor's
mechanical alert stopped it out (2026-11-20 16:38 UTC). IJR closed the
day *past* its stated stop zone, unevaluated. The declared policy on a
4th consecutive error day was an explicit exit-rule review rather than
letting the exception become the rule.

This module is that review, codified. It is strictly NARROWER than the
LLM: it closes risk that has already breached a pre-registered,
price-only threshold, and it flags mandates it refuses to execute.
It never opens risk.

Rules (pre-registered, mechanical, price-only):

  MF-1  Hard stop exit. A held position whose fresh mark has drawdown
        <= -adaptive_stop (the same regime-adjusted threshold the
        cooldown manager uses) is sold at 100%. Exit is risk-closing
        and price-verifiable; it does not require judgement. Stale-feed
        tickers are excluded (a frozen close must never trigger an
        exit — incident doctrine 2026-11-05). The Step-5 execution path
        re-checks ``can_sell``, whose stop-loss override already
        bypasses the weekly cap and min-hold, so a rulebook exit can
        never be blocked by bookkeeping.

  MF-2  Watch band. Positions within ``warning_band_pct`` of their stop
        (above it) are reported as watch items — no action. Pure
        visibility for the daily JSON and the session notes.

  MF-3  Cash-band mandate flag. If cash is outside the configured band
        (Rule-5 territory), the mandate is surfaced loudly — but NEVER
        executed. Deploying cash is an opening decision; a mechanical
        fallback that buys would be a new-position policy invented
        mid-outage. The mandate rolls to the next working LLM session.

The rulebook is pure: it takes plain data and returns a dict of actions
plus an informational block. It performs no I/O, no portfolio mutation,
and no LLM call. Non-finite or non-positive inputs are skipped loudly,
per the project's ingress-guard conventions.

Failure direction: when in doubt, do nothing. A false exit costs fees
and a realized loss; a missed exit while the monitor still runs
intraday is a bounded, monitored risk. The asymmetry favors silence.
"""

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

#: Default warning band (in percentage points of drawdown) above the hard
#: stop inside which a position is reported as a watch item.
DEFAULT_WARNING_BAND_PCT = 1.0

#: Rule-5 cash band (fraction of portfolio value), matching the prompt's
#: standing cash-band rule. A breach is surfaced, never executed.
CASH_BAND_FRACTION: Tuple[float, float] = (0.15, 0.30)


def _is_finite(value) -> bool:
    """Ingress guard: accept only finite, real numbers."""
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def run_mechanical_rulebook(
    positions: Sequence[Dict],
    current_prices: Dict[str, Optional[float]],
    stale_tickers: Sequence[str],
    adaptive_stop_pct: float,
    cash_pct: Optional[float] = None,
    warning_band_pct: float = DEFAULT_WARNING_BAND_PCT,
) -> Dict:
    """Evaluate the mechanical fallback rules over the held positions.

    Args:
        positions: Held positions, each a dict with at least ``ticker``
            and ``avg_price`` (and optionally ``market_value``). This is
            the shape produced by ``Portfolio.get_summary()['positions']``.
        current_prices: Fresh mark per ticker (None or missing = no
            trustworthy price; the position is skipped loudly).
        stale_tickers: Tickers whose feed is stale or missing. Excluded
            from MF-1/MF-2 — a frozen close must never trigger or
            calibrate an exit (2026-11-05 doctrine).
        adaptive_stop_pct: Regime-adjusted stop threshold in percent
            (positive number, e.g. 5.0 for the normal-vol 5% stop).
        cash_pct: Cash as a fraction of total portfolio value (0-1).
            None disables the MF-3 flag.
        warning_band_pct: Drawdown distance (in pp) below which a
            non-breaching position is reported as a watch item.

    Returns a dict::

        {
          "active": bool,              # True when the rulebook ran
          "stop_exits": [{ticker, drawdown_pct, price, avg_price}],
          "watch": [{ticker, drawdown_pct, distance_to_stop_pp}],
          "skipped": {ticker: reason}, # loud skips (stale/no price/guard)
          "cash_mandate": {
              "active": bool,
              "cash_pct": float|None,
              "band": [low, high],
              "side": "above"|"below"|None,
          },
          "actions": [{ticker, action, pct}],  # MF-1 sells only
          "rules": {"mf1_hard_stop": {...}, "mf2_watch": {...},
                    "mf3_cash_band": {...}},
        }

    Invariants: ``actions`` contains ONLY ``sell`` actions at pct=100;
    the rulebook never emits a buy; every emitted ticker passed the
    freshness, finiteness and positivity guards.
    """
    stale_set = set(stale_tickers or [])
    stop_exits: List[Dict] = []
    watch: List[Dict] = []
    skipped: Dict[str, str] = {}
    actions: List[Dict] = []

    if not _is_finite(adaptive_stop_pct) or adaptive_stop_pct <= 0:
        logger.warning(
            "Mechanical fallback: non-finite or non-positive adaptive_stop_pct %r; "
            "rulebook runs in flag-only mode (no MF-1 exits).",
            adaptive_stop_pct,
        )
        adaptive_stop_pct = math.nan

    for position in positions or []:
        ticker = position.get("ticker")
        if not ticker or not isinstance(ticker, str):
            skipped[str(ticker)] = "missing or invalid ticker"
            continue
        if ticker in stale_set:
            skipped[ticker] = "stale feed — excluded from mechanical rules (2026-11-05 doctrine)"
            continue

        avg_price = position.get("avg_price")
        price = current_prices.get(ticker)
        if not _is_finite(price) or price <= 0:
            skipped[ticker] = f"no fresh price (got {price!r})"
            continue
        if not _is_finite(avg_price) or avg_price <= 0:
            skipped[ticker] = f"invalid avg_price {avg_price!r}"
            continue

        drawdown_pct = ((price - avg_price) / avg_price) * 100.0
        if not _is_finite(drawdown_pct):
            skipped[ticker] = f"non-finite drawdown from price={price!r} avg={avg_price!r}"
            continue

        if math.isnan(adaptive_stop_pct):
            # Flag-only mode: threshold unverifiable, no exits, no watch.
            continue

        if drawdown_pct <= -adaptive_stop_pct:
            stop_exits.append({
                "ticker": ticker,
                "drawdown_pct": drawdown_pct,
                "price": price,
                "avg_price": avg_price,
            })
            actions.append({"ticker": ticker, "action": "sell", "pct": 100})
        elif drawdown_pct <= -(adaptive_stop_pct - warning_band_pct):
            watch.append({
                "ticker": ticker,
                "drawdown_pct": drawdown_pct,
                "distance_to_stop_pp": drawdown_pct + adaptive_stop_pct,
            })

    cash_mandate = {
        "active": False,
        "cash_pct": cash_pct if _is_finite(cash_pct) else None,
        "band": [CASH_BAND_FRACTION[0], CASH_BAND_FRACTION[1]],
        "side": None,
        "executed": False,
        "note": "Mandate surfaced only — mechanical fallback never buys.",
    }
    if _is_finite(cash_pct):
        if cash_pct > CASH_BAND_FRACTION[1]:
            cash_mandate.update(active=True, side="above")
        elif cash_pct < CASH_BAND_FRACTION[0]:
            cash_mandate.update(active=True, side="below")

    return {
        "active": True,
        "stop_exits": stop_exits,
        "watch": watch,
        "skipped": skipped,
        "cash_mandate": cash_mandate,
        "actions": actions,
        "rules": {
            "mf1_hard_stop": {
                "threshold_pct": adaptive_stop_pct if not math.isnan(adaptive_stop_pct) else None,
                "exits": [e["ticker"] for e in stop_exits],
            },
            "mf2_watch": {
                "warning_band_pct": warning_band_pct,
                "watch": [w["ticker"] for w in watch],
            },
            "mf3_cash_band": cash_mandate,
        },
    }
