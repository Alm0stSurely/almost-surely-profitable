"""Functional gate for the Step-3 stale-quote mark-persistence guard.

A stale-but-served close must never persist as a position's mark. Incident
shape (2026-11-05): the yfinance Euronext feed freezes; yfinance still
SERVES the last close (finite, positive — passes every guard in
``Portfolio.update_prices``). The nightly pipeline's Step 3 fetched current
prices via ``fetch_current_prices`` — which DISCARDS the ``last_bar_ts``
freshness metadata — and passed bare prices to ``update_prices``; the
caller's ``save_state()`` then persisted the stale close as the position's
``current_price``. Tomorrow's deltas, weights, stop checks, and
weekly-report marks would be computed against a frozen reference — the
exact poisoning the monitor incident documented ("persisted the stale
closes as previous_close references — poisoning the 12:15 run with
two-day-old deltas even after the feed recovered").

``data/feed_freshness.is_feed_stale``'s contract is explicit: "the caller
must not alert on it nor overwrite the saved reference with it." The
monitor honours this (``split_fresh_and_stale`` before every reference
update, monitor.py:797-810); the pipeline's Step 3 did not.

This gate drives the REAL code path: post-fix it calls
``daily_run._refresh_position_marks`` with the quote fetch monkeypatched;
pre-fix (helper absent) it replays the actual old Step-3 sequence. Exit 1
= stale close persisted (defect live); exit 0 = last fresh mark kept.

Run: python benchmarks/repro_benchmark_stale_quote_mark_persistence.py
"""

import sys
import tempfile
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from portfolio.portfolio import Portfolio  # noqa: E402

# Friday 2026-11-13 21:00 UTC — canonical nightly run time. The feed for
# AI.PA froze at Wednesday's close (incident replay); SPY is healthy.
NOW = datetime(2026, 11, 13, 21, 0, 0)
STALE_CLOSE = 356.93   # frozen Paris close, still finite/positive/served
ENTRY_PRICE = 379.56   # last fresh mark for AI.PA
FRESH_CLOSE = 100.02   # healthy US close

QUOTES = {
    "AI.PA": {"price": STALE_CLOSE, "last_bar_ts": "2026-11-11 14:30:00"},
    "SPY": {"price": FRESH_CLOSE, "last_bar_ts": "2026-11-13 20:59:00"},
}

with tempfile.TemporaryDirectory() as tmp:
    data_dir = Path(tmp)
    pf = Portfolio(data_dir=str(data_dir))
    pf.cash = 5000.0
    pf.buy("AI.PA", pct_of_cash=20.0, current_price=ENTRY_PRICE)
    pf.buy("SPY", pct_of_cash=20.0, current_price=99.50)

    try:
        from daily_run import _refresh_position_marks
    except ImportError:
        _refresh_position_marks = None

    if _refresh_position_marks is not None:
        # Post-fix: drive the real Step-3 helper with the fetch seam patched.
        import daily_run
        daily_run.fetch_current_quotes = lambda tickers, max_workers=1: QUOTES
        _refresh_position_marks(pf, NOW)
    else:
        # Pre-fix: the pipeline's actual Step-3 sequence (fetch_current_prices
        # drops freshness metadata; bare prices reach update_prices).
        bare_prices = {t: q["price"] for t, q in QUOTES.items()}
        pf.update_prices(bare_prices)

    pf.save_state()

    # Reload from disk: what mark did the pipeline persist for AI.PA?
    pf2 = Portfolio(data_dir=str(data_dir))
    persisted = pf2.positions["AI.PA"].current_price
    fresh_mark = pf2.positions["SPY"].current_price

    print(f"AI.PA persisted mark: {persisted} (last fresh was {ENTRY_PRICE})")
    print(f"SPY  persisted mark: {fresh_mark} (fresh close {FRESH_CLOSE})")

    failures = []
    if abs(persisted - STALE_CLOSE) < 1e-9:
        failures.append("STALE CLOSE PERSISTED as the AI.PA mark")
    if abs(fresh_mark - FRESH_CLOSE) > 1e-9:
        failures.append("fresh SPY close was NOT persisted (blast radius)")

    if failures:
        for f in failures:
            print(f"XX {f}")
        sys.exit(1)
    print("OK stale close rejected, last fresh mark kept, fresh sibling persisted")
