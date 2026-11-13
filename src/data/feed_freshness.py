"""Shared feed-freshness classification for monitor and nightly pipeline.

Production incident 2026-11-05 (monitor run 08:06 UTC): the yfinance
Euronext feed went stale overnight (Paris series frozen at the 2026-11-03
close) while the US feed stayed healthy. The monitor treated stale closes
as live prices, fired three spurious POSITION_MOVEMENT alerts (+3.1 % /
+3.3 % / -2.3 %), and persisted the stale closes as ``previous_close``
references — poisoning the 12:15 run with two-day-old deltas even after
the feed recovered. A price is only as trustworthy as the bar behind it.

The intraday monitor has guarded itself since that incident
(``split_fresh_and_stale`` on every quote). The nightly pipeline
(``daily_run.run_daily_pipeline``, 21:00 UTC) had no equivalent check
until 2026-11-13: a fetch failure silently shrinks the analyzed universe,
and a frozen-but-served daily frame is indistinguishable from a normal
close — so the LLM could decide, and the executor trade, on prices that
miss required trading sessions. The 2026-11-13 morning session note put
it plainly: *"le guard protège le monitor, pas le pipeline."* This module
is the shared classifier both consumers use.

Session windows below are generous (DST-agnostic) bounds in UTC; a window
that is too wide only risks classifying a stale feed as fresh, never the
reverse. Holidays are NOT modelled: on a US holiday the US tickers are
flagged stale while Euronext is open — suppressing US alerts for one day
is acceptable and documented here (same tradeoff as the monitor).
"""

from datetime import date, datetime, time as dtime, timedelta
from typing import Dict, Iterable, Optional, Set, Tuple

import pandas as pd

from utils import _is_finite_number

# (open_utc, close_utc) — generous bounds, winter and summer covered.
_EURONEXT_SESSION = (dtime(7, 0), dtime(16, 30))
_US_SESSION = (dtime(13, 0), dtime(21, 30))


def _market_of(ticker: str) -> str:
    """Classify a ticker into 'euronext' or 'us' for session arithmetic."""
    if ticker.endswith('.PA') or ticker == '^FCHI':
        return 'euronext'
    return 'us'


def _session_date_from_bar(last_bar_ts: Optional[str], market: str) -> Optional[date]:
    """Map a bar's UTC timestamp to the trading session it belongs to.

    yfinance stamps Euronext daily bars at 23:00 UTC of the *previous* day
    (midnight CET of the session date), while intraday 1m bars are stamped
    within the session itself. A bar at/after 22:30 UTC therefore belongs to
    the next calendar day's session; earlier bars belong to their own date.
    US bars (00:00 daily stamps or intraday) always map to their own date.
    """
    if not last_bar_ts:
        return None
    try:
        ts = datetime.fromisoformat(last_bar_ts)
    except (ValueError, TypeError):
        return None
    bar_date = ts.date()
    if market == 'euronext' and ts.time() >= dtime(22, 30):
        return bar_date + timedelta(days=1)
    return bar_date


def session_date_of_daily_bar(ts: pd.Timestamp, market: str) -> date:
    """Map a DAILY-interval bar's UTC timestamp to its trading session.

    Daily-interval anchors differ from intraday stamps: Euronext daily bars
    are anchored at local midnight — 23:00 UTC the previous day in winter
    (CET) and 22:00 UTC in summer (CEST). Both are evening stamps, so the
    intraday 22:30 boundary used by ``_session_date_from_bar`` would
    misclassify every summer daily bar as one session old. The boundary
    here is 20:00 UTC, safely below the 22:00 summer anchor and still far
    from any plausible intraday stamp this helper might see. US daily bars
    are anchored at midnight ET (04:00/05:00 UTC of the session date
    itself) — always their own calendar date.
    """
    bar_date = ts.date()
    if market == 'euronext' and ts.time() >= dtime(20, 0):
        return bar_date + timedelta(days=1)
    return bar_date


def _expected_session_date(now: datetime, market: str) -> date:
    """The session whose bar the feed must serve at time ``now``.

    During a market's session window the forming bar of today counts as
    fresh; outside it, the most recently completed session is required.
    Weekends roll back to Friday.
    """
    open_utc, _ = (
        _EURONEXT_SESSION if market == 'euronext' else _US_SESSION
    )
    d = now.date()
    if d.weekday() < 5 and now.time() >= open_utc:
        return d
    while True:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            return d


def is_feed_stale(last_bar_ts: Optional[str], ticker: str, now: datetime) -> bool:
    """Return True when the ticker's last bar misses a required session.

    A missing timestamp (fetch failure, non-finite close) is stale by
    definition: the caller must not alert on it nor overwrite the saved
    reference with it.
    """
    market = _market_of(ticker)
    bar_session = _session_date_from_bar(last_bar_ts, market)
    if bar_session is None:
        return True
    return bar_session < _expected_session_date(now, market)


def split_fresh_and_stale(
    quotes: Dict[str, Dict[str, Optional[str]]],
    now: datetime,
) -> Tuple[Dict[str, float], Set[str]]:
    """Partition quotes into fresh prices and stale tickers.

    Returns (fresh_prices, stale_tickers): fresh_prices maps ticker to a
    finite price only; stale_tickers collects every ticker whose quote is
    stale (old bar) or unusable (None price/timestamp, non-finite price).
    """
    fresh: Dict[str, float] = {}
    stale: Set[str] = set()
    for ticker, quote in quotes.items():
        price = quote.get('price')
        last_bar_ts = quote.get('last_bar_ts')
        if not _is_finite_number(price) or is_feed_stale(last_bar_ts, ticker, now):
            stale.add(ticker)
            continue
        fresh[ticker] = float(price)
    return fresh, stale


def classify_daily_frames(
    expected_tickers: Iterable[str],
    frames: Dict[str, pd.DataFrame],
    now: datetime,
) -> Dict[str, str]:
    """Classify bulk daily OHLCV frames as fresh or stale.

    ``frames`` is the ``fetch_historical_data`` output: ticker -> OHLCV
    DataFrame with a tz-naive UTC DatetimeIndex (see
    ``data.fetch_market_data._fetch_single_ticker``, which normalizes the
    index and drops non-finite Close rows at ingress).

    Returns a dict mapping each STALE ticker to a human-readable reason:

    - ``missing_frame``  — the fetcher returned nothing for the ticker
      (the live incident signature: yfinance "possibly delisted" errors
      leave the ticker absent from the result dict entirely);
    - ``empty_frame``    — a frame without usable rows;
    - ``stale_bar:<ts>`` — the last bar's trading session is older than
      the session required at ``now``.

    Fresh tickers are absent from the result. Callers decide what the
    classification means for them: the monitor excludes stale tickers
    from alerting and preserves saved references; the nightly pipeline
    banners everything stale and forces a HOLD when a held position's
    feed is untrustworthy.
    """
    stale: Dict[str, str] = {}
    for ticker in expected_tickers:
        frame = frames.get(ticker)
        if frame is None:
            stale[ticker] = 'missing_frame'
            continue
        if not isinstance(frame, pd.DataFrame) or frame.empty or 'Close' not in frame.columns:
            stale[ticker] = 'empty_frame'
            continue
        if not isinstance(frame.index, pd.DatetimeIndex):
            # Producer-contract violation (fetch_historical_data guarantees a
            # tz-naive UTC DatetimeIndex). A frame we cannot date cannot be
            # freshness-checked — fail closed rather than guess or crash.
            stale[ticker] = 'no_datetime_index'
            continue
        last_bar_ts = frame.index.max()
        market = _market_of(ticker)
        bar_session = session_date_of_daily_bar(last_bar_ts, market)
        if bar_session < _expected_session_date(now, market):
            stale[ticker] = f'stale_bar:{last_bar_ts.isoformat()}'
    return stale
