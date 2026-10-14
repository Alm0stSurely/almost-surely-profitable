"""
Live Equal-Weight Benchmark Tracker

Tracks an equal-weight portfolio in real-time alongside the LLM-driven strategy.
Provides a fair baseline for comparison.
"""

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from utils import _is_finite_number, dump_json_safe, load_json_dict_or_quarantine

logger = logging.getLogger(__name__)


class LiveEqualWeightBenchmark:
    """
    Live equal-weight benchmark for real-time strategy comparison.

    Holds equal weight in all available universe tickers.
    Rebalances daily (or on a configured frequency) to maintain equal weight.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        data_dir: str = "data",
        target_cash_buffer_pct: float = 10.0,  # Keep 10% cash, invest 90%
    ):
        if not _is_finite_number(initial_capital) or initial_capital <= 0:
            raise ValueError("initial_capital must be a positive finite number")
        if not _is_finite_number(target_cash_buffer_pct):
            raise ValueError("target_cash_buffer_pct must be a finite number")

        self.initial_capital = float(initial_capital)
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.state_file = self.data_dir / "equalweight_benchmark_state.json"
        self.target_cash_buffer_pct = float(target_cash_buffer_pct)

        self.shares: Dict[str, float] = {}
        self.cash: float = self.initial_capital
        self.start_date: Optional[str] = None
        self.last_rebalanced: Optional[str] = None

        self._load_state()

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return

        # Data-loss class (PR #60): a silent read-fallback here flows into
        # save_state() overwriting this very file at the next rebalance,
        # destroying the corrupt state without a trace. Quarantine instead.
        state = load_json_dict_or_quarantine(self.state_file, context="benchmark state")

        # No-claim boundary (PR #81): an empty result means the file was
        # absent (handled above) or quarantined as unreadable/wrong-shape.
        # Either way the source makes no claim, so the constructor defaults
        # are an honest fresh start — the refusal doctrine below applies
        # only to a file that parsed and therefore claims something.
        if not state:
            return

        # Monetary-base refusal (PR #81 doctrine, three-branch tree): cash is
        # the unit in which every benchmark value is measured — total_value,
        # total_return_pct, and every strategy-vs-benchmark comparison derive
        # from it. It is not an element of a collection (no per-element drop
        # applies), and a file whose JSON parses DOES claim a value (no honest
        # fallback exists: initial_capital fabricates a fresh baseline, 0.0
        # collides with a legitimate all-in state). Worse, rebalance() calls
        # save_state(), so a fabricated cash is persisted on the very next
        # rebalance and silently corrupts every future comparison. Refuse
        # instead: raise before any mutation, file byte-untouched.
        # daily_run.py wraps construction in try/except, so the benchmark
        # section is skipped loud and the trading run is unaffected.
        cash = state.get("cash")
        if not self._is_valid_cash(cash):
            raise ValueError(
                f"Refusing to load benchmark state {self.state_file}: "
                f"'cash' must be a finite non-negative number, got {cash!r}. "
                "File left untouched for manual repair."
            )

        self.shares = self._sanitize_shares(state.get("shares", {}))
        self.cash = float(cash)
        self.start_date = state.get("start_date")
        self.last_rebalanced = state.get("last_rebalanced")

    @staticmethod
    def _is_valid_cash(value: Any) -> bool:
        """Cash invariant: finite, non-bool, and >= 0.

        The lower bound is a healthy-path invariant: rebalance() invests
        (100 - cash_buffer)% and leaves cash = total * buffer_pct/100 >= 0,
        so a negative persisted cash can only come from corruption.
        """
        return _is_finite_number(value) and float(value) >= 0

    def _sanitize_shares(self, shares: Any) -> Dict[str, float]:
        """Return a dict of finite share counts, dropping invalid entries loudly.

        Per-element drop (PR #64 doctrine): a malformed entry invalidates
        only itself, never its healthy siblings. Every drop is one loud
        warning naming the ticker, the offending value, and the reason —
        a silent drop here is invisible until the next save_state()
        persists the pruned dict, destroying the evidence of which
        positions the corrupt file claimed.
        """
        if not isinstance(shares, dict):
            logger.warning(
                "Benchmark state 'shares' is not a mapping (got %r); "
                "treating as no positions.",
                type(shares).__name__,
            )
            return {}

        sanitized: dict[str, float] = {}
        for ticker, quantity in shares.items():
            if not _is_finite_number(quantity):
                logger.warning(
                    "Dropping benchmark share entry %r: quantity %r is not a "
                    "finite number.",
                    ticker,
                    quantity,
                )
                continue
            if float(quantity) < 0:
                logger.warning(
                    "Dropping benchmark share entry %r: quantity %r is "
                    "negative.",
                    ticker,
                    quantity,
                )
                continue
            sanitized[ticker] = float(quantity)
        return sanitized

    def save_state(self) -> None:
        """Persist benchmark state as strict JSON (no NaN/Infinity tokens)."""
        state = {
            "shares": self.shares,
            "cash": self.cash,
            "start_date": self.start_date,
            "last_rebalanced": self.last_rebalanced,
            "initial_capital": self.initial_capital,
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            dump_json_safe(state, f, indent=2)

    def get_value(self, current_prices: Dict[str, float]) -> Dict:
        """Calculate current benchmark value given current prices."""
        positions_value = 0.0
        position_details = {}

        for ticker, shares in self.shares.items():
            price = current_prices.get(ticker)
            if _is_finite_number(price) and price > 0:
                value = shares * price
                positions_value += value
                position_details[ticker] = {
                    "shares": shares,
                    "price": price,
                    "value": value,
                }

        total_value = self.cash + positions_value
        total_return_pct = (
            (total_value - self.initial_capital) / self.initial_capital
        ) * 100

        return {
            "total_value": total_value,
            "cash": self.cash,
            "positions_value": positions_value,
            "total_return_pct": total_return_pct,
            "num_positions": len(self.shares),
            "position_details": position_details,
        }

    def rebalance(self, current_prices: Dict[str, float]) -> Dict:
        """
        Rebalance to equal weight across all available tickers.

        Returns the updated benchmark value.
        """
        available_tickers = [
            t for t in current_prices.keys()
            if _is_finite_number(current_prices[t]) and current_prices[t] > 0
        ]
        n_tickers = len(available_tickers)

        if n_tickers == 0:
            return self.get_value(current_prices)

        # First time initialization
        if not self.start_date:
            self.start_date = datetime.now().strftime("%Y-%m-%d")

        # Calculate total value using only finite prices.
        total_value = self.cash
        for ticker, shares in self.shares.items():
            price = current_prices.get(ticker)
            if _is_finite_number(price):
                total_value += shares * price

        # Target allocation: invest (100 - cash_buffer)% equally across all tickers
        investable_pct = 1.0 - (self.target_cash_buffer_pct / 100)
        target_value_per_ticker = (total_value * investable_pct) / n_tickers

        # Guard against a degenerate total value (should not happen because
        # initial_capital > 0, but protects downstream loops).
        if not _is_finite_number(target_value_per_ticker) or target_value_per_ticker <= 0:
            return self.get_value(current_prices)

        # Rebalance: sell all positions first, then buy equal weight
        self.cash = total_value
        self.shares = {}

        for ticker in available_tickers:
            price = current_prices[ticker]
            shares = target_value_per_ticker / price
            cost = shares * price
            self.shares[ticker] = shares
            self.cash -= cost

        self.last_rebalanced = datetime.now().isoformat()
        self.save_state()

        return self.get_value(current_prices)

    def update(self, current_prices: Dict[str, float]) -> Dict:
        """Update benchmark value without rebalancing (just mark-to-market)."""
        return self.get_value(current_prices)

    def get_daily_summary(self, current_prices: Dict[str, float]) -> Dict:
        """Get a summary for daily reporting."""
        value = self.get_value(current_prices)
        return {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "total_value": value["total_value"],
            "cash": value["cash"],
            "positions_value": value["positions_value"],
            "total_return_pct": value["total_return_pct"],
            "num_positions": value["num_positions"],
            "start_date": self.start_date,
            "last_rebalanced": self.last_rebalanced,
        }
