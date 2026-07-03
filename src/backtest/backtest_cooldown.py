"""
Backtest-compatible position cooldown manager.

Mirrors the live trading PositionCooldownManager but uses simulated
dates instead of datetime.now(), enabling counterfactual analysis of
guardrail effectiveness on historical data.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CooldownConfig:
    min_hold_days: int = 5
    flip_cooldown_days: int = 10
    max_trades_per_week: int = 2
    allow_stop_loss_override: bool = True
    stop_loss_threshold_pct: float = 5.0


class BacktestCooldownManager:
    """
    Tracks position entry/exit times and enforces cooldown rules
    within a backtest simulation.

    Unlike the live PositionCooldownManager, this class does NOT
    persist state to disk and accepts an explicit `current_date`
    for all time calculations.
    """

    def __init__(self, config: Optional[CooldownConfig] = None):
        self.config = config or CooldownConfig()

        # Simulated timestamps keyed by ticker
        self.entries: Dict[str, datetime] = {}
        self.exits: Dict[str, datetime] = {}
        self.weekly_trades: List[datetime] = []

        # Metrics for reporting
        self.blocked_sells: int = 0
        self.blocked_buys: int = 0
        self.stop_loss_overrides: int = 0
        self.trade_attempts: int = 0

    def record_entry(self, ticker: str, current_date: datetime) -> None:
        """Record that we entered a position in ticker on current_date."""
        self.entries[ticker] = current_date
        self._record_trade(current_date)

    def record_exit(self, ticker: str, current_date: datetime) -> None:
        """Record that we exited a position in ticker on current_date."""
        self.exits[ticker] = current_date
        if ticker in self.entries:
            del self.entries[ticker]
        self._record_trade(current_date)

    @staticmethod
    def _week_start(dt: datetime) -> datetime:
        """Return the start of the ISO calendar week for dt (Monday 00:00)."""
        return (dt - timedelta(days=dt.weekday())).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

    def _record_trade(self, current_date: datetime) -> None:
        """Record a trade for weekly frequency counting."""
        # Keep only trades in the current ISO calendar week (Monday-Sunday).
        # This aligns the "weekly cap" with the live cooldown manager and the weekly report.
        week_start = self._week_start(current_date)
        self.weekly_trades = [t for t in self.weekly_trades if t >= week_start]
        self.weekly_trades.append(current_date)

    def can_sell(
        self,
        ticker: str,
        current_date: datetime,
        current_price: float,
        avg_price: float
    ) -> Tuple[bool, str]:
        """
        Check if selling ticker is allowed under cooldown rules.

        Returns:
            (allowed: bool, reason: str)
        """
        self.trade_attempts += 1
        
        # Filter expired trades before checking cap (ISO calendar week)
        week_start = self._week_start(current_date)
        self.weekly_trades = [t for t in self.weekly_trades if t >= week_start]

        # Check weekly trade cap
        if len(self.weekly_trades) >= self.config.max_trades_per_week:
            self.blocked_sells += 1
            return (
                False,
                f"Weekly trade cap reached ({self.config.max_trades_per_week})"
            )

        # Check minimum holding period
        entry_time = self.entries.get(ticker)
        if entry_time is None:
            self.blocked_sells += 1
            return (False, f"No entry record for {ticker}")

        hold_days = (current_date - entry_time).total_seconds() / 86400
        if hold_days < self.config.min_hold_days:
            # Check stop-loss override
            if self.config.allow_stop_loss_override and avg_price > 0:
                drawdown_pct = ((current_price - avg_price) / avg_price) * 100
                if drawdown_pct <= -self.config.stop_loss_threshold_pct:
                    self.stop_loss_overrides += 1
                    return (
                        True,
                        f"Stop-loss override (drawdown {drawdown_pct:.1f}%)"
                    )
            self.blocked_sells += 1
            return (
                False,
                f"Minimum hold period not met ({hold_days:.1f} < {self.config.min_hold_days} days)"
            )

        return (True, f"Hold period satisfied ({hold_days:.1f} days)")

    def can_buy(self, ticker: str, current_date: datetime) -> Tuple[bool, str]:
        """
        Check if buying ticker is allowed under flip cooldown.

        Returns:
            (allowed: bool, reason: str)
        """
        self.trade_attempts += 1
        
        # Filter expired trades before checking cap (ISO calendar week)
        week_start = self._week_start(current_date)
        self.weekly_trades = [t for t in self.weekly_trades if t >= week_start]

        # Check weekly trade cap
        if len(self.weekly_trades) >= self.config.max_trades_per_week:
            self.blocked_buys += 1
            return (
                False,
                f"Weekly trade cap reached ({self.config.max_trades_per_week})"
            )

        # Check flip cooldown
        exit_time = self.exits.get(ticker)
        if exit_time is not None:
            days_since_exit = (current_date - exit_time).total_seconds() / 86400
            if days_since_exit < self.config.flip_cooldown_days:
                self.blocked_buys += 1
                return (
                    False,
                    f"Flip cooldown active ({days_since_exit:.1f} < {self.config.flip_cooldown_days} days)"
                )

        return (True, "No cooldown restrictions")

    def get_status(self, current_date: datetime) -> Dict:
        """Return current cooldown status for reporting."""
        week_start = self._week_start(current_date)
        current_week_trades = [t for t in self.weekly_trades if t >= week_start]
        return {
            "active_entries": {
                k: {
                    "entry_date": v.isoformat(),
                    "hold_days": (current_date - v).total_seconds() / 86400,
                }
                for k, v in self.entries.items()
            },
            "recent_exits": {
                k: {
                    "exit_date": v.isoformat(),
                    "days_since_exit": (current_date - v).total_seconds() / 86400,
                }
                for k, v in self.exits.items()
                if (current_date - v).total_seconds() / 86400 < self.config.flip_cooldown_days * 2
            },
            "trades_this_week": len(current_week_trades),
            "weekly_cap": self.config.max_trades_per_week,
            "config": asdict(self.config),
        }

    def get_metrics(self) -> Dict:
        """Return aggregate cooldown metrics for the backtest run."""
        total_blocked = self.blocked_buys + self.blocked_sells
        return {
            "blocked_buys": self.blocked_buys,
            "blocked_sells": self.blocked_sells,
            "total_blocked": total_blocked,
            "stop_loss_overrides": self.stop_loss_overrides,
            "trade_attempts": self.trade_attempts,
            "block_rate": (total_blocked / self.trade_attempts if self.trade_attempts > 0 else 0),
        }
