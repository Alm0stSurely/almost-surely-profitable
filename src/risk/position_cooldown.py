"""
Position cooldown and minimum holding period guardrails.

Prevents overtrading by enforcing:
1. Minimum holding period before selling (default: 5 trading days)
2. Flip cooldown: cannot re-enter a ticker within N days of exiting (default: 10 days)
3. Weekly trade frequency cap (default: 2 non-hold actions)

These constraints are inspired by the observation that the LLM-driven
strategy exhibits ~318 trades/year with a 4.5% round-trip win rate.
Reducing trade frequency should improve signal-to-noise ratio.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict


@dataclass
class CooldownConfig:
    min_hold_days: int = 5
    flip_cooldown_days: int = 10
    max_trades_per_week: int = 2
    allow_stop_loss_override: bool = True
    stop_loss_threshold_pct: float = 5.0


class PositionCooldownManager:
    """
    Tracks position entry/exit times and enforces cooldown rules.
    
    Persists state to data/position_cooldowns.json.
    """

    def __init__(
        self,
        data_dir: str = "data",
        config: Optional[CooldownConfig] = None
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.state_file = self.data_dir / "position_cooldowns.json"
        self.config = config or CooldownConfig()

        self.entries: Dict[str, datetime] = {}
        self.exits: Dict[str, datetime] = {}
        self.weekly_trades: List[datetime] = []

        self._load_state()

    def _load_state(self) -> None:
        if not self.state_file.exists():
            return
        try:
            with open(self.state_file, "r") as f:
                state = json.load(f)
            self.entries = {
                k: datetime.fromisoformat(v)
                for k, v in state.get("entries", {}).items()
            }
            self.exits = {
                k: datetime.fromisoformat(v)
                for k, v in state.get("exits", {}).items()
            }
            self.weekly_trades = [
                datetime.fromisoformat(v)
                for v in state.get("weekly_trades", [])
            ]
        except Exception:
            self.entries = {}
            self.exits = {}
            self.weekly_trades = []

    def save_state(self) -> None:
        state = {
            "entries": {k: v.isoformat() for k, v in self.entries.items()},
            "exits": {k: v.isoformat() for k, v in self.exits.items()},
            "weekly_trades": [v.isoformat() for v in self.weekly_trades],
            "last_updated": datetime.now().isoformat(),
        }
        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def record_entry(self, ticker: str) -> None:
        """Record that we entered a position in ticker."""
        self.entries[ticker] = datetime.now()
        self._record_trade()

    def record_exit(self, ticker: str) -> None:
        """Record that we exited a position in ticker."""
        self.exits[ticker] = datetime.now()
        if ticker in self.entries:
            del self.entries[ticker]
        self._record_trade()

    def _record_trade(self) -> None:
        """Record a trade for weekly frequency counting."""
        now = datetime.now()
        # Keep only trades from the last 7 days
        cutoff = now - timedelta(days=7)
        self.weekly_trades = [t for t in self.weekly_trades if t > cutoff]
        self.weekly_trades.append(now)

    def can_sell(
        self,
        ticker: str,
        current_price: float,
        avg_price: float
    ) -> Tuple[bool, str]:
        """
        Check if selling ticker is allowed under cooldown rules.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check weekly trade cap
        if len(self.weekly_trades) >= self.config.max_trades_per_week:
            return (
                False,
                f"Weekly trade cap reached ({self.config.max_trades_per_week})"
            )

        # Check minimum holding period
        entry_time = self.entries.get(ticker)
        if entry_time is None:
            return (False, f"No entry record for {ticker}")

        hold_days = (datetime.now() - entry_time).total_seconds() / 86400
        if hold_days < self.config.min_hold_days:
            # Check stop-loss override
            if self.config.allow_stop_loss_override and avg_price > 0:
                drawdown_pct = ((current_price - avg_price) / avg_price) * 100
                if drawdown_pct <= -self.config.stop_loss_threshold_pct:
                    return (
                        True,
                        f"Stop-loss override (drawdown {drawdown_pct:.1f}%)"
                    )
            return (
                False,
                f"Minimum hold period not met ({hold_days:.1f} < {self.config.min_hold_days} days)"
            )

        return (True, f"Hold period satisfied ({hold_days:.1f} days)")

    def can_buy(self, ticker: str) -> Tuple[bool, str]:
        """
        Check if buying ticker is allowed under flip cooldown.
        
        Returns:
            (allowed: bool, reason: str)
        """
        # Check weekly trade cap
        if len(self.weekly_trades) >= self.config.max_trades_per_week:
            return (
                False,
                f"Weekly trade cap reached ({self.config.max_trades_per_week})"
            )

        # Check flip cooldown
        exit_time = self.exits.get(ticker)
        if exit_time is not None:
            days_since_exit = (datetime.now() - exit_time).total_seconds() / 86400
            if days_since_exit < self.config.flip_cooldown_days:
                return (
                    False,
                    f"Flip cooldown active ({days_since_exit:.1f} < {self.config.flip_cooldown_days} days)"
                )

        return (True, "No cooldown restrictions")

    def get_status(self) -> Dict:
        """Return current cooldown status for reporting."""
        now = datetime.now()
        return {
            "active_entries": {
                k: {
                    "entry_date": v.isoformat(),
                    "hold_days": (now - v).total_seconds() / 86400,
                }
                for k, v in self.entries.items()
            },
            "recent_exits": {
                k: {
                    "exit_date": v.isoformat(),
                    "days_since_exit": (now - v).total_seconds() / 86400,
                }
                for k, v in self.exits.items()
                if (now - v).total_seconds() / 86400 < self.config.flip_cooldown_days * 2
            },
            "trades_this_week": len(self.weekly_trades),
            "weekly_cap": self.config.max_trades_per_week,
            "config": asdict(self.config),
        }


def main():
    """Quick demo."""
    mgr = PositionCooldownManager()
    print("PositionCooldownManager initialized")
    print(f"Config: {mgr.config}")
    print(f"Status: {json.dumps(mgr.get_status(), indent=2)}")


if __name__ == "__main__":
    main()
