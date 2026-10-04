"""Pre-fix reproduction: numeric .get defaults in the cooldown/decision
display block of build_prompt launder absence into plausible readings.

Run from repo root:  python3 benchmarks/repro_display_defaults.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm.trading_agent import TradingAgent  # noqa: E402

agent = TradingAgent(api_key="dummy", history_file=Path("/tmp/nonexistent_history.json"))

MARKET = {"assets": {}, "correlations": {}}
PORTFOLIO = {"cash": 0.0, "total_value": 0.0}

# 1) Truncated cooldown dict: budget counters absent.
#    Producers (position_cooldown.get_status, backtest_cooldown.get_status)
#    always populate both keys; a truncated/partial dict fires the defaults.
prompt = agent.build_prompt(MARKET, PORTFOLIO, [], {"active_entries": {}})
print("1) truncated cooldown_status:")
for line in prompt.splitlines():
    if "Weekly trades used" in line:
        print("   ", line.strip())

# 2) Entry with missing hold_days + config without min_hold_days.
prompt = agent.build_prompt(
    MARKET, PORTFOLIO, [],
    {
        "trades_this_week": 1, "weekly_cap": 3,
        "active_entries": {"MC.PA": {"entry_date": "2026-09-01T00:00:00"}},
        "config": {},
    },
)
print("2) missing hold_days / min_hold_days:")
for line in prompt.splitlines():
    if "MC.PA:" in line:
        print("   ", line.strip())

# 3) Exit with missing days_since_exit + config without flip_cooldown_days.
prompt = agent.build_prompt(
    MARKET, PORTFOLIO, [],
    {
        "trades_this_week": 1, "weekly_cap": 3,
        "recent_exits": {"TLT": {"exit_date": "2026-09-15T00:00:00"}},
        "config": {},
    },
)
print("3) missing days_since_exit / flip_cooldown_days:")
for line in prompt.splitlines():
    if "TLT:" in line:
        print("   ", line.strip())

# 4) Historical action without pct (hold contract: pct optional).
prompt = agent.build_prompt(
    MARKET, PORTFOLIO,
    [{"timestamp": "2026-10-02T21:30:00",
      "reasoning": "x",
      "actions": [{"ticker": "SPY", "action": "hold"}]}],
)
print("4) action without pct:")
for line in prompt.splitlines():
    if "SPY:" in line:
        print("   ", line.strip())
