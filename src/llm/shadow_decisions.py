"""
Shadow-mode arm for prompt-variant experiments (H4 Phase 1).

Each evening the daily pipeline asks the live LLM for a decision. When a
variant experiment is in ``shadow`` mode, this module runs the SAME user
prompt against the VARIANT system prompt and logs the hypothetical decision
to ``data/shadow_decisions.json``. Hard invariants:

  - never executes trades (no portfolio mutation, no cooldown consumption),
  - never writes to ``decision_history.json`` (the live LLM's recent-decision
    context must not be contaminated with hypothetical decisions),
  - best-effort: a shadow failure is logged to the shadow log itself and
    never aborts the daily pipeline.

Experiment design: docs/experiments/2026-10-16-prompt-midband-deployment.md
"""

import json
import logging
import math
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from utils import dump_json_safe, load_json_list_or_quarantine
    JSON_SAFE_AVAILABLE = True
except ImportError:  # pragma: no cover - standalone fallback
    JSON_SAFE_AVAILABLE = False

from .prompt_variants import MIDBAND_DEPLOYMENT_VARIANT

logger = logging.getLogger(__name__)

DEFAULT_SHADOW_LOG_NAME = "shadow_decisions.json"
CONFIG_CONTEXT = "experiments config"
LOG_CONTEXT = "shadow decisions"

# Conviction-evidence keywords for protocol-violation detection (metric 3.4.5
# of the experiment design). A variant BUY whose reasoning cites none of
# these is "idle-cash deployment" — a protocol violation, logged as a flag
# (never blocked: shadow mode observes, it does not enforce).
CONVICTION_EVIDENCE_KEYWORDS = (
    "rsi", "bollinger", "trend", "sma", "support", "resistance",
    "momentum", "regime", "cvar", "correlation", "breakout",
    "moving average", "mean reversion", "oversold", "overbought",
    "drawdown", "volatility",
)


@dataclass
class ShadowConfig:
    """Resolved experiment config for the shadow arm."""

    enabled: bool = False
    mode: str = "off"                     # "off" | "shadow" | "live"
    variant: str = MIDBAND_DEPLOYMENT_VARIANT
    log_path: Optional[Path] = None       # resolved by the caller


@dataclass
class ShadowArmResult:
    """Outcome of one shadow-arm invocation."""

    logged: bool = False
    error: bool = False
    record: Dict[str, Any] = field(default_factory=dict)


def _is_finite_number(value) -> bool:
    if value is None or isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def load_shadow_config(path) -> ShadowConfig:
    """Load ``config/experiments.json`` and resolve the shadow arm's config.

    Failure direction: any unreadable/wrong-shape config disables the arm
    (``enabled=False``) with a loud error log. A shadow arm that mis-fires
    costs an extra API call per day; a shadow arm that silently runs against
    a misread config corrupts the experiment. Choosing the conservative
    side matches the live-manager convention.
    """
    path = Path(path)
    if not path.exists():
        return ShadowConfig(enabled=False, mode="off")

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except (ValueError, UnicodeDecodeError, OSError) as exc:
        logger.error(
            "Shadow config %s unreadable (%s: %s); shadow arm DISABLED.",
            path, type(exc).__name__, exc,
        )
        return ShadowConfig(enabled=False, mode="off")

    if not isinstance(data, dict):
        logger.error(
            "Shadow config %s wrong shape (expected dict, got %s); shadow arm DISABLED.",
            path, type(data).__name__,
        )
        return ShadowConfig(enabled=False, mode="off")

    exp = data.get(MIDBAND_DEPLOYMENT_VARIANT)
    if not isinstance(exp, dict):
        return ShadowConfig(enabled=False, mode="off")

    mode = exp.get("mode", "off")
    if not isinstance(mode, str):
        logger.error(
            "Shadow config %s: mode must be a string, got %s; shadow arm DISABLED.",
            path, type(mode).__name__,
        )
        return ShadowConfig(enabled=False, mode="off")

    return ShadowConfig(enabled=(mode == "shadow"), mode=mode)


def summarize_actions(actions) -> str:
    """Compact one-line summary of a decision's actions for the diff audit."""
    if not actions:
        return "no actions"
    parts: List[str] = []
    for a in actions:
        if not isinstance(a, dict):
            continue
        ticker = a.get("ticker", "?")
        act = a.get("action", "?")
        if act == "buy":
            pct = a.get("pct", 0)
            parts.append(f"{ticker} buy {pct}%")
        elif act == "sell":
            pct = a.get("pct", 0)
            parts.append(f"{ticker} sell {pct}%")
        else:
            parts.append(f"{ticker} hold")
    return ", ".join(parts) if parts else "no actions"


def _detect_protocol_violation(decision: Dict) -> bool:
    """Flag variant BUYs whose reasoning cites no conviction evidence.

    Metric 3.4.5 of the experiment design: deployment must name concrete
    trend/RSI/Bollinger evidence. Log-only heuristic — shadow mode never
    blocks. HOLD decisions cannot violate (inaction is always in-protocol).
    """
    actions = decision.get("actions") or []
    has_buy = any(
        isinstance(a, dict) and a.get("action") == "buy" for a in actions
    )
    if not has_buy:
        return False
    reasoning = str(decision.get("reasoning") or "").lower()
    return not any(kw in reasoning for kw in CONVICTION_EVIDENCE_KEYWORDS)


def build_shadow_record(
    *,
    variant: str,
    decision: Dict,
    portfolio_summary: Dict,
    vol_regime: str,
    live_actions_summary: str,
) -> Dict[str, Any]:
    """Assemble one shadow-decision record for the append-only log.

    Guards (empty-set / non-finite family): cash_pct is None when the
    portfolio summary cannot support the ratio — a missing measurement
    stays absent rather than laundering into a fictitious 0.0.
    """
    cash = (portfolio_summary or {}).get("cash")
    total = (portfolio_summary or {}).get("total_value")
    if _is_finite_number(cash) and _is_finite_number(total) and total > 0:
        cash_pct = round(float(cash) / float(total) * 100, 2)
    else:
        cash_pct = None

    return {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "timestamp": datetime.now().isoformat(),
        "variant": variant,
        "mode": "shadow",
        "cash_pct": cash_pct,
        "vol_regime": vol_regime if isinstance(vol_regime, str) else "unknown",
        "live_actions_summary": live_actions_summary,
        "actions": decision.get("actions") or [],
        "reasoning": decision.get("reasoning", ""),
        "protocol_violation": _detect_protocol_violation(decision),
        "error": bool(decision.get("error", False)),
    }


def append_shadow_record(path, record: Dict[str, Any]) -> None:
    """Append *record* to the shadow log (append-then-overwrite).

    Corrupt-log failure direction: ``load_json_list_or_quarantine``
    preserves the corrupt file (``.corrupt-*`` rename, loud error) and
    returns [] so the new record persists without destroying evidence.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if JSON_SAFE_AVAILABLE:
        records = load_json_list_or_quarantine(path, context=LOG_CONTEXT)
    else:  # pragma: no cover - standalone fallback
        records = []
        if path.exists():
            try:
                with open(path, "r") as f:
                    records = json.load(f)
            except (ValueError, OSError):
                logger.error("Shadow log %s unreadable; starting fresh.", path)
                records = []
        if not isinstance(records, list):
            records = []

    records.append(record)
    if JSON_SAFE_AVAILABLE:
        with open(path, "w") as f:
            dump_json_safe(records, f)
    else:  # pragma: no cover
        with open(path, "w") as f:
            json.dump(records, f, indent=2)


def run_shadow_arm(
    *,
    agent,
    market_analysis: Dict,
    portfolio_summary: Dict,
    cooldown_status: Optional[Dict],
    vol_regime: str,
    log_path,
    live_actions_summary: str,
) -> ShadowArmResult:
    """Run the variant prompt once and log the hypothetical decision.

    Never raises: any failure (LLM error, parse error, log write error) is
    captured into an error record so the gap is visible in the log itself
    rather than a silent missing day.
    """
    log_path = Path(log_path)
    try:
        decision = agent.get_shadow_decision(
            market_data=market_analysis,
            portfolio_summary=portfolio_summary,
            cooldown_status=cooldown_status,
        )
    except Exception as exc:  # noqa: BLE001 - best-effort arm, never aborts pipeline
        logger.error("Shadow arm LLM call failed (%s: %s); logging error record.",
                     type(exc).__name__, exc)
        decision = {
            "actions": [],
            "reasoning": "",
            "error": True,
            "exception": f"{type(exc).__name__}: {exc}",
        }

    record = build_shadow_record(
        variant=MIDBAND_DEPLOYMENT_VARIANT,
        decision=decision,
        portfolio_summary=portfolio_summary,
        vol_regime=vol_regime,
        live_actions_summary=live_actions_summary,
    )
    try:
        append_shadow_record(log_path, record)
    except Exception as exc:  # noqa: BLE001
        logger.error("Shadow arm failed to log to %s (%s: %s).",
                     log_path, type(exc).__name__, exc)
        return ShadowArmResult(logged=False, error=True, record=record)

    return ShadowArmResult(logged=True, error=bool(record["error"]), record=record)
