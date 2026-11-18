"""
LLM outage escalation tracker.

The daily pipeline fail-opens to hold-all when the LLM decision call fails
(``decision.error == True`` in the daily result). That fallback is safe for
one day, but it is silent: nothing distinguishes "the LLM looked at the
market and chose to hold" from "the LLM was never reached". Repeated
consecutive failures mean the portfolio drifts unmanaged — stop overrides,
regime checks and cash-band reasoning are all skipped by the fallback.

This module turns the silent fail-open into a counted, escalating state:

  - Every daily result (``results/daily/YYYY-MM-DD.json``) whose live-arm
    decision carries ``error: true`` counts as one outage day.
  - When the count of *consecutive* outage days (ending today) reaches
    ``OUTAGE_ALERT_DAYS`` (default 2, env-overridable), the run is in
    ALERT state: the daily result gains an ``llm_outage`` block and a
    persistent ``results/llm_outage_status.json`` file is rewritten so
    external monitors can poll a single artifact instead of scanning the
    whole daily-results directory.

Failure direction: a corrupt or unreadable daily file is *skipped loudly*
and breaks the consecutive run (we cannot prove that day was an outage
day, so we must not extend the streak across an unknown). Missing files
for calendar days break the streak the same way — the streak is over
*trading sessions we can observe*, and a gap in observed sessions is not
an observed non-outage session.

Only the live arm is counted. A shadow-arm-only failure (logged in
``data/shadow_decisions.json``) leaves the live decision intact and must
not escalate the pipeline state.
"""

import json
import logging
import os
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

#: Default number of consecutive live-arm error days that triggers ALERT.
DEFAULT_ALERT_DAYS = 2

#: Strict daily-result filename contract: YYYY-MM-DD.json (dry runs excluded).
_DAILY_FILE_RE = re.compile(r"^(\d{4})-(\d{2})-(\d{2})\.json$")


def _alert_threshold() -> int:
    """Read the escalation threshold from the environment with validation."""
    raw = os.getenv("OUTAGE_ALERT_DAYS")
    if raw is None:
        return DEFAULT_ALERT_DAYS
    try:
        value = int(raw)
    except ValueError:
        logger.warning(
            "OUTAGE_ALERT_DAYS=%r is not an integer; falling back to %d",
            raw,
            DEFAULT_ALERT_DAYS,
        )
        return DEFAULT_ALERT_DAYS
    if value < 1:
        logger.warning(
            "OUTAGE_ALERT_DAYS=%d must be >= 1; falling back to %d",
            value,
            DEFAULT_ALERT_DAYS,
        )
        return DEFAULT_ALERT_DAYS
    return value


def iter_daily_results(results_dir: Path) -> List[Dict]:
    """Collect (date, decision_error) for every parseable daily result.

    Returns a list of ``{"date": date, "error": bool}`` sorted ascending
    by date. Unreadable or malformed files are skipped with a loud log
    line and never raise: this is a diagnostic scanner, not an ingress
    validator.
    """
    if not results_dir.is_dir():
        return []

    entries: List[Dict] = []
    for path in sorted(results_dir.iterdir()):
        match = _DAILY_FILE_RE.match(path.name)
        if not match:
            continue
        try:
            day = date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        except ValueError:
            logger.warning("Outage scan: skipping file with invalid date %s", path.name)
            continue
        try:
            with open(path) as fh:
                payload = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Outage scan: skipping unreadable result %s: %s", path.name, exc)
            continue
        if not isinstance(payload, dict):
            logger.warning("Outage scan: skipping non-dict result %s", path.name)
            continue
        decision = payload.get("decision")
        # Absent decision block is not an outage signal — the file may be
        # a partial artefact from a crashed run; do not extend the streak.
        error = bool(isinstance(decision, dict) and decision.get("error"))
        entries.append({"date": day, "error": error})
    entries.sort(key=lambda e: e["date"])
    return entries


def count_consecutive_error_days(
    results_dir: Path, today: Optional[date] = None
) -> Dict:
    """Count consecutive live-arm outage days ending at ``today``.

    The streak walks backwards from ``today`` over observed daily results.
    A calendar day with no result file breaks the streak (an unobserved
    day is not evidence of a healthy run — e.g. the cron itself may be
    down, which is a worse failure mode worth alerting on after resume).

    Returns a dict::

        {
          "consecutive_error_days": int,   # streak length ending today
          "today_is_error": bool,          # today's file has error: true
          "today_observed": bool,          # a result file exists for today
          "streak_dates": [date, ...],     # the outage days in the streak
        }
    """
    today = today or date.today()
    entries = iter_daily_results(results_dir)

    by_date = {e["date"]: e["error"] for e in entries if e["date"] <= today}
    streak: List[date] = []
    cursor = today

    today_observed = today in by_date
    today_is_error = bool(by_date.get(today, False))

    if today_is_error:
        streak.append(today)
        cursor -= timedelta(days=1)
        while cursor in by_date and by_date[cursor]:
            streak.append(cursor)
            cursor -= timedelta(days=1)

    return {
        "consecutive_error_days": len(streak),
        "today_is_error": today_is_error,
        "today_observed": today_observed,
        "streak_dates": [d.isoformat() for d in streak],
    }


def assess_outage(
    results_dir: Path,
    today: Optional[date] = None,
    today_error: Optional[bool] = None,
) -> Dict:
    """Full outage assessment for the daily pipeline.

    Adds the escalation state on top of the raw streak:

      - ``threshold``: the configured OUTAGE_ALERT_DAYS.
      - ``alert``: True when the streak has reached the threshold.
      - ``state``: "OK" | "DEGRADED" (streak >= 1, below threshold) |
        "ALERT" (streak >= threshold).

    ``today_error`` lets callers assess a run whose result file has not been
    persisted yet (the daily pipeline builds the outage block *before*
    writing the daily JSON). When provided, on-disk results are consulted
    only for days strictly before ``today`` and the in-memory flag is used
    for ``today`` itself. When None, today's on-disk file is used as-is.

    The caller decides what to do with ALERT (loud banner, status file,
    non-zero exit for monitors). This function never raises on missing or
    corrupt data — an unassessable history is reported as a zero streak,
    because escalating on unverifiable data would itself be a false alarm.
    """
    threshold = _alert_threshold()
    today = today or date.today()

    if today_error is None:
        streak = count_consecutive_error_days(results_dir, today=today)
        days = streak["consecutive_error_days"]
        streak_dates = streak["streak_dates"]
        today_is_error = streak["today_is_error"]
        today_observed = streak["today_observed"]
    else:
        # In-memory mode: history is everything strictly before today.
        prior = count_consecutive_error_days(results_dir, today=today - timedelta(days=1))
        today_is_error = bool(today_error)
        today_observed = True  # the caller IS today's run
        if today_is_error:
            days = prior["consecutive_error_days"] + 1
            streak_dates = [today.isoformat(), *prior["streak_dates"]]
        else:
            days = 0
            streak_dates = []

    if days >= threshold:
        state = "ALERT"
    elif days >= 1:
        state = "DEGRADED"
    else:
        state = "OK"

    return {
        "consecutive_error_days": days,
        "today_is_error": today_is_error,
        "today_observed": today_observed,
        "streak_dates": streak_dates,
        "threshold": threshold,
        "state": state,
        "alert": state == "ALERT",
    }


def write_outage_status(status: Dict, results_dir: Path) -> Path:
    """Persist the assessment to ``results/llm_outage_status.json``.

    Returns the path written. The file is rewritten on every run so its
    mtime doubles as a pipeline-liveness signal for external monitors.
    """
    results_dir.mkdir(parents=True, exist_ok=True)
    target = results_dir / "llm_outage_status.json"
    with open(target, "w") as fh:
        json.dump(status, fh, indent=2, default=str)
        fh.write("\n")
    return target
