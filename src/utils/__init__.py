"""Utilities for loading and validating daily trading results."""
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


MIN_ASSETS_FOR_VALID_RUN = 5


def is_valid_daily_result(data: Dict) -> bool:
    """Return True if a daily result dict should be used for analysis.

    Filters out:
    - Dry-run results (explicit ``dry_run: true`` or ``_dry_run`` filename).
    - Test/placeholder artifacts (e.g. portfolio reset to 10 000 EUR with no
      positions and reasoning containing "HOLD for test").
    - Runs with too few assets analyzed (likely partial tests).
    """
    if data.get("dry_run", False):
        return False

    reasoning = (data.get("decision", {}).get("reasoning", "") or "").lower()
    if "test" in reasoning or "placeholder" in reasoning:
        return False

    market_summary = data.get("market_summary")
    if market_summary is not None:
        assets = market_summary.get("assets_analyzed", 0)
        if assets < MIN_ASSETS_FOR_VALID_RUN:
            return False

    return True


def load_valid_daily_results(
    results_dir: str = "results/daily",
    pattern: str = "*.json",
    skip_dry_run_files: bool = True,
) -> List[Dict]:
    """Load all valid daily results from ``results_dir``.

    Returns results sorted by filename (which is date-ordered for ISO dates).
    """
    path = Path(results_dir)
    if not path.exists():
        return []

    results: List[Dict] = []
    for file in sorted(path.glob(pattern)):
        if skip_dry_run_files and "_dry_run" in file.name:
            continue
        try:
            with open(file) as f:
                data = json.load(f)
            if is_valid_daily_result(data):
                results.append(data)
        except Exception:
            continue
    return results


def load_valid_daily_results_limited(
    results_dir: str = "results/daily",
    days: int = 30,
) -> List[Dict]:
    """Load the most recent ``days`` valid daily results."""
    all_results = load_valid_daily_results(results_dir)
    return all_results[-days:]


def _is_finite_number(value: Any) -> bool:
    """Return True if *value* is a finite scalar number."""
    if value is None or isinstance(value, bool) or isinstance(value, str):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def sanitize_for_json(obj: Any) -> Any:
    """Recursively replace non-finite floats in dicts/lists with ``None``.

    Integers, strings, and finite floats are preserved. Non-finite floats
    (``NaN``, ``Infinity``, ``-Infinity``) become ``None`` so that downstream
    JSON consumers never have to parse non-standard tokens such as ``NaN``.

    This is a defensive last line of defense: upstream modules should still
    validate their own numeric outputs.
    """
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(sanitize_for_json(v) for v in obj)
    if isinstance(obj, float):
        return obj if _is_finite_number(obj) else None
    return obj


def dump_json_safe(
    obj: Any,
    f,
    indent: int = 2,
    default=None,
    **kwargs,
) -> None:
    """Serialize *obj* to JSON using ``allow_nan=False``.

    Non-finite floats are sanitized to ``None`` before writing so that a single
    missed guard does not crash the pipeline or produce invalid JSON. Any
    remaining serialization error is re-raised with the object type to help
    debugging.
    """
    cleaned = sanitize_for_json(obj)
    try:
        json.dump(cleaned, f, indent=indent, default=default, allow_nan=False, **kwargs)
    except ValueError as exc:
        raise ValueError(f"JSON serialization failed for {type(obj).__name__}: {exc}") from exc


def _quarantine_file(path: Path) -> Tuple[Optional[Path], Optional[OSError]]:
    """Rename *path* aside as ``<name>.corrupt-<timestamp>``.

    Returns ``(quarantine_path, None)`` on success, or ``(None, error)`` when
    the rename itself failed (e.g. filesystem errors). The caller owns the
    logging and the fresh-start decision; either way the corrupt file no
    longer sits at its original location to be silently overwritten by the
    next save.
    """
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S-%f")
    quarantine = path.with_name(f"{path.name}.corrupt-{stamp}")
    try:
        path.rename(quarantine)
        return quarantine, None
    except OSError as exc:
        return None, exc


def load_json_list_or_quarantine(path, *, context: str = "ledger") -> List[Any]:
    """Load a JSON list file for an append-then-overwrite flow.

    Append-then-overwrite flows (trade ledger, decision history) read the whole
    backing file, append the new record, and rewrite it. If the read silently
    falls back to ``[]`` on corruption, the rewrite *truncates the ledger to a
    single record* — the inconsistency destroys the evidence. A corrupt ledger
    is data to preserve, not an empty list.

    On any read/parse/shape failure the file is renamed to
    ``<name>.corrupt-<timestamp>`` (preserved for manual recovery) and an empty
    list is returned so the caller can persist the new record without
    destroying history. The quarantine makes the failure loud: an error is
    logged and the stray ``.corrupt-*`` file is visible to the nightly
    reconcile / a human.

    If the quarantine rename itself fails (e.g. filesystem errors), the error
    is logged and an empty list is still returned; the caller's subsequent
    write will then surface any real filesystem problem at the write site.
    """
    path = Path(path)
    if not path.exists():
        return []

    data: Any = None
    reason = "unparseable"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        reason = f"wrong shape (expected list, got {type(data).__name__})"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        reason = f"unreadable ({type(exc).__name__}: {exc})"

    if not isinstance(data, list):
        quarantine, quarantine_err = _quarantine_file(path)
        if quarantine is not None:
            logger.error(
                "Corrupt %s %s (%s) quarantined to %s; starting fresh ledger "
                "(previous records preserved for recovery).",
                context, path, reason, quarantine,
            )
        else:
            logger.error(
                "Corrupt %s %s (%s) could not be quarantined (%s); proceeding "
                "with fresh ledger.",
                context, path, reason, quarantine_err,
            )
        return []

    return data


def load_json_dict_or_quarantine(path, *, context: str = "state") -> Dict[str, Any]:
    """Load a JSON dict file for a read-modify-write state cache.

    Same data-loss class as :func:`load_json_list_or_quarantine`, for dict-
    shaped state (alert history, market state): the caller loads the dict,
    mutates or derives from it, and rewrites the same path within the run. A
    silent read-fallback lets that rewrite destroy the corrupt file — the
    evidence of the inconsistency — without a trace.

    On any read/parse failure, or when the file holds valid JSON that is not
    a dict (including ``null``, a list, or a scalar), the file is quarantined
    and an empty dict is returned. The failure is loud (error log + stray
    ``.corrupt-*`` file) and recoverable (original bytes preserved).
    """
    path = Path(path)
    if not path.exists():
        return {}

    data: Any = None
    reason = "unparseable"
    try:
        with open(path, "r") as f:
            data = json.load(f)
        reason = f"wrong shape (expected dict, got {type(data).__name__})"
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
        reason = f"unreadable ({type(exc).__name__}: {exc})"

    if not isinstance(data, dict):
        quarantine, quarantine_err = _quarantine_file(path)
        if quarantine is not None:
            logger.error(
                "Corrupt %s %s (%s) quarantined to %s; starting fresh state "
                "(previous contents preserved for recovery).",
                context, path, reason, quarantine,
            )
        else:
            logger.error(
                "Corrupt %s %s (%s) could not be quarantined (%s); proceeding "
                "with fresh state.",
                context, path, reason, quarantine_err,
            )
        return {}

    return data
