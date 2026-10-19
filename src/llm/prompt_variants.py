"""
Prompt variants for controlled experiments (H-series).

Each variant is a *pure function* of the base SYSTEM_PROMPT so that:
  - the live prompt constant is never mutated,
  - the applied delta is reviewable diff-by-diff against the design doc,
  - a drifted base prompt fails loudly (ValueError) instead of running a
    half-applied variant.

Active experiment:
  H4 mid-band deployment language
  (docs/experiments/2026-10-16-prompt-midband-deployment.md)

Per LEARNINGS (2026-09-28): anchor strings must be single-line fragments of
the literal — multi-line assertions break on wrapping/indentation.
"""

MIDBAND_DEPLOYMENT_VARIANT = "midband_deployment"

# Single-line anchor fragments (see LEARNINGS 2026-09-28).
_RULE5_ANCHOR = "waiting for a perfect signal."
_DEFAULT_HOLD_ANCHOR = "- When in doubt, default to HOLD."
_PREMATURE_PROFIT_FRAGMENT = "Premature profit-taking is a form of loss aversion"

# New content inserted by Edit A (governed mid-band deployment bullet).
EDIT_A_BULLET = (
    "\n"
    "   - When cash is WITHIN the regime band (mid-band) and no primary signal\n"
    "     fires (no RSI extreme, no Bollinger break, no confirmed trend change),\n"
    "     cash is capital awaiting deployment, not a comfort buffer. Evaluate\n"
    "     the highest-conviction setup available: the position or candidate\n"
    "     with the best-aligned trend + RSI/Bollinger evidence, including\n"
    "     partial alignment (uptrend + neutral RSI, or neutral trend + RSI\n"
    "     extreme). If a highest-conviction setup exists, deploy a moderate\n"
    "     pct (10-20% of portfolio value) into it. \"Cash is idle\" is never\n"
    "     sufficient justification by itself — name the conviction evidence."
)

# Edit B replacement for the default-HOLD line (keeps the premature-profit
# clause that shares the bullet).
EDIT_B_REPLACEMENT = (
    "- When in doubt AND cash is below the regime lower bound, default to\n"
    "     HOLD. When cash is mid-band, prefer governed deployment (section 5)\n"
    "     over reflexive inaction."
)

# Exported marker tuples for downstream audit/logging.
VARIANT_EDIT_A_MARKERS = (
    "cash is capital awaiting deployment, not a comfort buffer",
    "highest-conviction setup",
)
VARIANT_EDIT_B_MARKERS = (
    "When in doubt AND cash is below the regime lower bound, default to HOLD",
    "prefer governed deployment (section 5) over reflexive inaction",
)


def apply_midband_deployment_delta(base_prompt: str) -> str:
    """Return *base_prompt* with H4 Edits A and B applied.

    Edit A — section 5: new bullet after the Rule-5 mandate giving governed
    deployment language to the mid-band zone (cash within the regime band,
    no primary signal firing).

    Edit B — section 10: the default-HOLD line is amended so reflexive
    inaction applies only below the regime lower bound; mid-band prefers
    governed deployment.

    Raises:
        ValueError: if either single-line anchor is absent from
            *base_prompt* (prompt drift). A half-applied variant would
            silently change the experiment's meaning — never run one.
    """
    if not isinstance(base_prompt, str) or not base_prompt:
        raise ValueError("Edit A/B: base prompt must be a non-empty string")

    if _RULE5_ANCHOR not in base_prompt:
        raise ValueError(
            f"Edit A anchor missing from base prompt: {_RULE5_ANCHOR!r} "
            "(prompt drifted? refuse to run a half-applied variant)"
        )
    if _DEFAULT_HOLD_ANCHOR not in base_prompt:
        raise ValueError(
            f"Edit B anchor missing from base prompt: {_DEFAULT_HOLD_ANCHOR!r} "
            "(prompt drifted? refuse to run a half-applied variant)"
        )

    variant = base_prompt.replace(
        _RULE5_ANCHOR,
        _RULE5_ANCHOR + EDIT_A_BULLET,
        1,
    )

    # Edit B: replace the whole old sentence, preserving the premature-profit
    # clause that follows it on the same bullet.
    old_line = (
        _DEFAULT_HOLD_ANCHOR
        + " "
        + _PREMATURE_PROFIT_FRAGMENT
    )
    new_line = EDIT_B_REPLACEMENT + " " + _PREMATURE_PROFIT_FRAGMENT
    if old_line not in variant:
        raise ValueError(
            "Edit B: default-HOLD anchor found but the premature-profit "
            "continuation is missing (prompt drifted?)"
        )
    variant = variant.replace(old_line, new_line, 1)

    return variant
