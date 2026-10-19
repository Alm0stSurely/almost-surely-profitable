"""
Tests for the H4 mid-band deployment prompt variant (shadow mode).

Experiment design: docs/experiments/2026-10-16-prompt-midband-deployment.md

The variant is a *pure function* of the base SYSTEM_PROMPT: two surgical
edits (A: new mid-band bullet in section 5 after the Rule-5 mandate; B:
amend the default-HOLD line in section 10). Everything else must be
byte-identical. A drifted base prompt must fail loudly (ValueError), never
run a half-applied variant.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.prompt_variants import (
    MIDBAND_DEPLOYMENT_VARIANT,
    VARIANT_EDIT_A_MARKERS,
    VARIANT_EDIT_B_MARKERS,
    apply_midband_deployment_delta,
)
from llm.trading_agent import SYSTEM_PROMPT


class TestApplyMidbandDeploymentDelta:
    """Edit A and Edit B are applied exactly as designed."""

    def test_edit_a_inserts_midband_bullet(self):
        variant = apply_midband_deployment_delta(SYSTEM_PROMPT)
        # New governed-deployment language present in the variant only.
        assert "cash is capital awaiting deployment, not a comfort buffer" in variant
        assert "highest-conviction setup" in variant
        # Edit A anchors to the end of the Rule-5 mandate line.
        assert "waiting for a perfect signal." in variant

    def test_edit_b_amends_default_hold(self):
        variant = apply_midband_deployment_delta(SYSTEM_PROMPT)
        # Single-line fragments only (LEARNINGS 2026-09-28): the replacement
        # wraps across lines, so assert per-line pieces.
        assert "When in doubt AND cash is below the regime lower bound, default to" in variant
        assert "HOLD. When cash is mid-band, prefer governed deployment (section 5)" in variant
        assert "over reflexive inaction." in variant

    def test_base_default_hold_wording_removed(self):
        """The amended line must replace, not duplicate, the old default."""
        variant = apply_midband_deployment_delta(SYSTEM_PROMPT)
        # Old wording gone (the sentence continues with the premature-profit
        # clause, which must survive).
        assert "- When in doubt, default to HOLD." not in variant
        assert "Premature profit-taking is a form of loss aversion" in variant

    def test_other_sections_untouched(self):
        """Everything outside the two edit sites is byte-identical."""
        variant = apply_midband_deployment_delta(SYSTEM_PROMPT)
        for section_marker in [
            "1. LOSS AVERSION",
            "2. RISK SENSITIVITY",
            "3. MOMENTUM & MEAN REVERSION",
            "4. DEFLATED SHARPE RATIO MINDSET",
            "6. DIVERSIFICATION",
            "7. STOP LOSS MENTALITY",
            "8. META-LABELING PRINCIPLE",
            "9. MARKET REGIME ADAPTATION",
            "OUTPUT FORMAT",
        ]:
            assert section_marker in variant, f"section lost: {section_marker}"
        # Rule-5 mandate itself unchanged (anchor text still there verbatim).
        assert "you are under-invested and" in variant
        assert "Do not let cash sit above the upper bound while" in variant

    def test_base_prompt_constant_unchanged(self):
        """The variant is a derived string; the module constant is immutable."""
        snapshot = SYSTEM_PROMPT
        apply_midband_deployment_delta(SYSTEM_PROMPT)
        assert SYSTEM_PROMPT is snapshot
        assert "highest-conviction setup" not in SYSTEM_PROMPT

    def test_variant_longer_than_base(self):
        variant = apply_midband_deployment_delta(SYSTEM_PROMPT)
        assert len(variant) > len(SYSTEM_PROMPT)

    def test_missing_anchor_a_fails_loud(self):
        """A drifted base prompt must raise, never run a half-applied variant."""
        broken = SYSTEM_PROMPT.replace("waiting for a perfect signal.", "drifted anchor.")
        with pytest.raises(ValueError, match="Edit A"):
            apply_midband_deployment_delta(broken)

    def test_missing_anchor_b_fails_loud(self):
        broken = SYSTEM_PROMPT.replace(
            "When in doubt, default to HOLD.",
            "When in doubt, do something else.",
        )
        with pytest.raises(ValueError, match="Edit B"):
            apply_midband_deployment_delta(broken)

    def test_edit_a_scoped_to_mid_band(self):
        """Edit A must not touch the Rule-5 above-upper-bound mandate."""
        variant = apply_midband_deployment_delta(SYSTEM_PROMPT)
        # Rule-5 language for above-band cash survives verbatim (single-line
        # fragment per LEARNINGS 2026-09-28).
        assert "Default to deploying available cash into the best" in variant

    def test_variant_marker_constants(self):
        """Marker tuples are exported for downstream audit/logging."""
        assert isinstance(VARIANT_EDIT_A_MARKERS, tuple)
        assert isinstance(VARIANT_EDIT_B_MARKERS, tuple)
        assert MIDBAND_DEPLOYMENT_VARIANT == "midband_deployment"
