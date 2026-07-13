"""Tests for behavioral_analysis.py keyword concept matching."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.behavioral_analysis import KEYWORD_CONCEPTS, count_keyword_concepts


def make_decision(reasoning="", error=False):
    return {"reasoning": reasoning, "error": error}


class TestCountKeywordConcepts:
    def test_empty_decisions_returns_zeros(self):
        counts = count_keyword_concepts([], {})
        assert counts == {}

    def test_single_variant_match(self):
        decisions = [
            make_decision("I see loss aversion in the market."),
        ]
        counts = count_keyword_concepts(decisions, {"loss aversion": ["loss aversion"]})
        assert counts["loss aversion"] == 1

    def test_multi_variant_match(self):
        decisions = [
            make_decision("The weekly trade cap is reached."),
            make_decision("We hit the trade limit today."),
            make_decision("Cash buffer is high."),
        ]
        concepts = {
            "trade cap": ["trade cap", "trade limit"],
            "cash buffer": ["cash buffer"],
        }
        counts = count_keyword_concepts(decisions, concepts)
        assert counts["trade cap"] == 2
        assert counts["cash buffer"] == 1

    def test_case_insensitive(self):
        decisions = [make_decision("Loss Aversion is a key factor.")]
        counts = count_keyword_concepts(decisions, {"loss aversion": ["loss aversion"]})
        assert counts["loss aversion"] == 1

    def test_decision_counted_once_per_concept(self):
        decisions = [
            make_decision("Loss aversion and loss aversion again."),
        ]
        counts = count_keyword_concepts(decisions, {"loss aversion": ["loss aversion"]})
        assert counts["loss aversion"] == 1

    def test_error_decisions_can_be_included(self):
        decisions = [
            make_decision("loss aversion", error=False),
            make_decision("cash buffer", error=True),
        ]
        counts = count_keyword_concepts(decisions, {"loss aversion": ["loss aversion"], "cash buffer": ["cash buffer"]})
        assert counts["loss aversion"] == 1
        assert counts["cash buffer"] == 1

    def test_default_keyword_concepts(self):
        assert "cooldown" in KEYWORD_CONCEPTS
        assert "trade cap" in KEYWORD_CONCEPTS
        assert "let winners run" in KEYWORD_CONCEPTS

    def test_variant_captures_paraphrase(self):
        """"trade cap" concept should match common paraphrases."""
        decisions = [
            make_decision("weekly trade cap reached"),
            make_decision("weekly trade limit reached"),
            make_decision("trades used: 2"),
            make_decision("trades remaining: 1"),
        ]
        counts = count_keyword_concepts(decisions, KEYWORD_CONCEPTS)
        # Each of the four paraphrases should match the "trade cap" concept once.
        assert counts["trade cap"] == 4

    def test_cooldown_concept_variants(self):
        decisions = [
            make_decision("holding period is active"),
            make_decision("flip cooldown prevents re-entry"),
            make_decision("cooldown guardrails are in place"),
        ]
        counts = count_keyword_concepts(decisions, KEYWORD_CONCEPTS)
        assert counts["cooldown"] == 3

    def test_let_winners_run_variants(self):
        decisions = [
            make_decision("let winners run discipline"),
            make_decision("ride winners until reversal"),
        ]
        counts = count_keyword_concepts(decisions, KEYWORD_CONCEPTS)
        assert counts["let winners run"] == 2

    def test_no_match(self):
        decisions = [make_decision("This decision has no relevant keywords.")]
        counts = count_keyword_concepts(decisions, KEYWORD_CONCEPTS)
        assert all(v == 0 for v in counts.values())
