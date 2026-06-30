"""
Comprehensive tests for enhanced_prompt.py.

Covers:
- get_enhanced_system_prompt returns correct content
- Prompt contains all required sections and principles
- Prompt contains specific thresholds and values
- Prompt contains valid JSON output format example
- IMPROVEMENTS_SUMMARY contains expected categories
- Prompt length is reasonable (not empty, not truncated)
"""

import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.enhanced_prompt import (
    get_enhanced_system_prompt,
    ENHANCED_SYSTEM_PROMPT,
    IMPROVEMENTS_SUMMARY,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_json_example(prompt: str) -> str:
    """Extract the JSON example block from the prompt."""
    start = prompt.find('{\n  "actions":')
    if start == -1:
        # Try single-line variant
        start = prompt.find('{"actions":')
    if start == -1:
        return ""
    # Find the matching closing brace
    brace_count = 0
    end = start
    for i, ch in enumerate(prompt[start:]):
        if ch == '{':
            brace_count += 1
        elif ch == '}':
            brace_count -= 1
            if brace_count == 0:
                end = start + i + 1
                break
    return prompt[start:end]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetEnhancedSystemPrompt:
    def test_returns_string(self):
        result = get_enhanced_system_prompt()
        assert isinstance(result, str)

    def test_returns_enhanced_constant(self):
        result = get_enhanced_system_prompt()
        assert result is ENHANCED_SYSTEM_PROMPT
        assert result == ENHANCED_SYSTEM_PROMPT

    def test_not_empty(self):
        result = get_enhanced_system_prompt()
        assert len(result) > 1000

    def test_contains_prospect_theory(self):
        prompt = get_enhanced_system_prompt()
        assert "Prospect Theory" in prompt

    def test_contains_loss_aversion(self):
        prompt = get_enhanced_system_prompt()
        assert "LOSS AVERSION" in prompt
        assert "2.25x" in prompt

    def test_contains_cvar_reference(self):
        prompt = get_enhanced_system_prompt()
        assert "CVaR" in prompt
        assert "Conditional Value at Risk" in prompt

    def test_contains_meta_labeling(self):
        prompt = get_enhanced_system_prompt()
        assert "META-LABELING" in prompt
        assert "PRIMARY MODEL" in prompt
        assert "SECONDARY MODEL" in prompt

    def test_contains_specific_rsi_thresholds(self):
        prompt = get_enhanced_system_prompt()
        # Original was 30/70, enhanced uses 24/76
        assert "RSI < 24" in prompt
        assert "RSI > 76" in prompt

    def test_contains_vixy_thresholds(self):
        prompt = get_enhanced_system_prompt()
        assert "VIXY < 15" in prompt
        assert "VIXY 15-25" in prompt
        assert "VIXY > 25" in prompt
        assert "VIXY > 35" in prompt

    def test_contains_position_sizing_rules(self):
        prompt = get_enhanced_system_prompt()
        assert "Maximum 25%" in prompt
        assert "10-30% cash buffer" in prompt

    def test_contains_stop_loss_thresholds(self):
        prompt = get_enhanced_system_prompt()
        assert "drawdown > 5%" in prompt
        assert "drawdown > 8%" in prompt
        assert "drawdown > 3%" in prompt
        assert "drawdown > 5% total" in prompt

    def test_contains_deflated_sharpe(self):
        prompt = get_enhanced_system_prompt()
        assert "Deflated Sharpe Ratio" in prompt
        assert "multiple testing" in prompt.lower()

    def test_contains_json_output_format(self):
        prompt = get_enhanced_system_prompt()
        assert '"actions"' in prompt
        assert '"ticker"' in prompt
        assert '"action"' in prompt
        assert '"pct"' in prompt
        assert '"reasoning"' in prompt

    def test_json_example_is_valid(self):
        prompt = get_enhanced_system_prompt()
        json_str = _extract_json_example(prompt)
        assert json_str, "No JSON example found in prompt"
        example = json.loads(json_str)
        assert "actions" in example
        assert "reasoning" in example
        assert isinstance(example["actions"], list)
        for action in example["actions"]:
            assert "ticker" in action
            assert "action" in action

    def test_contains_emergency_protocol(self):
        prompt = get_enhanced_system_prompt()
        assert "Emergency mode" in prompt
        assert "70%+" in prompt or "70%" in prompt

    def test_contains_macro_checklist(self):
        prompt = get_enhanced_system_prompt()
        assert "MACRO CONTEXT CHECKLIST" in prompt
        assert "[ ] VIXY" in prompt
        assert "[ ] SPY" in prompt

    def test_contains_trend_regime_principles(self):
        prompt = get_enhanced_system_prompt()
        assert "STRONG DOWNTREND" in prompt
        assert "STRONG UPTREND" in prompt
        assert "CHOPPY" in prompt or "RANGING" in prompt

    def test_contains_intraday_alerts(self):
        prompt = get_enhanced_system_prompt()
        assert "INTRADAY ALERT RESPONSE" in prompt
        assert "Flash crash" in prompt

    def test_mentions_initial_capital(self):
        prompt = get_enhanced_system_prompt()
        assert "10,000 EUR" in prompt

    def test_mentions_behavioral_finance(self):
        prompt = get_enhanced_system_prompt()
        assert "Behavioral Finance" in prompt

    def test_mentions_lopez_de_prado(self):
        prompt = get_enhanced_system_prompt()
        assert "Lopez de Prado" in prompt

    def test_contains_diversification_section(self):
        prompt = get_enhanced_system_prompt()
        assert "DIVERSIFICATION" in prompt
        assert "correlations" in prompt.lower()

    def test_contains_action_definitions(self):
        prompt = get_enhanced_system_prompt()
        assert '"buy" with "pct"' in prompt
        assert '"sell" with "pct"' in prompt
        assert '"hold"' in prompt

    def test_ends_with_risk_management_reminder(self):
        prompt = get_enhanced_system_prompt()
        assert "RISK MANAGEMENT" in prompt
        assert "Preserve capital first" in prompt


class TestImprovementsSummary:
    def test_is_string(self):
        assert isinstance(IMPROVEMENTS_SUMMARY, str)

    def test_not_empty(self):
        assert len(IMPROVEMENTS_SUMMARY) > 200

    def test_contains_specific_thresholds_section(self):
        assert "SPECIFIC THRESHOLDS" in IMPROVEMENTS_SUMMARY

    def test_contains_meta_labeling_section(self):
        assert "META-LABELING FRAMEWORK" in IMPROVEMENTS_SUMMARY

    def test_contains_emergency_protocols_section(self):
        assert "EMERGENCY PROTOCOLS" in IMPROVEMENTS_SUMMARY

    def test_contains_intraday_guidance_section(self):
        assert "INTRADAY GUIDANCE" in IMPROVEMENTS_SUMMARY

    def test_contains_macro_checklist_section(self):
        assert "MACRO CHECKLIST" in IMPROVEMENTS_SUMMARY

    def test_references_trading_observations(self):
        assert "live trading" in IMPROVEMENTS_SUMMARY.lower() or "trading observations" in IMPROVEMENTS_SUMMARY.lower()

    def test_mentions_mean_reversion_traps(self):
        assert "mean reversion" in IMPROVEMENTS_SUMMARY.lower()

    def test_mentions_cash_buffers(self):
        assert "cash" in IMPROVEMENTS_SUMMARY.lower()


class TestPromptConsistency:
    def test_prompt_and_summary_are_separate(self):
        assert ENHANCED_SYSTEM_PROMPT is not IMPROVEMENTS_SUMMARY
        assert ENHANCED_SYSTEM_PROMPT != IMPROVEMENTS_SUMMARY

    def test_prompt_does_not_contain_summary_text(self):
        assert "Key improvements in enhanced prompt" not in ENHANCED_SYSTEM_PROMPT

    def test_summary_references_prompt_content(self):
        assert "RSI < 24" in IMPROVEMENTS_SUMMARY
        assert "VIXY" in IMPROVEMENTS_SUMMARY

    def test_thresholds_are_consistent_between_prompt_and_summary(self):
        assert "RSI < 24" in ENHANCED_SYSTEM_PROMPT
        assert "RSI < 24" in IMPROVEMENTS_SUMMARY
        assert "VIXY > 35" in ENHANCED_SYSTEM_PROMPT
        assert "VIXY > 35" in IMPROVEMENTS_SUMMARY
