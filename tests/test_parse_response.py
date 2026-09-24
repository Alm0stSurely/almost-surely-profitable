"""Regression tests for TradingAgent.parse_response failure direction.

One malformed action must not void the valid actions in the same response
(all-or-nothing hold-all would also skip valid exits — the unbounded-loss
direction). Dropping the malformed action individually costs at most one
session of delay on that ticker: bounded and reversible.

Extraction: nested objects before the "actions" key must not hide the
envelope from the extractor (old flat-prefix regex fell through to
full-text parsing and failed).
"""

import json
import logging

import pytest

from src.llm.trading_agent import TradingAgent


@pytest.fixture
def agent(tmp_path):
    return TradingAgent(
        api_key="test-key",
        history_file=str(tmp_path / "decisions.json"),
    )


class TestParseResponseHealthy:
    """Healthy pins: behavior unchanged for well-formed responses."""

    def test_pure_json(self, agent):
        response = json.dumps({
            "actions": [{"ticker": "SPY", "action": "buy", "pct": 10}],
            "reasoning": "ok",
        })
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "SPY", "action": "buy", "pct": 10}]
        assert result["reasoning"] == "ok"
        assert "error" not in result

    def test_markdown_wrapped(self, agent):
        payload = json.dumps({"actions": [{"ticker": "GLD", "action": "hold"}], "reasoning": "steady"})
        result = agent.parse_response(f'Here you go:\n```json\n{payload}\n```\nDone.')
        assert result["actions"] == [{"ticker": "GLD", "action": "hold"}]
        assert result["reasoning"] == "steady"

    def test_embedded_in_prose(self, agent):
        payload = json.dumps({"actions": [{"ticker": "SPY", "action": "sell", "pct": 25}], "reasoning": "trim"})
        result = agent.parse_response(f'Analysis complete. Decision: {payload} End of message.')
        assert result["actions"] == [{"ticker": "SPY", "action": "sell", "pct": 25}]

    def test_extra_keys_preserved(self, agent):
        response = json.dumps({
            "actions": [{"ticker": "SPY", "action": "hold"}],
            "reasoning": "ok",
            "confidence": 0.8,
        })
        result = agent.parse_response(response)
        assert result["confidence"] == 0.8
        assert result["actions"] == [{"ticker": "SPY", "action": "hold"}]


class TestParseResponseCaseNormalization:
    """Case/whitespace variants are the dominant false-failure class."""

    @pytest.mark.parametrize("variant", ["BUY", "Buy", " buy ", "bUy"])
    def test_action_case_normalized(self, agent, variant):
        response = json.dumps({"actions": [{"ticker": "SPY", "action": variant, "pct": 5}]})
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "SPY", "action": "buy", "pct": 5}]
        assert "error" not in result

    def test_mixed_case_batch(self, agent):
        response = json.dumps({
            "actions": [
                {"ticker": "SPY", "action": "HOLD"},
                {"ticker": "GLD", "action": "SELL", "pct": 50},
            ],
            "reasoning": "rotate",
        })
        result = agent.parse_response(response)
        assert [a["action"] for a in result["actions"]] == ["hold", "sell"]
        assert "error" not in result


class TestParseResponsePerActionFailureDirection:
    """The core doctrine: partial malformation -> partial drop, never global hold-all."""

    def test_one_malformed_action_keeps_valid_ones(self, agent):
        # A valid stop-loss exit alongside one malformed action must survive.
        response = json.dumps({
            "actions": [
                {"ticker": "SPY", "action": "sell", "pct": 100},
                {"ticker": "GLD", "action": "byu", "pct": 10},
            ],
            "reasoning": "exit SPY, typo on GLD",
        })
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "SPY", "action": "sell", "pct": 100}]
        assert "error" not in result
        assert "dropped 1 malformed" in result["reasoning"]

    def test_drop_logged_loudly(self, agent, caplog):
        response = json.dumps({
            "actions": [
                {"ticker": "SPY", "action": "hold"},
                {"action": "buy", "pct": 10},
            ],
        })
        with caplog.at_level(logging.ERROR):
            result = agent.parse_response(response)
        assert len(result["actions"]) == 1
        summary = [r for r in caplog.records if "malformed action(s)" in r.getMessage()]
        per_action = [r for r in caplog.records if "dropped:" in r.getMessage()]
        assert len(summary) == 1
        assert len(per_action) == 1

    def test_unknown_action_dropped(self, agent):
        response = json.dumps({
            "actions": [{"ticker": "SPY", "action": "short", "pct": 10}],
        })
        result = agent.parse_response(response)
        assert result["error"] is True
        assert result["actions"] == []

    def test_non_dict_action_dropped(self, agent):
        response = json.dumps({
            "actions": ["SPY buy 10", {"ticker": "GLD", "action": "hold"}],
        })
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "GLD", "action": "hold"}]
        assert "dropped 1 malformed" in result["reasoning"]

    def test_blank_ticker_dropped(self, agent):
        response = json.dumps({
            "actions": [{"ticker": "  ", "action": "buy"}, {"ticker": "SPY", "action": "hold"}],
        })
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "SPY", "action": "hold"}]

    def test_non_finite_pct_dropped(self, agent):
        response = json.dumps({
            "actions": [
                {"ticker": "SPY", "action": "buy", "pct": "ten"},
                {"ticker": "GLD", "action": "hold"},
            ],
        })
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "GLD", "action": "hold"}]
        assert "dropped 1 malformed" in result["reasoning"]

    def test_all_malformed_falls_back_to_hold_all(self, agent):
        # Zero survivors keep the established envelope-level fallback.
        response = json.dumps({
            "actions": [{"ticker": "SPY", "action": "byu"}, {"action": "sell"}],
        })
        result = agent.parse_response(response)
        assert result["error"] is True
        assert result["actions"] == []
        assert "hold all" in result["reasoning"]

    def test_actions_not_a_list_falls_back(self, agent):
        response = json.dumps({"actions": {"ticker": "SPY", "action": "buy"}})
        result = agent.parse_response(response)
        assert result["error"] is True
        assert result["actions"] == []

    def test_missing_actions_key_falls_back(self, agent):
        result = agent.parse_response(json.dumps({"reasoning": "no actions today"}))
        assert result["error"] is True
        assert result["actions"] == []


class TestParseResponseExtraction:
    """Nested structures before the actions key must not hide the envelope."""

    def test_nested_context_before_actions(self, agent):
        response = (
            '{"reasoning": "rotate defensively", '
            '"context": {"regime": {"trend": "down", "depth": {"level": 2}}}, '
            '"actions": [{"ticker": "SPY", "action": "sell", "pct": 50}]}'
        )
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "SPY", "action": "sell", "pct": 50}]
        assert "error" not in result

    def test_nested_object_in_prose(self, agent):
        payload = (
            '{"meta": {"nested": {"a": {"b": 1}}}, '
            '"actions": [{"ticker": "GLD", "action": "buy", "pct": 5}], '
            '"reasoning": "hedge"}'
        )
        result = agent.parse_response(f'Result: {payload} — end.')
        assert result["actions"] == [{"ticker": "GLD", "action": "buy", "pct": 5}]
        assert "error" not in result

    def test_no_actions_key_in_text(self, agent):
        result = agent.parse_response('I cannot decide today, no trades.')
        assert result["error"] is True
        assert result["actions"] == []


class TestParseResponsePctSemantics:
    """Missing pct keeps executor default (0); present pct must be finite."""

    def test_missing_pct_accepted(self, agent):
        response = json.dumps({"actions": [{"ticker": "SPY", "action": "hold"}]})
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "SPY", "action": "hold"}]
        assert "pct" not in result["actions"][0]

    def test_bool_pct_dropped(self, agent):
        response = json.dumps({
            "actions": [{"ticker": "SPY", "action": "buy", "pct": True}],
        })
        result = agent.parse_response(response)
        assert result["error"] is True  # single action, dropped

    def test_overflow_pct_dropped(self, agent):
        # 1e999 parses to inf via Python json — must not reach the executor.
        response = '{"actions": [{"ticker": "SPY", "action": "buy", "pct": 1e999}, {"ticker": "GLD", "action": "hold"}]}'
        result = agent.parse_response(response)
        assert result["actions"] == [{"ticker": "GLD", "action": "hold"}]
        assert "dropped 1 malformed" in result["reasoning"]
