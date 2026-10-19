"""
Tests for the shadow-decision arm of the H4 experiment.

Shadow mode runs the variant prompt alongside the live prompt each evening,
logging variant actions + reasoning to data/shadow_decisions.json. Hard
invariants:
  - the shadow arm never executes trades,
  - never consumes cooldown budget,
  - never writes to decision_history.json (live LLM context must not be
    contaminated with hypothetical decisions),
  - a shadow failure is logged and never aborts the daily pipeline.
"""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.shadow_decisions import (
    ShadowArmResult,
    append_shadow_record,
    build_shadow_record,
    load_shadow_config,
    run_shadow_arm,
    summarize_actions,
)
from llm.trading_agent import TradingAgent
from llm.prompt_variants import apply_midband_deployment_delta, MIDBAND_DEPLOYMENT_VARIANT


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

class TestLoadShadowConfig:
    def test_missing_config_disables_shadow(self, tmp_path):
        cfg = load_shadow_config(tmp_path / "nope.json")
        assert cfg.enabled is False
        assert cfg.mode == "off"

    def test_shadow_mode_enables(self, tmp_path):
        p = tmp_path / "experiments.json"
        p.write_text(json.dumps({"midband_deployment": {"mode": "shadow"}}))
        cfg = load_shadow_config(p)
        assert cfg.enabled is True
        assert cfg.mode == "shadow"
        assert cfg.variant == MIDBAND_DEPLOYMENT_VARIANT

    def test_off_mode_disables(self, tmp_path):
        p = tmp_path / "experiments.json"
        p.write_text(json.dumps({"midband_deployment": {"mode": "off"}}))
        cfg = load_shadow_config(p)
        assert cfg.enabled is False

    def test_live_mode_disables_shadow_arm(self, tmp_path):
        """Phase 3 flag value: the shadow arm must not run when live."""
        p = tmp_path / "experiments.json"
        p.write_text(json.dumps({"midband_deployment": {"mode": "live"}}))
        cfg = load_shadow_config(p)
        assert cfg.enabled is False
        assert cfg.mode == "live"

    def test_corrupt_config_fails_loud_and_disables(self, tmp_path):
        p = tmp_path / "experiments.json"
        p.write_text("{not json")
        cfg = load_shadow_config(p)
        assert cfg.enabled is False

    def test_wrong_shape_config_disables(self, tmp_path):
        p = tmp_path / "experiments.json"
        p.write_text(json.dumps(["a", "list"]))
        cfg = load_shadow_config(p)
        assert cfg.enabled is False


# ---------------------------------------------------------------------------
# Action summarizing + record building
# ---------------------------------------------------------------------------

class TestBuildShadowRecord:
    def test_summarize_actions(self):
        actions = [
            {"ticker": "SPY", "action": "hold"},
            {"ticker": "TLT", "action": "buy", "pct": 15},
        ]
        s = summarize_actions(actions)
        assert "TLT buy" in s
        assert "SPY hold" in s

    def test_summarize_empty(self):
        assert summarize_actions([]) == "no actions"

    def test_record_contains_required_fields(self):
        record = build_shadow_record(
            variant="midband_deployment",
            decision={"actions": [], "reasoning": "r", "error": False},
            portfolio_summary={"cash": 3000.0, "total_value": 10000.0},
            vol_regime="normal",
            live_actions_summary="all hold",
        )
        assert record["variant"] == "midband_deployment"
        assert record["mode"] == "shadow"
        assert record["cash_pct"] == pytest.approx(30.0)
        assert record["vol_regime"] == "normal"
        assert record["protocol_violation"] is False
        assert record["error"] is False
        assert "date" in record and "timestamp" in record

    def test_idle_cash_buy_flagged_as_protocol_violation(self):
        """Metric 3.4.5: a buy citing only idle cash violates the protocol."""
        decision = {
            "actions": [{"ticker": "SPY", "action": "buy", "pct": 10}],
            "reasoning": "Cash is idle and should be deployed.",
            "error": False,
        }
        record = build_shadow_record(
            variant="midband_deployment",
            decision=decision,
            portfolio_summary={"cash": 2000.0, "total_value": 10000.0},
            vol_regime="normal",
            live_actions_summary="all hold",
        )
        assert record["protocol_violation"] is True

    def test_evidence_cited_buy_not_flagged(self):
        decision = {
            "actions": [{"ticker": "SPY", "action": "buy", "pct": 10}],
            "reasoning": "RSI pulled back to 42 from 68 while price held above the rising SMA-20, partial alignment in an uptrend.",
            "error": False,
        }
        record = build_shadow_record(
            variant="midband_deployment",
            decision=decision,
            portfolio_summary={"cash": 2000.0, "total_value": 10000.0},
            vol_regime="normal",
            live_actions_summary="all hold",
        )
        assert record["protocol_violation"] is False

    def test_hold_never_flagged(self):
        decision = {
            "actions": [{"ticker": "SPY", "action": "hold"}],
            "reasoning": "No setup, cash idle.",
            "error": False,
        }
        record = build_shadow_record(
            variant="midband_deployment",
            decision=decision,
            portfolio_summary={"cash": 2000.0, "total_value": 10000.0},
            vol_regime="normal",
            live_actions_summary="all hold",
        )
        assert record["protocol_violation"] is False


# ---------------------------------------------------------------------------
# Append-only shadow log
# ---------------------------------------------------------------------------

class TestAppendShadowRecord:
    def test_creates_file_on_first_record(self, tmp_path):
        log = tmp_path / "shadow_decisions.json"
        append_shadow_record(log, {"date": "2026-10-19", "variant": "midband_deployment"})
        records = json.loads(log.read_text())
        assert len(records) == 1
        assert records[0]["date"] == "2026-10-19"

    def test_appends_preserving_order(self, tmp_path):
        log = tmp_path / "shadow_decisions.json"
        append_shadow_record(log, {"n": 1})
        append_shadow_record(log, {"n": 2})
        records = json.loads(log.read_text())
        assert [r["n"] for r in records] == [1, 2]

    def test_corrupt_log_quarantined_and_starts_fresh(self, tmp_path):
        log = tmp_path / "shadow_decisions.json"
        log.write_text("{garbage")
        append_shadow_record(log, {"n": 1})
        records = json.loads(log.read_text())
        assert [r["n"] for r in records] == [1]
        # The corrupt file is preserved for recovery, not silently overwritten.
        quarantined = list(tmp_path.glob("shadow_decisions.json.corrupt-*"))
        assert len(quarantined) == 1


# ---------------------------------------------------------------------------
# TradingAgent shadow decision: no history contamination
# ---------------------------------------------------------------------------

class TestGetShadowDecision:
    def _agent(self, tmp_path):
        return TradingAgent(
            api_key="dummy",
            history_file=str(tmp_path / "decision_history.json"),
        )

    def test_shadow_decision_not_saved_to_history(self, tmp_path):
        agent = self._agent(tmp_path)
        with patch.object(TradingAgent, "call_llm", return_value=json.dumps({
            "actions": [{"ticker": "SPY", "action": "buy", "pct": 10}],
            "reasoning": "test",
        })):
            decision = agent.get_shadow_decision(
                market_data={"assets": {}},
                portfolio_summary={"cash": 1000.0, "total_value": 10000.0, "positions": []},
            )
        assert decision["actions"][0]["ticker"] == "SPY"
        assert not (tmp_path / "decision_history.json").exists()

    def test_shadow_uses_variant_system_prompt(self, tmp_path):
        agent = self._agent(tmp_path)
        with patch.object(TradingAgent, "call_llm", return_value=json.dumps({
            "actions": [], "reasoning": "x",
        })) as mock_call:
            agent.get_shadow_decision(
                market_data={"assets": {}},
                portfolio_summary={"cash": 1000.0, "total_value": 10000.0, "positions": []},
            )
        system_prompt_used = mock_call.call_args.kwargs["system_prompt"]
        assert "highest-conviction setup" in system_prompt_used

    def test_shadow_default_prompt_is_variant(self):
        """get_shadow_decision defaults to the mid-band variant prompt."""
        from llm.trading_agent import SYSTEM_PROMPT
        from llm.prompt_variants import apply_midband_deployment_delta
        expected = apply_midband_deployment_delta(SYSTEM_PROMPT)
        agent = TradingAgent(api_key="dummy")
        with patch.object(TradingAgent, "call_llm", return_value=json.dumps({
            "actions": [], "reasoning": "x",
        })) as mock_call:
            agent.get_shadow_decision(
                market_data={"assets": {}},
                portfolio_summary={"cash": 1000.0, "total_value": 10000.0, "positions": []},
            )
        assert mock_call.call_args.kwargs["system_prompt"] == expected

    def test_shadow_api_error_returns_error_decision(self, tmp_path):
        agent = self._agent(tmp_path)
        with patch.object(TradingAgent, "call_llm", return_value=None):
            decision = agent.get_shadow_decision(
                market_data={"assets": {}},
                portfolio_summary={"cash": 1000.0, "total_value": 10000.0, "positions": []},
            )
        assert decision["error"] is True
        assert decision["actions"] == []
        # Still no history write on failure.
        assert not (tmp_path / "decision_history.json").exists()

    def test_call_llm_backward_compatible_default(self):
        """call_llm without system_prompt must keep sending the base prompt."""
        agent = TradingAgent(api_key="dummy")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"actions": [], "reasoning": "ok"})}}]
        }
        with patch("llm.trading_agent.requests.post", return_value=mock_resp) as post:
            out = agent.call_llm("hello")
        assert out is not None
        payload = post.call_args.kwargs["json"]
        from llm.trading_agent import SYSTEM_PROMPT
        assert payload["messages"][0]["content"] == SYSTEM_PROMPT

    def test_call_llm_accepts_variant_system_prompt(self):
        agent = TradingAgent(api_key="dummy")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "choices": [{"message": {"content": json.dumps({"actions": [], "reasoning": "ok"})}}]
        }
        with patch("llm.trading_agent.requests.post", return_value=mock_resp) as post:
            agent.call_llm("hello", system_prompt="VARIANT-PROMPT")
        payload = post.call_args.kwargs["json"]
        assert payload["messages"][0]["content"] == "VARIANT-PROMPT"
        assert payload["messages"][1]["content"] == "hello"


# ---------------------------------------------------------------------------
# run_shadow_arm orchestration
# ---------------------------------------------------------------------------

class TestRunShadowArm:
    def _context(self, tmp_path):
        return {
            "agent": TradingAgent(api_key="dummy", history_file=str(tmp_path / "h.json")),
            "market_analysis": {"assets": {}},
            "portfolio_summary": {"cash": 2000.0, "total_value": 10000.0, "positions": []},
            "cooldown_status": {"trades_this_week": 0, "weekly_cap": 3},
            "vol_regime": "normal",
            "log_path": tmp_path / "shadow_decisions.json",
            "live_actions_summary": "all hold",
        }

    def test_happy_path_logs_record(self, tmp_path):
        ctx = self._context(tmp_path)
        with patch.object(TradingAgent, "call_llm", return_value=json.dumps({
            "actions": [{"ticker": "SPY", "action": "hold"}],
            "reasoning": "no edge",
        })):
            result = run_shadow_arm(**ctx)
        assert isinstance(result, ShadowArmResult)
        assert result.logged is True
        records = json.loads(ctx["log_path"].read_text())
        assert len(records) == 1
        assert records[0]["variant"] == "midband_deployment"

    def test_failure_still_logs_error_record(self, tmp_path):
        """A shadow failure must be visible in the log, never crash the caller."""
        ctx = self._context(tmp_path)
        with patch.object(TradingAgent, "call_llm", return_value=None):
            result = run_shadow_arm(**ctx)
        assert result.logged is True
        records = json.loads(ctx["log_path"].read_text())
        assert records[0]["error"] is True
