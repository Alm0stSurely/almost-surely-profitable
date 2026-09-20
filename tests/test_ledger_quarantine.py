"""
Regression tests for corrupt-ledger quarantine in append-then-overwrite flows.

Sites: ``portfolio.portfolio.Portfolio.save_trade`` and
``llm.trading_agent.TradingAgent.save_decision``.

Pre-fix behaviour: a corrupt backing file was silently treated as an empty
list and then *overwritten* by the append — the ledger/history was truncated
to a single record with no trace of the destroyed records. Post-fix: the
corrupt file is quarantined (renamed ``<name>.corrupt-<timestamp>``), an error
is logged, and the new record starts a fresh file.
"""

import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from llm.trading_agent import TradingAgent
from portfolio.portfolio import Portfolio, Trade


def _make_trade(ticker: str = "SPY") -> Trade:
    return Trade(
        timestamp=datetime.now().isoformat(),
        ticker=ticker,
        action="buy",
        quantity=1.0,
        price=100.0,
        total_value=100.0,
        fees=0.0,
    )


def _make_decision(ticker: str = "SPY") -> dict:
    return {
        "timestamp": datetime.now().isoformat(),
        "actions": [{"ticker": ticker, "action": "hold"}],
        "reasoning": "test decision",
    }


def _quarantine_siblings(path: Path):
    return sorted(path.parent.glob(f"{path.name}.corrupt-*"))


class TestSaveTradeQuarantine:
    """Corrupt trade ledger must be quarantined, not truncated."""

    def test_corrupt_ledger_quarantined_and_new_trade_saved(self, tmp_path, caplog):
        pf = Portfolio(data_dir=str(tmp_path))
        pf.trades_file.write_text("{not valid json")

        with caplog.at_level(logging.ERROR):
            pf.save_trade(_make_trade())

        # New trade persisted as a fresh single-record ledger.
        saved = json.loads(pf.trades_file.read_text())
        assert isinstance(saved, list) and len(saved) == 1
        assert saved[0]["ticker"] == "SPY"

        # Old contents preserved under a .corrupt-* quarantine file.
        quarantines = _quarantine_siblings(pf.trades_file)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "{not valid json"

        # Failure was loud.
        assert any("quarantined" in r.message and "trade ledger" in r.message
                   for r in caplog.records)

    def test_wrong_shape_ledger_quarantined(self, tmp_path, caplog):
        """Valid JSON that is not a list must not crash .append or be kept."""
        pf = Portfolio(data_dir=str(tmp_path))
        pf.trades_file.write_text(json.dumps({"unexpected": "object"}))

        with caplog.at_level(logging.ERROR):
            pf.save_trade(_make_trade())

        saved = json.loads(pf.trades_file.read_text())
        assert isinstance(saved, list) and len(saved) == 1
        assert len(_quarantine_siblings(pf.trades_file)) == 1
        assert any("wrong shape" in r.message for r in caplog.records)

    def test_healthy_ledger_appends_without_quarantine(self, tmp_path, caplog):
        pf = Portfolio(data_dir=str(tmp_path))
        pf.save_trade(_make_trade("SPY"))
        pf.save_trade(_make_trade("GLD"))

        with caplog.at_level(logging.ERROR):
            pf.save_trade(_make_trade("QQQ"))

        saved = json.loads(pf.trades_file.read_text())
        assert [t["ticker"] for t in saved] == ["SPY", "GLD", "QQQ"]
        assert _quarantine_siblings(pf.trades_file) == []
        assert caplog.records == []

    def test_second_save_after_quarantine_is_normal(self, tmp_path):
        """After a quarantine, the fresh ledger grows normally (no repeat)."""
        pf = Portfolio(data_dir=str(tmp_path))
        pf.trades_file.write_text("garbage")

        pf.save_trade(_make_trade("SPY"))
        pf.save_trade(_make_trade("GLD"))

        saved = json.loads(pf.trades_file.read_text())
        assert [t["ticker"] for t in saved] == ["SPY", "GLD"]
        # Exactly one quarantine from the first save only.
        assert len(_quarantine_siblings(pf.trades_file)) == 1

    def test_missing_ledger_starts_fresh_without_quarantine(self, tmp_path, caplog):
        pf = Portfolio(data_dir=str(tmp_path))

        with caplog.at_level(logging.ERROR):
            pf.save_trade(_make_trade())

        saved = json.loads(pf.trades_file.read_text())
        assert len(saved) == 1
        assert _quarantine_siblings(pf.trades_file) == []
        assert caplog.records == []


class TestSaveDecisionQuarantine:
    """Corrupt decision history must be quarantined, not truncated."""

    def _agent(self, tmp_path) -> TradingAgent:
        return TradingAgent(history_file=str(tmp_path / "decision_history.json"))

    def test_corrupt_history_quarantined_and_new_decision_saved(self, tmp_path, caplog):
        agent = self._agent(tmp_path)
        agent.history_file.write_text("[{broken")

        with caplog.at_level(logging.ERROR):
            agent.save_decision(_make_decision())

        saved = json.loads(agent.history_file.read_text())
        assert isinstance(saved, list) and len(saved) == 1
        assert saved[0]["reasoning"] == "test decision"

        quarantines = _quarantine_siblings(agent.history_file)
        assert len(quarantines) == 1
        assert quarantines[0].read_text() == "[{broken"

        assert any("quarantined" in r.message and "decision history" in r.message
                   for r in caplog.records)

    def test_wrong_shape_history_quarantined(self, tmp_path, caplog):
        agent = self._agent(tmp_path)
        agent.history_file.write_text(json.dumps("a string, not a list"))

        with caplog.at_level(logging.ERROR):
            agent.save_decision(_make_decision())

        saved = json.loads(agent.history_file.read_text())
        assert isinstance(saved, list) and len(saved) == 1
        assert len(_quarantine_siblings(agent.history_file)) == 1

    def test_healthy_history_appends_without_quarantine(self, tmp_path, caplog):
        agent = self._agent(tmp_path)
        agent.save_decision(_make_decision("SPY"))

        with caplog.at_level(logging.ERROR):
            agent.save_decision(_make_decision("GLD"))

        saved = json.loads(agent.history_file.read_text())
        assert [d["actions"][0]["ticker"] for d in saved] == ["SPY", "GLD"]
        assert _quarantine_siblings(agent.history_file) == []
        assert caplog.records == []
