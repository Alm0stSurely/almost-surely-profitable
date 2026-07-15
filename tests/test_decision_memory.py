"""
Comprehensive tests for decision_memory.py.

Covers:
- DecisionRecord serialization (to_dict / from_dict)
- DecisionMemory persistence (load / save)
- add_decision with auto-save every 10 entries
- get_decision_summary (filtering, counts, win rate, P&L, hold times)
- get_pattern_analysis (correlations, holding buckets, behavioral flags)
- generate_lessons_learned (all branches: win rate, overtrading, P&L, patterns)
- get_memory_context_for_llm (formatting)
- export_to_dataframe (empty / non-empty)
- get_similar_decisions (similarity scoring, limits, missing data)
- Edge cases: zero P&L, None fields, invalid JSON, missing files
"""

import json
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from analysis.decision_memory import DecisionRecord, DecisionMemory


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_record(
    date="2026-05-10",
    ticker="AI.PA",
    action="buy",
    quantity=10.0,
    price=150.0,
    portfolio_value_before=10000.0,
    portfolio_value_after=9850.0,
    rsi=None,
    bollinger_position=None,
    sma_20=None,
    sma_50=None,
    volatility=None,
    reasoning="test",
    exit_price=None,
    holding_period_days=None,
    pnl_pct=None,
    max_drawdown_during_hold=None,
):
    return DecisionRecord(
        date=date,
        ticker=ticker,
        action=action,
        quantity=quantity,
        price=price,
        portfolio_value_before=portfolio_value_before,
        portfolio_value_after=portfolio_value_after,
        rsi=rsi,
        bollinger_position=bollinger_position,
        sma_20=sma_20,
        sma_50=sma_50,
        volatility=volatility,
        reasoning=reasoning,
        exit_price=exit_price,
        holding_period_days=holding_period_days,
        pnl_pct=pnl_pct,
        max_drawdown_during_hold=max_drawdown_during_hold,
    )


def make_memory_file(tmp_path, records):
    """Write a list of DecisionRecord dicts to a temp file and return the path."""
    path = tmp_path / "decision_memory.json"
    path.write_text(json.dumps([r.to_dict() for r in records]))
    return str(path)


# ---------------------------------------------------------------------------
# DecisionRecord
# ---------------------------------------------------------------------------

class TestDecisionRecord:
    def test_to_dict_round_trip(self):
        r = make_record(rsi=30.0, bollinger_position=-0.5)
        d = r.to_dict()
        assert d["date"] == "2026-05-10"
        assert d["ticker"] == "AI.PA"
        assert d["rsi"] == 30.0
        assert d["bollinger_position"] == -0.5

    def test_from_dict_round_trip(self):
        r1 = make_record(rsi=30.0, bollinger_position=-0.5, pnl_pct=5.0)
        r2 = DecisionRecord.from_dict(r1.to_dict())
        assert r2.date == r1.date
        assert r2.ticker == r1.ticker
        assert r2.rsi == r1.rsi
        assert r2.pnl_pct == r1.pnl_pct

    def test_defaults(self):
        r = DecisionRecord(
            date="2026-01-01",
            ticker="X",
            action="hold",
            quantity=0.0,
            price=0.0,
            portfolio_value_before=0.0,
            portfolio_value_after=0.0,
        )
        assert r.rsi is None
        assert r.bollinger_position is None
        assert r.reasoning == ""
        assert r.exit_price is None


# ---------------------------------------------------------------------------
# DecisionMemory init / load
# ---------------------------------------------------------------------------

class TestDecisionMemoryInit:
    def test_default_memory_file(self, tmp_path):
        with patch.object(DecisionMemory, "_load_memory"):
            mem = DecisionMemory()
            assert mem.memory_file == Path("data/decision_memory.json")

    def test_custom_memory_file(self, tmp_path):
        path = tmp_path / "custom.json"
        path.write_text("[]")
        mem = DecisionMemory(memory_file=str(path))
        assert mem.decisions == []

    def test_loads_existing_records(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record(), make_record(ticker="MC.PA")])
        mem = DecisionMemory(memory_file=path)
        assert len(mem.decisions) == 2
        assert mem.decisions[0].ticker == "AI.PA"
        assert mem.decisions[1].ticker == "MC.PA"

    def test_missing_file_empty_list(self, tmp_path):
        path = tmp_path / "does_not_exist.json"
        mem = DecisionMemory(memory_file=str(path))
        assert mem.decisions == []

    def test_invalid_json_graceful(self, tmp_path, capsys):
        path = tmp_path / "bad.json"
        path.write_text("not json")
        mem = DecisionMemory(memory_file=str(path))
        assert mem.decisions == []
        captured = capsys.readouterr()
        assert "Warning" in captured.out


# ---------------------------------------------------------------------------
# save_memory / add_decision
# ---------------------------------------------------------------------------

class TestDecisionMemoryPersistence:
    def test_save_creates_file(self, tmp_path):
        path = tmp_path / "nested" / "mem.json"
        mem = DecisionMemory(memory_file=str(path))
        mem.add_decision(make_record())
        mem.save_memory()
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 1

    def test_add_decision_appends(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record()])
        mem = DecisionMemory(memory_file=path)
        mem.add_decision(make_record(ticker="MC.PA"))
        assert len(mem.decisions) == 2

    def test_auto_save_every_ten(self, tmp_path):
        path = tmp_path / "auto.json"
        mem = DecisionMemory(memory_file=str(path))
        for i in range(9):
            mem.add_decision(make_record(date=f"2026-01-{i+1:02d}"))
        assert not path.exists()  # not yet
        mem.add_decision(make_record(date="2026-01-10"))
        assert path.exists()
        data = json.loads(path.read_text())
        assert len(data) == 10


# ---------------------------------------------------------------------------
# update_outcomes (stub)
# ---------------------------------------------------------------------------

class TestUpdateOutcomes:
    def test_stub_does_not_crash(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record()])
        mem = DecisionMemory(memory_file=path)
        mem.update_outcomes("AI.PA", 160.0, "2026-05-11")
        # Should be a no-op stub
        assert mem.decisions[0].exit_price is None


# ---------------------------------------------------------------------------
# get_decision_summary
# ---------------------------------------------------------------------------

class TestGetDecisionSummary:
    def test_empty_decisions(self, tmp_path):
        mem = DecisionMemory(memory_file=str(tmp_path / "empty.json"))
        summary = mem.get_decision_summary(days=30)
        assert summary["total_decisions"] == 0
        assert "message" in summary

    def test_no_decisions_in_period(self, tmp_path):
        old = make_record(date="2026-01-01")
        path = make_memory_file(tmp_path, [old])
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=7)
        assert summary["total_decisions"] == 0

    def test_action_counts(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, action="buy"),
            make_record(date=today, action="buy"),
            make_record(date=today, action="sell"),
            make_record(date=today, action="hold"),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=30)
        assert summary["action_breakdown"]["buy"] == 2
        assert summary["action_breakdown"]["sell"] == 1
        assert summary["action_breakdown"]["hold"] == 1

    def test_win_rate_calculation(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=5.0),
            make_record(date=today, pnl_pct=-2.0),
            make_record(date=today, pnl_pct=0.0),  # zero = loser
            make_record(date=today, action="hold"),  # no pnl = not counted
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=30)
        assert summary["completed_trades"] == 3
        assert summary["win_rate"] == 1 / 3
        assert summary["best_trade"] == 5.0
        assert summary["worst_trade"] == -2.0

    def test_avg_holding_days(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=5.0, holding_period_days=5),
            make_record(date=today, pnl_pct=-2.0, holding_period_days=15),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=30)
        assert summary["avg_holding_days"] == 10.0

    def test_avg_pnl(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=10.0),
            make_record(date=today, pnl_pct=-4.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=30)
        assert summary["avg_pnl_pct"] == 3.0


# ---------------------------------------------------------------------------
# get_pattern_analysis
# ---------------------------------------------------------------------------

class TestGetPatternAnalysis:
    def test_insufficient_data(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record(pnl_pct=1.0) for _ in range(5)])
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert analysis["status"] == "insufficient_data"

    def test_basic_winners_losers(self, tmp_path):
        records = [
            make_record(pnl_pct=1.0 + i) for i in range(5)
        ] + [
            make_record(pnl_pct=-1.0 - i) for i in range(5)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert analysis["status"] == "ok"
        assert analysis["total_analyzed"] == 10
        assert analysis["winners"] == 5
        assert analysis["losers"] == 5

    def test_rsi_correlation_positive(self, tmp_path):
        records = [
            make_record(rsi=20.0 + i * 5, pnl_pct=-5.0 + i * 2.5)
            for i in range(10)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert "rsi_correlation" in analysis
        assert analysis["rsi_correlation"] > 0.3

    def test_rsi_correlation_negative(self, tmp_path):
        records = [
            make_record(rsi=80.0 - i * 5, pnl_pct=-5.0 + i * 2.5)
            for i in range(10)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert "rsi_correlation" in analysis
        assert analysis["rsi_correlation"] < -0.3

    def test_bollinger_correlation(self, tmp_path):
        records = [
            make_record(bollinger_position=-1.0 + i * 0.2, pnl_pct=-5.0 + i * 2.5)
            for i in range(10)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert "bollinger_correlation" in analysis
        assert analysis["bollinger_correlation"] > 0.3

    def test_holding_period_performance(self, tmp_path):
        records = [
            make_record(pnl_pct=2.0, holding_period_days=3),
            make_record(pnl_pct=3.0, holding_period_days=4),
            make_record(pnl_pct=-1.0, holding_period_days=10),
            make_record(pnl_pct=-2.0, holding_period_days=15),
            make_record(pnl_pct=5.0, holding_period_days=25),
            make_record(pnl_pct=6.0, holding_period_days=30),
            make_record(pnl_pct=1.0, holding_period_days=5),  # boundary: short_term
            make_record(pnl_pct=0.0, holding_period_days=20),  # boundary: medium_term
            make_record(pnl_pct=-0.5, holding_period_days=21),  # long_term
            make_record(pnl_pct=0.5, holding_period_days=1),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        hp = analysis["holding_period_performance"]
        assert hp["short_term_5d"] is not None
        assert hp["medium_term_5_20d"] is not None
        assert hp["long_term_20d_plus"] is not None
        # Short term should average (2+3+1+0.5)/4 = 1.625
        assert abs(hp["short_term_5d"] - 1.625) < 0.001

    def test_behavioral_overtrading_flag(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        # Need 10+ completed trades for pattern analysis + 20 decisions for behavioral calc
        records = [make_record(date=today, pnl_pct=1.0) for _ in range(10)]
        records += [make_record(date=today) for _ in range(10)]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert analysis["behavioral_indicators"]["overtrading_flag"] is True
        assert analysis["behavioral_indicators"]["avg_trades_per_day"] > 3

    def test_behavioral_no_overtrading(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
        two_days_ago = (datetime.now() - timedelta(days=2)).strftime("%Y-%m-%d")
        three_days_ago = (datetime.now() - timedelta(days=3)).strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=1.0) for _ in range(2)
        ] + [
            make_record(date=yesterday, pnl_pct=1.0) for _ in range(2)
        ] + [
            make_record(date=two_days_ago, pnl_pct=1.0) for _ in range(3)
        ] + [
            make_record(date=three_days_ago, pnl_pct=1.0) for _ in range(3)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert analysis["behavioral_indicators"]["overtrading_flag"] is False

    def test_recent_concentration(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [make_record(date=today, ticker="AI.PA", pnl_pct=1.0) for _ in range(5)]
        records += [make_record(date=today, ticker="MC.PA", pnl_pct=1.0) for _ in range(5)]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        assert analysis["behavioral_indicators"]["recent_concentration"] == 2


# ---------------------------------------------------------------------------
# generate_lessons_learned
# ---------------------------------------------------------------------------

class TestGenerateLessonsLearned:
    def test_no_decisions(self, tmp_path):
        mem = DecisionMemory(memory_file=str(tmp_path / "empty.json"))
        lessons = mem.generate_lessons_learned()
        assert len(lessons) == 1
        assert "No trading history" in lessons[0]

    def test_low_win_rate(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=-2.0),
            make_record(date=today, pnl_pct=-3.0),
            make_record(date=today, pnl_pct=1.0),
            make_record(date=today, pnl_pct=-1.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        win_rate_lessons = [l for l in lessons if "win rate" in l]
        assert len(win_rate_lessons) == 1
        assert "below random" in win_rate_lessons[0]

    def test_high_win_rate(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=2.0),
            make_record(date=today, pnl_pct=3.0),
            make_record(date=today, pnl_pct=1.0),
            make_record(date=today, pnl_pct=2.5),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        win_rate_lessons = [l for l in lessons if "win rate" in l]
        assert len(win_rate_lessons) == 1
        assert "showing edge" in win_rate_lessons[0]

    def test_overtrading_lesson(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [make_record(date=today) for _ in range(65)]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        overtrade = [l for l in lessons if "High trade frequency" in l]
        assert len(overtrade) == 1

    def test_negative_avg_pnl(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=-2.0),
            make_record(date=today, pnl_pct=-3.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        pnl_lessons = [l for l in lessons if "Average loss" in l]
        assert len(pnl_lessons) == 1

    def test_positive_avg_pnl(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        records = [
            make_record(date=today, pnl_pct=2.0),
            make_record(date=today, pnl_pct=3.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        pnl_lessons = [l for l in lessons if "Average gain" in l]
        assert len(pnl_lessons) == 1

    def test_rsi_mean_reversion_lesson(self, tmp_path):
        records = [
            make_record(rsi=80.0 - i * 5, pnl_pct=-5.0 + i * 2.5)
            for i in range(10)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        rsi_lessons = [l for l in lessons if "RSI" in l or "Lower RSI" in l]
        assert len(rsi_lessons) == 1
        assert "mean reversion" in rsi_lessons[0].lower() or "Lower RSI" in rsi_lessons[0]

    def test_rsi_momentum_lesson(self, tmp_path):
        records = [
            make_record(rsi=20.0 + i * 5, pnl_pct=-5.0 + i * 2.5)
            for i in range(10)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        rsi_lessons = [l for l in lessons if "RSI" in l or "Higher RSI" in l]
        assert len(rsi_lessons) == 1
        assert "momentum" in rsi_lessons[0].lower() or "Higher RSI" in rsi_lessons[0]

    def test_bollinger_lesson(self, tmp_path):
        # Negative correlation: lower Bollinger -> higher P&L
        records = [
            make_record(bollinger_position=1.0 - i * 0.2, pnl_pct=-5.0 + i * 2.5)
            for i in range(10)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        bb_lessons = [l for l in lessons if "Bollinger" in l]
        assert len(bb_lessons) == 1
        assert "lower" in bb_lessons[0].lower()

    def test_short_term_outperform_lesson(self, tmp_path):
        records = [
            make_record(pnl_pct=5.0, holding_period_days=3) for _ in range(5)
        ] + [
            make_record(pnl_pct=-1.0, holding_period_days=25) for _ in range(5)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        hp_lessons = [l for l in lessons if "Short-term" in l or "quicker profit" in l]
        assert len(hp_lessons) == 1

    def test_long_term_outperform_lesson(self, tmp_path):
        records = [
            make_record(pnl_pct=-1.0, holding_period_days=3) for _ in range(5)
        ] + [
            make_record(pnl_pct=5.0, holding_period_days=25) for _ in range(5)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        hp_lessons = [l for l in lessons if "Longer holds" in l or "Let winners" in l]
        assert len(hp_lessons) == 1

    def test_overtrading_behavioral_lesson(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        # Need 10+ completed trades for pattern analysis + overtrading flag
        records = [make_record(date=today, pnl_pct=1.0) for _ in range(10)]
        records += [make_record(date=today) for _ in range(10)]  # push avg > 3
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        ot_lessons = [l for l in lessons if "Overtrading" in l or "cooling-off" in l]
        assert len(ot_lessons) == 1

    def test_default_lesson_when_nothing_else(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        # Win rate exactly 0.5, avg_pnl 0.0, <60 decisions, <10 completed -> patterns insufficient
        records = [
            make_record(date=today, pnl_pct=1.0),
            make_record(date=today, pnl_pct=-1.0),
            make_record(date=today, pnl_pct=1.0),
            make_record(date=today, pnl_pct=-1.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        lessons = mem.generate_lessons_learned()
        # Should have building track record or similar (no win_rate lesson since 0.5)
        assert any("track record" in l or "consistent" in l for l in lessons)


# ---------------------------------------------------------------------------
# get_memory_context_for_llm
# ---------------------------------------------------------------------------

class TestGetMemoryContextForLlm:
    def test_includes_summary(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        path = make_memory_file(tmp_path, [make_record(date=today, pnl_pct=5.0)])
        mem = DecisionMemory(memory_file=path)
        context = mem.get_memory_context_for_llm()
        assert "Total decisions" in context
        assert "Win rate" in context
        assert "Average P&L" in context

    def test_includes_lessons(self, tmp_path):
        today = datetime.now().strftime("%Y-%m-%d")
        path = make_memory_file(tmp_path, [make_record(date=today, pnl_pct=5.0)])
        mem = DecisionMemory(memory_file=path)
        context = mem.get_memory_context_for_llm()
        assert "KEY LESSONS" in context


# ---------------------------------------------------------------------------
# export_to_dataframe
# ---------------------------------------------------------------------------

class TestExportToDataframe:
    def test_empty_returns_empty_df(self, tmp_path):
        mem = DecisionMemory(memory_file=str(tmp_path / "empty.json"))
        df = mem.export_to_dataframe()
        assert df.empty

    def test_non_empty_returns_df(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record(rsi=30.0), make_record(ticker="MC.PA")])
        mem = DecisionMemory(memory_file=path)
        df = mem.export_to_dataframe()
        assert len(df) == 2
        assert "ticker" in df.columns
        assert "rsi" in df.columns


# ---------------------------------------------------------------------------
# get_similar_decisions
# ---------------------------------------------------------------------------

class TestGetSimilarDecisions:
    def test_empty_returns_empty(self, tmp_path):
        mem = DecisionMemory(memory_file=str(tmp_path / "empty.json"))
        result = mem.get_similar_decisions("AI.PA", 50.0, 0.0)
        assert result == []

    def test_no_matching_ticker(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record(ticker="MC.PA", rsi=50.0, bollinger_position=0.0)])
        mem = DecisionMemory(memory_file=path)
        result = mem.get_similar_decisions("AI.PA", 50.0, 0.0)
        assert result == []

    def test_missing_rsi_or_bollinger_skipped(self, tmp_path):
        path = make_memory_file(tmp_path, [make_record(ticker="AI.PA")])
        mem = DecisionMemory(memory_file=path)
        result = mem.get_similar_decisions("AI.PA", 50.0, 0.0)
        assert result == []

    def test_sorts_by_similarity(self, tmp_path):
        records = [
            make_record(ticker="AI.PA", rsi=30.0, bollinger_position=-0.5),
            make_record(ticker="AI.PA", rsi=50.0, bollinger_position=0.0),
            make_record(ticker="AI.PA", rsi=70.0, bollinger_position=0.5),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        result = mem.get_similar_decisions("AI.PA", 50.0, 0.0, n=2)
        assert len(result) == 2
        # Closest should be the exact match (50.0, 0.0)
        assert result[0].rsi == 50.0
        assert result[0].bollinger_position == 0.0

    def test_respects_n_limit(self, tmp_path):
        records = [
            make_record(ticker="AI.PA", rsi=30.0 + i, bollinger_position=0.0)
            for i in range(20)
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        result = mem.get_similar_decisions("AI.PA", 50.0, 0.0, n=5)
        assert len(result) == 5

    def test_similarity_computation(self, tmp_path):
        # rsi_diff = |30-50|/100 = 0.2, bb_diff = |-0.5-0| = 0.5, total = 0.7
        # rsi_diff = |50-50|/100 = 0,   bb_diff = |0-0| = 0,     total = 0
        records = [
            make_record(ticker="AI.PA", rsi=30.0, bollinger_position=-0.5),
            make_record(ticker="AI.PA", rsi=50.0, bollinger_position=0.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        result = mem.get_similar_decisions("AI.PA", 50.0, 0.0, n=2)
        assert result[0].rsi == 50.0  # exact match first
        assert result[1].rsi == 30.0  # farther match second


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_zero_pnl_classified_as_loser(self, tmp_path):
        records = [
            make_record(pnl_pct=0.0),
            make_record(pnl_pct=1.0),
        ] + [make_record(pnl_pct=2.0) for _ in range(8)]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        # Zero P&L is classified as loser (pnl_pct > 0 required for winner)
        assert analysis["winners"] == 9
        assert analysis["losers"] == 1  # the zero P&L record

    def test_none_holding_period_ignored(self, tmp_path):
        records = [
            make_record(pnl_pct=1.0, holding_period_days=None),
            make_record(pnl_pct=2.0, holding_period_days=5),
        ] + [make_record(pnl_pct=1.0, holding_period_days=5) for _ in range(8)]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        analysis = mem.get_pattern_analysis()
        hp = analysis["holding_period_performance"]
        # Only the 5-day trades count; None is ignored
        assert hp["short_term_5d"] is not None

    def test_large_numbers(self, tmp_path):
        records = [
            make_record(date="2026-06-15", pnl_pct=999.0, holding_period_days=999),
            make_record(date="2026-06-15", pnl_pct=-999.0, holding_period_days=1),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=30)
        assert summary["best_trade"] == 999.0
        assert summary["worst_trade"] == -999.0

    def test_exact_boundary_date_included(self, tmp_path):
        # A record dated exactly `days` ago should still be included in the window.
        from unittest.mock import patch
        from analysis import decision_memory as dm

        records = [
            make_record(date="2026-06-15", pnl_pct=5.0),  # exactly 30 days before freeze
            make_record(date="2026-07-14", pnl_pct=3.0),
        ]
        path = make_memory_file(tmp_path, records)
        mem = DecisionMemory(memory_file=path)

        frozen_now = datetime(2026, 7, 15, 12, 0, 0)
        with patch.object(dm, "datetime", wraps=datetime) as mock_dt:
            mock_dt.now.return_value = frozen_now
            summary = mem.get_decision_summary(days=30)
        assert summary["total_decisions"] == 2
        assert summary["completed_trades"] == 2
        assert summary["best_trade"] == 5.0

    def test_very_old_dates_not_in_recent_summary(self, tmp_path):
        old = make_record(date="2020-01-01", pnl_pct=100.0)
        path = make_memory_file(tmp_path, [old])
        mem = DecisionMemory(memory_file=path)
        summary = mem.get_decision_summary(days=30)
        assert summary["total_decisions"] == 0
