"""Regression guard: the daily pipeline must anchor the LLM decision history.

Background (2026-09-24 research session): ``daily_run.py`` constructed
``TradingAgent()`` with the default relative ``history_file``. When cron ran
the pipeline from the workspace root instead of the repo directory, decisions
were silently written to ``<workspace>/data/decision_history.json``, splitting
the canonical history. 21 scattered decisions (2026-07-24 .. 2026-09-24) were
routed away from the repo and missing from every research analysis.

The guard parses ``src/daily_run.py`` and requires the ``TradingAgent`` call
to pass an explicit ``history_file`` keyword anchored to ``DATA_DIR``.
"""

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DAILY_RUN = REPO_ROOT / "src" / "daily_run.py"


def _trading_agent_calls(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            if name == "TradingAgent":
                yield node


def test_daily_run_anchors_trading_agent_history_file():
    source = DAILY_RUN.read_text()
    tree = ast.parse(source)

    calls = list(_trading_agent_calls(tree))
    assert calls, "daily_run.py must construct a TradingAgent"

    for call in calls:
        kw_names = {kw.arg for kw in call.keywords if kw.arg is not None}
        assert "history_file" in kw_names, (
            "TradingAgent() in daily_run.py must be given an explicit "
            "history_file anchored to DATA_DIR; a bare default relative path "
            "silently routes decisions to the cron working directory "
            "(regression: split decision history 2026-07-24..2026-09-24)."
        )


def test_daily_run_history_file_is_repo_anchored():
    """The anchored path must resolve inside the repo data dir, not the CWD."""
    source = DAILY_RUN.read_text()
    tree = ast.parse(source)

    for call in _trading_agent_calls(tree):
        hist_kw = next(kw for kw in call.keywords if kw.arg == "history_file")
        # Accept either str(DATA_DIR / "decision_history.json") or a plain
        # absolute path string literal.
        if isinstance(hist_kw.value, ast.Constant):
            path = Path(hist_kw.value.value)
            assert path.is_absolute(), "history_file literal must be absolute"
        else:
            # BinOp(str, DATA_DIR / "..."): require DATA_DIR as an operand.
            src_segment = ast.get_source_segment(source, hist_kw.value) or ""
            assert "DATA_DIR" in src_segment, (
                f"history_file must be anchored to DATA_DIR, got: {src_segment}"
            )
