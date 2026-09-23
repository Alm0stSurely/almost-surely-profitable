"""Suite-level guards shared by every test.

The daily pipeline and the intraday monitor persist runtime state under
``data/`` that is **gitignored**: a test writing those files mutates
production state invisibly (``git status`` stays clean) and can silently
destroy real alert-dedup history. The session fixture below snapshots the
known runtime state files before the first test and fails the suite after
the last test if any of them changed on disk.

When this guard fires, the fix is never to weaken the guard: redirect the
path with ``monkeypatch`` inside the offending test (see
``tests/test_monitor_state_quarantine.py`` for the house pattern).
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent

# Gitignored runtime state written by the monitor / daily pipeline. Extend
# this tuple when a new runtime state file appears in data/.
RUNTIME_STATE_FILES = (
    REPO_ROOT / "data" / "alert_history.json",
    REPO_ROOT / "data" / "market_state.json",
)


def _snapshot(path: Path):
    """Return an (mtime_ns, size, content) tuple, or None if missing."""
    if not path.exists():
        return None
    st = path.stat()
    return (st.st_mtime_ns, st.st_size, path.read_bytes())


@pytest.fixture(autouse=True, scope="session")
def _runtime_state_isolation():
    """Fail the suite if any test writes production runtime state.

    Snapshots the gitignored state files before the first test and compares
    after the last test. A load-then-save round trip through a production
    path changes mtime and content even when the data is "the same", and a
    test-recorded alert poisons the real dedup history — both are leaks.
    """
    before = {p: _snapshot(p) for p in RUNTIME_STATE_FILES}
    yield
    leaks = [p.name for p in RUNTIME_STATE_FILES if _snapshot(p) != before[p]]
    assert not leaks, (
        f"test suite wrote production runtime state: {leaks}. "
        "Redirect the path with monkeypatch instead — see "
        "tests/test_monitor_state_quarantine.py for the house pattern."
    )
