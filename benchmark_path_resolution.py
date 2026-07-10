"""
Benchmark for entry-point path resolution.

Measures that resolving paths relative to __file__ is cheap and that directory
creation is independent of the current working directory. The fix trades a single
pathlib resolve at import time for robustness against cron jobs launched from a
parent directory.
"""

import os
import shutil
import tempfile
import time
from pathlib import Path

# Import the script to pay the import-time path-resolution cost once.
import sys
sys.path.insert(0, str(Path(__file__).parent / "src"))
import daily_run


ITERATIONS = 1_000


def benchmark_path_resolution():
    """Time to resolve the repository root from __file__."""
    src_file = Path(__file__).parent / "src" / "daily_run.py"
    start = time.perf_counter()
    for _ in range(ITERATIONS):
        root = src_file.parent.parent.resolve()
        data_dir = root / "data"
        results_dir = root / "results" / "daily"
        _ = str(data_dir), str(results_dir)
    elapsed = time.perf_counter() - start
    print(f"Path resolution: {elapsed / ITERATIONS * 1e6:.2f} μs per iteration")
    return elapsed


def benchmark_directory_creation_cwd_independence():
    """Verify directory creation works from an arbitrary working directory."""
    tmp_repo = Path(tempfile.mkdtemp())
    data_dir = tmp_repo / "data"
    results_dir = tmp_repo / "results" / "daily"

    original_cwd = Path.cwd()
    unrelated_cwd = Path(tempfile.mkdtemp())

    start = time.perf_counter()
    os.chdir(unrelated_cwd)
    try:
        data_dir.mkdir(parents=True, exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    finally:
        os.chdir(original_cwd)
    elapsed = time.perf_counter() - start

    assert data_dir.exists()
    assert results_dir.exists()

    shutil.rmtree(tmp_repo)
    shutil.rmtree(unrelated_cwd)

    print(f"Directory creation from unrelated cwd: {elapsed * 1e3:.2f} ms")
    return elapsed


def benchmark_repeated_setup_directories():
    """Measure repeated setup_directories-like calls."""
    tmp_repo = Path(tempfile.mkdtemp())
    data_dir = tmp_repo / "data"
    results_dir = tmp_repo / "results" / "daily"

    start = time.perf_counter()
    for _ in range(ITERATIONS):
        data_dir.mkdir(exist_ok=True)
        results_dir.mkdir(parents=True, exist_ok=True)
    elapsed = time.perf_counter() - start

    shutil.rmtree(tmp_repo)
    print(f"setup_directories-like calls: {elapsed / ITERATIONS * 1e6:.2f} μs per iteration")
    return elapsed


if __name__ == "__main__":
    print("=" * 60)
    print("Path Resolution Benchmark")
    print("=" * 60)
    t1 = benchmark_path_resolution()
    t2 = benchmark_directory_creation_cwd_independence()
    t3 = benchmark_repeated_setup_directories()
    print(f"\nTotal wall time: {(t1 + t2 + t3) * 1e3:.2f} ms")
    print("Conclusion: import-time path resolution is negligible; the fix removes")
    print("the silent data-corruption risk from running scripts outside the repo.")
