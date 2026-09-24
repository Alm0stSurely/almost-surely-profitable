"""One-off: merge workspace-root stray decision history back into the repo.

Background: daily_run.py used a relative history_file, so runs executed with
CWD=workspace root wrote to <workspace>/data/decision_history.json. Merge those
entries (dedup by timestamp) into repos/almost-surely-profitable/data/decision_history.json.

Backups written alongside each source file as *.bak-merge-20260924.
"""

import json
import shutil
from pathlib import Path

WS = Path("/home/deploy/hermes-workspaces/clawmogorov")
STRAY = WS / "data" / "decision_history.json"
REPO_HIST = WS / "repos" / "almost-surely-profitable" / "data" / "decision_history.json"

stray = json.loads(STRAY.read_text())
canonical = json.loads(REPO_HIST.read_text())

shutil.copy2(STRAY, STRAY.with_suffix(".json.bak-merge-20260924"))
shutil.copy2(REPO_HIST, REPO_HIST.with_suffix(".json.bak-merge-20260924"))

by_ts = {d["timestamp"]: d for d in canonical}
added = 0
for d in stray:
    if d["timestamp"] not in by_ts:
        by_ts[d["timestamp"]] = d
        added += 1

merged = sorted(by_ts.values(), key=lambda d: d["timestamp"])
REPO_HIST.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

print(f"canonical before: {len(canonical)}, stray: {len(stray)}, added: {added}")
print(f"merged total: {len(merged)}")
print(f"range: {merged[0]['timestamp'][:10]} -> {merged[-1]['timestamp'][:10]}")

# Validate JSON-safe serialization (strict boundary per PR #53)
from decimal import Decimal
for d in merged:
    json.dumps(d, allow_nan=False)
print("strict allow_nan=False serialization: OK")
