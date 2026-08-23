#!/usr/bin/env python3
"""Pull the writing samples off issue #244 into samples/<model>.md."""
import json, os, re, subprocess, sys
from pathlib import Path

OUT = Path(__file__).parent / "samples"
OUT.mkdir(exist_ok=True)
URL = "https://api.github.com/repos/oaustegard/claude-workspace/issues/244/comments?per_page=100"

raw = subprocess.run(
    ["curl", "-sS", "-H", f"Authorization: Bearer {os.environ['GH_TOKEN']}",
     "-H", "Accept: application/vnd.github+json", URL],
    capture_output=True, text=True, check=True).stdout
comments = json.loads(raw)

FENCE = re.compile(r"````+\s*markdown\n(.*?)\n````+", re.S)
MODEL = re.compile(r"^MODEL:\s*(\S+)", re.M)

# Comments arrive oldest first, so the suffix is stable across re-runs: the
# first sample a model posts is always -a, the second -b.
seen: dict[str, int] = {}
n = 0
for c in comments:
    body = c["body"]
    m, f = MODEL.search(body), FENCE.search(body)
    if not (m and f):
        print(f"skip comment {c['id']}: no MODEL line or fence", file=sys.stderr)
        continue
    name = m.group(1).replace("claude-", "").replace("-20251001", "")
    seen[name] = seen.get(name, 0) + 1
    stem = f"{name}-{chr(ord('a') + seen[name] - 1)}"
    (OUT / f"{stem}.md").write_text(f.group(1).strip() + "\n")
    print(f"{stem}.md  <- comment {c['id']}")
    n += 1
print(f"{n} sample(s) across {len(seen)} model(s)")
