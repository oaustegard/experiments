"""Mine merged PRs into a file-localization query set.

Query  = PR title + first 400 chars of body.
Truth  = the files that PR changed, restricted to files the index actually
         carries (extension filters, size caps and later deletions all mean a
         changed path may have no chunk; scoring against an unreachable target
         measures the corpus, not the ranker).
Repo   = known, which is the point: it lets recall be split by repo size.
"""
import collections
import json
import os
import random
import sys
import urllib.request

sys.path.insert(0, "/home/claude/cw/scripts")
from pathlib import Path

import xr

TOK = os.environ["GH_TOKEN"]

def api(path):
    r = urllib.request.Request("https://api.github.com" + path,
        headers={"Authorization": f"token {TOK}", "User-Agent": "muninn-raven",
                 "Accept": "application/vnd.github+json"})
    with urllib.request.urlopen(r, timeout=30) as f:
        return json.load(f)

idx = xr.Index(max(Path(xr.CACHE).glob("prepared-*")))
files = {str(f) for f in idx.files}
repos = collections.Counter(f.split("/")[0] for f in files)
Path("repo_files.json").write_text(json.dumps(repos))

random.seed(0)
out = []
names = sorted(repos, key=lambda r: -repos[r])
for name in names:
    if repos[name] < 40:            # too small to localize into
        continue
    try:
        prs = api(f"/repos/oaustegard/{name}/pulls?state=closed&per_page=100")
    except Exception as e:  # noqa: BLE001 - one unreachable repo must not
        print("skip", name, type(e).__name__)  # end the sweep
        continue
    merged = [p for p in prs if p.get("merged_at")]
    random.shuffle(merged)
    kept = 0
    for p in merged:
        if kept >= 8:
            break
        body = (p.get("body") or "").strip()
        q = (p["title"] + " " + body[:400]).strip()
        if len(q) < 60:
            continue
        try:
            ch = api(f"/repos/oaustegard/{name}/pulls/{p['number']}/files?per_page=100")
        except Exception:  # noqa: BLE001, S112 - same: skip this PR, keep going
            continue
        truth = [f"{name}/{c['filename']}" for c in ch
                 if f"{name}/{c['filename']}" in files]
        if not truth:
            continue
        out.append({"repo": name, "pr": p["number"], "q": q, "truth": truth})
        kept += 1
    print(f"{name}: {kept} queries ({repos[name]} chunks)", flush=True)

Path("queries.json").write_text(json.dumps(out, indent=1))
print("TOTAL", len(out))
Path("/tmp/qs.done").write_text("ok")
