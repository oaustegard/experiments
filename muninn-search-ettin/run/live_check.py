"""Live Worker against the same Worker code run locally on the same index: same top-5 pages?

    EXPERIMENTS_SPOKES_ROOT=... python3 -I run/live_check.py INDEX_DIR

INDEX_DIR holds muninn.kbi and muninn.kbc/ as published to KV. Writes results/live.json.
"""
import json, subprocess, sys, tempfile, time, urllib.request
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib.paths import spoke   # EXPERIMENTS_SPOKES_ROOT holds remax, remax_kb, muninn.austegard.com
sys.path.insert(0, str(spoke("muninn.austegard.com") / "scripts"))
from ettin_embedder import ensure_package
ROOT = Path(__file__).resolve().parents[1]
URL = "https://muninn-search.austegard.workers.dev/api/search"
Q = ["a decision model in a browser tab", "classify bluesky accounts as bots or humans", "centered SimHash",
     "teaching a raven to sleep", "launching subagents from a script", "WebGPU kernel for attention"]
with tempfile.TemporaryDirectory() as td:
    qf, of = Path(td) / "q.json", Path(td) / "o.json"
    qf.write_text(json.dumps([{"query": q} for q in Q]))
    subprocess.run(["node", str(ROOT / "run" / "worker_local.mjs"), sys.argv[1], str(ensure_package()),
                    str(spoke("muninn.austegard.com")), str(qf), str(of)], check=True, stdout=subprocess.DEVNULL)
    local = {r["query"]: [i.split("#")[0] for i in r["ids"]] for r in json.loads(of.read_text())}
rows = []
for q in Q:
    # an isolate still on the previous deploy answers 500, so retry; urllib's default user agent
    # gets Cloudflare error 1010, so send our own
    for attempt in range(3):
        req = urllib.request.Request(URL, data=json.dumps({"query": q, "k": 5}).encode(),
                                     headers={"content-type": "application/json", "user-agent": "muninn-search-live-check/1"})
        try:
            j = json.loads(urllib.request.urlopen(req, timeout=60).read()); break
        except Exception as e:
            j = {"error": str(e)}; time.sleep(5)
    if "hits" not in j:
        raise SystemExit(f"live search failed for {q!r}: {j}")
    live = [h["chunk_id"].split("#")[0] for h in j["hits"]]
    rows.append({"query": q, "live_top5": live, "same_as_local": live == local[q]})
out = {"date": time.strftime("%Y-%m-%d"), "queries": len(rows), "same_as_local": sum(r["same_as_local"] for r in rows), "rows": rows}
(ROOT / "results" / "live.json").write_text(json.dumps(out, indent=1)); print(out["queries"], out["same_as_local"])
