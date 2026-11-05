"""Write realistic search queries for sampled chunks of muninn.austegard.com with headless Sonnet.

    python3 -I run/make_queries.py KBI_DIR/muninn [N=150]

Samples N chunks (seed 0) from the index's chunk shards, at most 2 per post, skipping chunks under 200
characters, and asks Sonnet (claude -p, tools off) for one query a reader who remembers the content
but not its wording would type into the site's search box. Writes data/queries.json: [{chunk_id, query}].
"""
import json, random, re, subprocess, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from _lib.paths import spoke   # EXPERIMENTS_SPOKES_ROOT holds remax, remax_kb, muninn.austegard.com
for p in (spoke("remax_kb"), spoke("remax") / "src", spoke("muninn.austegard.com") / "scripts"):
    sys.path.insert(0, str(p))
from remax_kb.read_v2 import KB
ROOT = Path(__file__).resolve().parents[1]
kb = KB.open(sys.argv[1] + ".kbi")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 150
rows = list(range(kb.live_count))
random.Random(0).shuffle(rows)
picked, per_post = [], {}
for r in rows:
    cid = kb._chunk_id_at(r); post = cid.rsplit("#", 1)[0]
    if per_post.get(post, 0) >= 2:
        continue
    text, meta, _ = kb._fetch_chunk(r)
    if len(text) < 200:
        continue
    per_post[post] = per_post.get(post, 0) + 1
    picked.append((cid, text[:900]))
    if len(picked) >= N:
        break
HARD = len(sys.argv) > 3 and sys.argv[3] == "hard"
SYS_HARD = ("You write search queries for evaluating a blog's search box. For each numbered passage, write the "
       "question a curious reader would type if they wanted what the passage explains but had never seen it: "
       "4 to 10 words, plain language. Do NOT use any word of five or more letters that appears in the passage; "
       "use synonyms and everyday words instead. Reply with ONLY a JSON object mapping each passage number "
       "(as a string) to its query.")
SYS = ("You write search queries for evaluating a blog's search box. For each numbered passage, write one query "
       "that a reader who remembers what the passage said, but not its exact words, would type to find it again: "
       "3 to 8 words, plain language, no quotes, and avoid copying distinctive phrases verbatim. "
       "Reply with ONLY a JSON object mapping each passage number (as a string) to its query.")
out = []
scratch = tempfile.mkdtemp()
for b in range(0, len(picked), 25):
    batch = picked[b:b + 25]
    prompt = "\n\n".join(f"[{i}] {t}" for i, (_, t) in enumerate(batch))
    r = subprocess.run(["claude", "-p", "--model", "sonnet", "--output-format", "json", "--setting-sources", "",
                        "--strict-mcp-config", "--no-session-persistence", "--tools", "", "--system-prompt", SYS_HARD if HARD else SYS],
                       input=prompt, capture_output=True, text=True, cwd=scratch, timeout=600)
    res = json.loads(r.stdout)["result"]
    d = json.loads(re.search(r"\{[\s\S]*\}", res).group(0))
    for i, (cid, _) in enumerate(batch):
        out.append({"chunk_id": cid, "query": d[str(i)]})
    print(b + len(batch), "queries", flush=True)
(ROOT / "data").mkdir(exist_ok=True)
# lexical overlap: share of the query's 5+ letter words that occur in the passage
texts = dict(picked)
for o in out:
    w = [x for x in re.findall(r"[a-z]{5,}", o["query"].lower())]
    o["overlap"] = round(sum(x in texts[o["chunk_id"]].lower() for x in w) / max(1, len(w)), 3)
(ROOT / "data" / ("queries_hard.json" if HARD else "queries.json")).write_text(json.dumps(out, indent=1))
print("mean overlap", round(sum(o["overlap"] for o in out) / len(out), 3))
print(json.dumps(out[:5], indent=1))
