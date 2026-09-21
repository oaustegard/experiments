"""Assemble data/paper_corpus.jsonl (train.py format) from the worker's data/papers/*.jsonl,
and provide the cue regex used by the baseline and the cue-subset analysis.
Rows: {"url": "pmid:<id>", "label": "msd"|"not_msd", "title", "text": abstract, "has_cue", "neg_set", "year"}.
Also writes results/cue_regex.json: the regex baseline's precision/recall on the test split."""
import json, os, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train import split_rows

HERE = os.path.dirname(os.path.abspath(__file__))
CUE = re.compile(r"meso\s*scale|mesoscale|\bmsd\b|[vusr]-plex|quickplex|sector\s+imager|sulfo-?tag|multi-?array|electrochemiluminescen", re.I)

def has_cue(title, abstract): return bool(CUE.search((title or "") + " " + (abstract or "")))

def build():
    rows = []; seen = set()
    for fn, neg in [("positives.jsonl", None), ("hard.jsonl", "hard"), ("easy.jsonl", "easy")]:
        p = os.path.join(HERE, "data", "papers", fn)
        if not os.path.exists(p): print("missing", p); continue
        for l in open(p):
            if not l.strip(): continue
            r = json.loads(l); pmid = str(r.get("pmid") or "")
            if not pmid or pmid in seen or len((r.get("abstract") or "").split()) < 50: continue
            seen.add(pmid)
            rows.append({"url": f"pmid:{pmid}", "label": "msd" if neg is None else "not_msd", "title": r.get("title") or "", "text": r.get("abstract") or "",
                         "has_cue": has_cue(r.get("title"), r.get("abstract")), "neg_set": neg or "pos", "year": r.get("year"), "query": r.get("query")})
    out = os.path.join(HERE, "data", "paper_corpus.jsonl")
    with open(out, "w") as f:
        for r in rows: f.write(json.dumps(r) + "\n")
    pos = [r for r in rows if r["label"] == "msd"]; neg = [r for r in rows if r["label"] != "msd"]
    print(f"{len(rows)} rows: {len(pos)} msd ({sum(r['has_cue'] for r in pos)} with cue), {len(neg)} not_msd ({sum(r['has_cue'] for r in neg)} with cue; hard {sum(r['neg_set']=='hard' for r in neg)}, easy {sum(r['neg_set']=='easy' for r in neg)})")
    tr, dv, te = split_rows(rows)
    tp = sum(1 for r in te if r["label"] == "msd" and r["has_cue"]); fp = sum(1 for r in te if r["label"] != "msd" and r["has_cue"]); fn = sum(1 for r in te if r["label"] == "msd" and not r["has_cue"])
    res = {"arm": "cue_regex", "n_test": len(te), "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "tp": tp, "fp": fp, "fn": fn,
           "cue_free_positive_recall": 0.0, "test_positives": tp + fn, "test_cue_free_positives": fn}
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True); json.dump(res, open(os.path.join(HERE, "results", "cue_regex.json"), "w"), indent=1)
    print("cue regex on test:", res); print(f"done -> {out}")

if __name__ == "__main__": build()
