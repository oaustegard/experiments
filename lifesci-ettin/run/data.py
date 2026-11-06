"""Training texts for the distillation arms, and the held-out PubMed title->abstract eval.

    python3 -I run/data.py   # .work/data/{train.jsonl, pubmed_eval.json, stats.json}

Abstracts: lifesci-vocab-probe/data/abstracts_*.jsonl (PubMed E-utilities, 2010-2025). Removed: any
abstract whose normalised title matches a SciFact / NFCorpus / TREC-COVID corpus title. Held out: the
last 2,000 of the 2023-25 window (the vocab probe's test set), as a title -> abstract retrieval eval.
Train rows: role "q" = the title (a query), role "d" = a passage of the abstract (sentences packed to
<= 90 words), each with its PMID.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from evalset import SETS, load  # noqa: E402

SRC = HERE.parent / "lifesci-vocab-probe" / "data"
OUT = HERE / ".work" / "data"
norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[])")


def split_title(text):
    """fetch.py stored title + ' ' + abstract; the title is the first sentence ending in '.' before the abstract."""
    m = re.match(r"(.{20,400}?[.?])\s+(.*)", text, re.S)
    return (m.group(1), m.group(2)) if m else ("", text)


def passages(abstract, max_words=90):
    out, cur = [], []
    for s in SENT.split(abstract):
        w = s.split()
        if cur and len(cur) + len(w) > max_words:
            out.append(" ".join(cur))
            cur = []
        cur += w
    if cur:
        out.append(" ".join(cur))
    return out


def main():
    rows = {}
    for p in sorted(SRC.glob("abstracts_*.jsonl")):
        for l in p.open():
            r = json.loads(l)
            rows.setdefault(r["pmid"], (p.name, r["text"]))
    evtitles = set()
    for n in SETS:
        d, _, _ = load(n)
        evtitles |= {norm(t) for t in d["title"] if len(t) > 15}
    recent = [k for k, (f, _) in rows.items() if "2023" in f]
    held = set(recent[-2000:])
    stats = {"abstracts": len(rows), "eval_title_overlap_removed": 0, "held_out": len(held)}
    OUT.mkdir(parents=True, exist_ok=True)
    ev = {"ids": [], "title": [], "text": []}
    n_q = n_d = 0
    with (OUT / "train.jsonl").open("w") as f:
        for k, (_, text) in rows.items():
            title, ab = split_title(text)
            if norm(title) in evtitles:
                stats["eval_title_overlap_removed"] += 1
                continue
            if k in held:
                ev["ids"].append(k); ev["title"].append(title); ev["text"].append(ab)
                continue
            if title:
                f.write(json.dumps({"pmid": k, "role": "q", "t": title}) + "\n"); n_q += 1
            for ps in passages(ab):
                if len(ps.split()) >= 12:
                    f.write(json.dumps({"pmid": k, "role": "d", "t": ps}) + "\n"); n_d += 1
    (OUT / "pubmed_eval.json").write_text(json.dumps(ev))
    stats |= {"train_queries": n_q, "train_passages": n_d, "pubmed_eval_docs": len(ev["ids"])}
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print(stats)


if __name__ == "__main__":
    main()
