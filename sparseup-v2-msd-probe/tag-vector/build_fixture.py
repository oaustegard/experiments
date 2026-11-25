"""Assemble data/fixture.json from PubMed metadata dumps (saved PubMed MCP get_article_metadata output).

Inputs: data/mesh_map.json (concept index -> MeSH descriptor, verified), raw dumps passed as args.
Relevance: a doc is relevant to a concept when the concept's MeSH descriptor is among the doc's
MeSH terms (NLM indexing). Concepts whose descriptor is shared with another concept are dropped.
Split: 30% of concepts dev, 70% test, seeded.
"""
from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter

vocab = json.load(open("data/vocab.json", encoding="utf-8"))
cids = list(vocab["concepts"])
mm = json.load(open("data/mesh_map.json", encoding="utf-8"))   # {"<idx>": {"mesh":..., "symbol_ok": bool}}

docs, seen = [], set()
def records(path):
    """Either a saved get_article_metadata JSON dump or a ' ||| ' line file (pmid, title, mesh;..., abstract)."""
    raw = open(path, encoding="utf-8").read()
    if raw.lstrip().startswith("{") or '"articles"' in raw[:2000]:
        for a in json.loads(raw[raw.index("{"):raw.rindex("}") + 1])["articles"]:
            yield a["identifiers"]["pmid"], a.get("title") or "", a.get("mesh_terms") or [], a.get("abstract") or ""
        return
    for line in raw.splitlines():
        parts = line.split(" ||| ")
        if len(parts) != 4 or not parts[0].strip().isdigit():
            continue
        pmid, title, mesh, abstract = (p.strip() for p in parts)
        yield pmid, title, [m.strip() for m in mesh.split(";") if m.strip()], "" if abstract == "NA" else abstract


for path in sys.argv[1:]:
    for pmid, title, mesh, abstract in records(path):
        if pmid in seen or len(abstract) < 200:
            continue
        seen.add(pmid)
        docs.append({"pmid": pmid, "title": title, "abstract": abstract, "mesh": mesh})
docs.sort(key=lambda d: int(d["pmid"]))

# Label check: the descriptor I named must be how NLM indexed the seed docs. Where it isn't (a name
# PubMed's search translated to another heading), report it; the concept is dropped, not re-labelled.
seed = {l.split("\t")[0]: l.split("\t")[1].split() for l in open("data/seed_pmids.tsv") if l.strip()}
by_pmid = {d["pmid"]: d for d in docs}
for idx, v in list(mm.items()):
    got = [by_pmid[p] for p in seed.get(idx, []) if p in by_pmid]
    hit = sum(v["mesh"] in d["mesh"] for d in got)
    if got and hit < len(got) / 2:
        common = Counter(m for d in got for m in d["mesh"]).most_common(3)
        print("label mismatch", idx, v["mesh"], f"{hit}/{len(got)}", common, file=sys.stderr)
        del mm[idx]

count = Counter(v["mesh"] for v in mm.values())
concepts = []
for idx, v in sorted(mm.items(), key=lambda kv: int(kv[0])):
    if count[v["mesh"]] > 1:
        continue
    c = vocab["concepts"][cids[int(idx)]]
    forms = {"catalog": c["catalog_names"][0], "mesh": v["mesh"]}
    if c["symbol"] and v.get("symbol_ok", True):
        forms["symbol"] = c["symbol"]
    n_rel = sum(1 for d in docs if v["mesh"] in d["mesh"])
    concepts.append({"idx": int(idx), "id": c["id"], "mesh": v["mesh"], "forms": forms, "n_rel": n_rel})
concepts = [c for c in concepts if c["n_rel"] >= 3]
rnd = random.Random(20261125)
order = list(range(len(concepts)))
rnd.shuffle(order)
ndev = round(0.3 * len(concepts))
for r, i in enumerate(order):
    concepts[i]["split"] = "dev" if r < ndev else "test"

blob = json.dumps(docs, ensure_ascii=False, sort_keys=True).encode()
fx = {"n_docs": len(docs), "docs_sha256": hashlib.sha256(blob).hexdigest(), "concepts": concepts, "docs": docs}
json.dump(fx, open("data/fixture.json", "w"), ensure_ascii=False)
with open("data/fixture_labels.tsv", "w") as f:   # compact pin: pmid -> label concept idx
    lab = {c["mesh"]: c["idx"] for c in concepts}
    for d in docs:
        f.write(d["pmid"] + "\t" + ",".join(str(lab[m]) for m in d["mesh"] if m in lab) + "\n")
print({"docs": len(docs), "concepts": len(concepts), "dev": ndev, "sha": fx["docs_sha256"][:12],
       "n_rel_median": sorted(c["n_rel"] for c in concepts)[len(concepts) // 2]})
