"""Build data/vocab.json: catalog concepts, their HGNC mapping, and the alias map (key -> concept)."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict

from tagger import catalog_surfaces, keys, load_hgnc

HGNC = sys.argv[1] if len(sys.argv) > 1 else "/tmp/hgnc.txt"
analytes = [l.strip() for l in open("../data/msd_analytes.txt", encoding="utf-8") if l.strip()]
rows, index, sym_index, surfaces = load_hgnc(HGNC)
by_id = {r["hgnc_id"]: r for r in rows}


def map_hgnc(name: str) -> str | None:
    """Catalog name -> one HGNC id: approved-symbol match first, then a unique alias/name match."""
    parts = catalog_surfaces(name)
    for p in parts:
        for k in keys(p):
            if k in sym_index:
                return sym_index[k]
    for p in parts:
        hits = set()
        for k in keys(p):
            hits |= index.get(k, set())
        if len(hits) == 1:
            return next(iter(hits))
    return None


concepts: dict[str, dict] = {}
for a in analytes:
    hid = map_hgnc(a)
    if hid:
        cid = hid
    else:
        base = re.sub(r"\s*\([^)]*\)\s*", " ", a).strip() or a
        cid = "cat:" + sorted(keys(base))[0]
    c = concepts.setdefault(cid, {"id": cid, "catalog_names": [], "hgnc": hid,
                                  "symbol": by_id[hid]["symbol"] if hid else None})
    c["catalog_names"].append(a)

# surface forms per concept
forms: dict[str, set[str]] = defaultdict(set)
catalog_keys: dict[str, set[str]] = defaultdict(set)
for cid, c in concepts.items():
    for a in c["catalog_names"]:
        forms[cid].update(catalog_surfaces(a))
        for f in catalog_surfaces(a):
            catalog_keys[cid] |= keys(f)
    if c["hgnc"]:
        forms[cid].update(surfaces[c["hgnc"]])

# key -> concepts; a key is dropped if it names >1 concept here, or (HGNC-sourced) >1 human gene
key_to = defaultdict(set)
for cid, fs in forms.items():
    for f in fs:
        for k in keys(f):
            if len(k) >= 2:
                key_to[k].add(cid)
alias, dropped = {}, defaultdict(list)
for k, cids in key_to.items():
    # a key that is the catalog's own name for exactly one concept goes to that concept
    by_catalog = {c for c in cids if k in catalog_keys[c]}
    if len(by_catalog) == 1:
        cids = by_catalog
    if len(cids) > 1:
        dropped["multi_concept"].append(k)
        continue
    cid = next(iter(cids))
    genes = index.get(k, set())
    own = concepts[cid]["hgnc"]
    # the catalog's own name for a concept is kept even if HGNC uses it for another gene
    if k not in catalog_keys[cid] and genes and (genes - ({own} if own else set())):
        dropped["other_gene"].append(k)
        continue
    alias[k] = cid

catalog_genes = {c["hgnc"] for c in concepts.values() if c["hgnc"]}
block = sorted(k for k, g in index.items() if k not in alias and len(k) >= 2 and (g - catalog_genes))
# dev-split change (2026-11-25): the short-key case rule inherits from the catalog's own spelling.
# A short key the catalog writes as a plain word (Furin, Tau, Renin: one capital at most, letters
# only) matches any case. c-Kit, c-Met keep the uppercase rule.
case_free = sorted({k for cid, c in concepts.items() for a in c["catalog_names"] for f in catalog_surfaces(a)
                    if re.fullmatch(r"[A-Z]?[a-z]+", f) for k in keys(f) if k in alias and alias[k] == cid})
out = {"case_free": case_free, "hgnc_file_sha256": None, "n_catalog_names": len(analytes), "n_concepts": len(concepts),
       "n_hgnc_mapped": sum(1 for c in concepts.values() if c["hgnc"]),
       "n_alias_keys": len(alias), "dropped": {k: len(v) for k, v in dropped.items()},
       "concepts": concepts, "alias": alias, "block": block}
json.dump(out, open("data/vocab.json", "w"), indent=1, ensure_ascii=False)
print({k: v for k, v in out.items() if k not in ("concepts", "alias", "block")}, "block", len(block))
