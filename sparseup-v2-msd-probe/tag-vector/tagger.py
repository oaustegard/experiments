"""Catalog-term tag vocabulary and a dictionary tagger.

One dimension per catalog concept. Surface forms come from two sources only:
  1. the MSD catalog name itself (whole name, '/'-separated parts, name without its parenthetical);
  2. HGNC (approved symbol, name, alias/previous symbols and names) for concepts that map to a
     human gene.
Generic normalization collapses case, punctuation, hyphens/spaces and Greek letters
(written as the spelled name and as its Latin initial). No surface form is written by hand.
"""
from __future__ import annotations

import csv
import re
import unicodedata

GREEK = {"α": "alpha", "β": "beta", "γ": "gamma", "δ": "delta", "ε": "epsilon", "ζ": "zeta",
         "η": "eta", "θ": "theta", "ι": "iota", "κ": "kappa", "λ": "lambda", "μ": "mu",
         "ν": "nu", "ξ": "xi", "π": "pi", "ρ": "rho", "σ": "sigma", "τ": "tau", "φ": "phi",
         "χ": "chi", "ψ": "psi", "ω": "omega"}
GREEK_INITIAL = {"alpha": "a", "beta": "b", "gamma": "g", "delta": "d", "kappa": "k",
                 "epsilon": "e", "lambda": "l", "zeta": "z", "theta": "t", "omega": "o"}
MAX_NGRAM = 8
SPLIT = re.compile(r"[\s,;:()\[\]{}\"'“”‘’/-]+")


def _ascii(s: str) -> str:
    s = unicodedata.normalize("NFKC", s)
    for g, name in GREEK.items():
        s = s.replace(g, name).replace(g.upper(), name)
    return s


def keys(surface: str) -> set[str]:
    """Normalized match keys for one surface form: spelled Greek, and Latin-initial Greek."""
    s = _ascii(surface).lower()
    k1 = re.sub(r"[^a-z0-9]", "", s)
    out = {k1} if k1 else set()
    k2 = s
    for name, ini in GREEK_INITIAL.items():
        k2 = re.sub(rf"(?<=[0-9\-\s]){name}\b", ini, k2)
    k2 = re.sub(r"[^a-z0-9]", "", k2)
    if k2:
        out.add(k2)
    return out


def needs_case(key: str) -> bool:
    """Short purely alphabetic keys (ACE, KIT, MET) match only an uppercase-heavy raw form."""
    return key.isalpha() and len(key) <= 5


def upper_ok(raw: str) -> bool:
    return sum(c.isupper() for c in raw) >= 2


def catalog_surfaces(name: str) -> list[str]:
    out = [name]
    base = re.sub(r"\s*\([^)]*\)\s*", " ", name).strip()
    if base and base != name:
        out.append(base)
    for part in re.split(r"/", base):
        part = part.strip()
        if part and part not in out:
            out.append(part)
    return out


def load_hgnc(path: str):
    rows = list(csv.DictReader(open(path, encoding="utf-8"), delimiter="\t"))
    rows = [r for r in rows if r["status"] == "Approved"]
    index: dict[str, set[str]] = {}   # key -> hgnc ids
    sym_index: dict[str, str] = {}
    surfaces: dict[str, list[str]] = {}
    for r in rows:
        hid = r["hgnc_id"]
        forms = [r["symbol"], r["name"]]
        for f in ("alias_symbol", "prev_symbol", "alias_name", "prev_name"):
            forms += [x for x in r[f].split("|") if x]
        surfaces[hid] = forms
        for f in forms:
            for k in keys(f):
                index.setdefault(k, set()).add(hid)
        for k in keys(r["symbol"]):
            sym_index[k] = hid
    return rows, index, sym_index, surfaces


class Tagger:
    def __init__(self, alias: dict[str, str], block: set[str] | frozenset = frozenset(),
                 case_free: set[str] | frozenset = frozenset()):
        self.alias = alias            # key -> concept id
        self.case_free = case_free    # short keys the catalog itself writes in lower/title case (Furin, Tau)
        self.block = block            # keys of non-catalog human genes: matched, consumed, no dim
        self.maxlen = MAX_NGRAM

    def tag(self, text: str) -> list[str]:
        words = [w.strip(".") for w in SPLIT.split(text) if w.strip(".")]
        nk = [keys(w) for w in words]
        found: list[str] = []
        i = 0
        while i < len(words):
            hit = None
            for n in range(min(self.maxlen, len(words) - i), 0, -1):
                cands = {""}
                for j in range(i, i + n):
                    cands = {c + k for c in cands for k in nk[j]} if nk[j] else cands
                    if len(cands) > 16:
                        cands = set(list(cands)[:16])
                raw = " ".join(words[i:i + n])
                ok = [c for c in sorted(cands) if not needs_case(c) or c in self.case_free or upper_ok(raw)]
                for c in ok:
                    if c in self.alias:
                        hit = (self.alias[c], n)
                        break
                if not hit:
                    for c in ok:
                        if c in self.block:
                            hit = (None, n)
                            break
                if hit:
                    break
            if hit:
                if hit[0]:
                    found.append(hit[0])
                i += hit[1]
            else:
                i += 1
        return found
