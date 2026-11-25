"""Measure how linkup-sparseup-embed-v2's tokenizer fragments MSD catalog names.

Tokenizer only. The model weights (and its vocab_fold_index buffer) were not
reachable from the sandbox, so nothing here runs the SPLADE head: expansions
are not measured, and case_space folding is approximated by lowercasing and
dropping the byte-level leading-space marker.

Each term is tokenized the way modeling_splade.py sees it inside a document:
with add_prefix_space=True (a leading " "), no [CLS]/[SEP], no [D]/[Q] prefix.
"""
import json, re, statistics, unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
DATA, OUT = HERE / "data", HERE / "results"

# byte-level BPE alphabet -> bytes, to show pieces as real text
_bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
_cs = _bs[:]
_n = 0
for _b in range(256):
    if _b not in _bs:
        _bs.append(_b); _cs.append(256 + _n); _n += 1
BYTE_OF = {chr(c): b for b, c in zip(_bs, _cs)}


def piece_text(tok):
    return bytes(BYTE_OF[ch] for ch in tok).decode("utf-8", errors="replace")


def fold(tok):
    """Approximate case_space fold: drop leading space, lowercase."""
    return piece_text(tok).lstrip(" ").lower()


def load_tok():
    p = HERE / "tokenizer.json"
    if not p.exists():
        from build_tokenizer import build
        build(p)
    return Tokenizer.from_file(str(p))


def is_letter(ch):
    return unicodedata.category(ch).startswith("L")


def letter_runs(s):
    runs, i = [], 0
    while i < len(s):
        if is_letter(s[i]):
            j = i
            while j < len(s) and is_letter(s[j]):
                j += 1
            runs.append((i, j)); i = j
        else:
            i += 1
    return runs


def analyse(tk, name):
    text = " " + name
    enc = tk.encode(text, add_special_tokens=False)
    toks, offs = enc.tokens, enc.offsets
    runs = letter_runs(text)
    frag_runs, shards = [], []
    for a, b in runs:
        idx = [k for k, (s, e) in enumerate(offs) if s < b and e > a]
        if len(idx) > 1:
            frag_runs.append(text[a:b])
            for k in idx:
                letters = "".join(c for c in piece_text(toks[k]) if is_letter(c))
                if 0 < len(letters) <= 2:
                    shards.append(piece_text(toks[k]).strip())
    dims = [fold(t) for t in toks]
    return {
        "name": name,
        "pieces": [piece_text(t) for t in toks],
        "n": len(toks),
        "fragmented_words": frag_runs,
        "shards": shards,
        "dims": sorted(set(d for d in dims if d)),
    }


def summarise(rows):
    n = [r["n"] for r in rows]
    return {
        "count": len(rows),
        "tokens_mean": round(statistics.mean(n), 2),
        "tokens_median": statistics.median(n),
        "share_3plus": round(sum(x >= 3 for x in n) / len(n), 3),
        "share_single_token": round(sum(x == 1 for x in n) / len(n), 3),
        "share_fragmented_word": round(sum(bool(r["fragmented_words"]) for r in rows) / len(rows), 3),
        "share_with_shard": round(sum(bool(r["shards"]) for r in rows) / len(rows), 3),
    }


TERM_SPLIT = re.compile(r"[\s/(),;&]+")


def catalog_terms(names):
    terms = Counter()
    for nm in names:
        for t in TERM_SPLIT.split(nm):
            t = t.strip("-.")
            if t and any(is_letter(c) for c in t):
                terms[t.lower()] += 1
    return terms


def main():
    tk = load_tok()
    sets = {k: [l.strip() for l in (DATA / f"{f}.txt").read_text().splitlines() if l.strip()]
            for k, f in [("msd_analytes", "msd_analytes"), ("msd_products", "msd_products"),
                         ("lifesci", "lifesci"), ("plain", "plain")]}
    rows = {k: [analyse(tk, x) for x in v] for k, v in sets.items()}
    summary = {k: summarise(v) for k, v in rows.items()}

    # Greek letters and hyphen/number analytes
    probes = ["IL-1β", "IL-1 beta", "TNF-α", "TNF-alpha", "TNF alpha", "IFN-γ", "IFN-gamma",
              "IFN-λ1", "MIP-1α", "SDF-1α", "TGF-β1", "Aβ42", "Abeta42", "amyloid beta 42",
              "pTau-217", "Tau (pT217)", "p-tau217", "IL-6", "IL-16", "IL-17A/F", "IL-12/IL-23p40",
              "CXCL10", "IP-10", "NT-proBNP", "TNFRSF9", "V-PLEX", "U-PLEX", "S-PLEX", "MULTI-SPOT",
              "electrochemiluminescence"]
    greek = [analyse(tk, p) for p in probes]
    greek_chars = {}
    for ch, word in [("α", "alpha"), ("β", "beta"), ("γ", "gamma"), ("δ", "delta"), ("κ", "kappa"), ("λ", "lambda")]:
        a, w = analyse(tk, ch), analyse(tk, word)
        greek_chars[ch] = {"dims": a["dims"], "word_dims": w["dims"],
                           "shares_dim_with_word": bool(set(a["dims"]) & set(w["dims"]))}
    an = [r for r in rows["msd_analytes"]]
    greek_names = [r for r in an if re.search("[α-ωΑ-Ω]", r["name"])]

    # Terms that would need a new vocab dim to be one sparse coordinate
    names = sets["msd_analytes"] + sets["msd_products"]
    terms = catalog_terms(names)
    need = {t: c for t, c in terms.items() if len(tk.encode(" " + t, add_special_tokens=False).tokens) > 1}
    single = {t: c for t, c in terms.items() if t not in need}
    variant = re.compile(r"^([a-z]\d{3}[a-z]|[a-z]{1,3}(\.\d+)+)$")
    need_nv = {t: c for t, c in need.items() if not variant.match(t)}
    terms_nv = [t for t in terms if not variant.match(t)]
    whole_names_single = sum(1 for r in rows["msd_analytes"] if r["n"] == 1)

    # Bag-of-dims confusion among analytes (SPLADE max-pools, so order is lost)
    dimsets = {r["name"]: frozenset(r["dims"]) for r in an}
    by_set = defaultdict(list)
    for nm, d in dimsets.items():
        by_set[d].append(nm)
    collisions = [v for v in by_set.values() if len(v) > 1]
    subsumed = []
    for a, da in dimsets.items():
        others = [b for b, db in dimsets.items() if b != a and da <= db]
        if others:
            subsumed.append({"name": a, "covered_by": len(others), "example": others[:3]})
    cross = [x for x in subsumed
             if any(x["name"].lower() not in b.lower()
                    for b, db in dimsets.items() if b != x["name"] and dimsets[x["name"]] <= db)]
    non_sars = [x for x in cross if "SARS" not in x["name"]]
    dim_freq = Counter(d for s in dimsets.values() for d in s)
    short = [len([d for d in s if len(d) <= 2]) / len(s) for s in dimsets.values()]

    from check_tokenizer import check
    n_checks, fails = check(HERE / "tokenizer.json")
    added = [a for a in json.loads((HERE / "tokenizer.json").read_text())["added_tokens"]]
    res = {
        "tokenizer": {"vocab_size": tk.get_vocab_size(), "checks": n_checks,
                      "checks_passed": n_checks - len(fails),
                      "unused_slots": sum(a["content"].startswith("[unused") for a in added),
                      "il_fold_demo": {"IL-6": analyse(tk, "IL-6")["dims"],
                                       "electrochemiluminescence": analyse(tk, "electrochemiluminescence")["dims"]}},
        "summary": summary,
        "greek": {"chars": greek_chars, "probes": {g["name"]: " | ".join(x.strip() for x in g["pieces"]) for g in greek},
                  "analytes_with_greek": len(greek_names),
                  "greek_analyte_tokens_mean": round(statistics.mean(r["n"] for r in greek_names), 2)},
        "new_dims": {
            "distinct_terms": len(terms),
            "terms_single_token": len(single),
            "terms_needing_new_dim": len(need),
            "share_needing_new_dim": round(len(need) / len(terms), 3),
            "analyte_names_single_token": whole_names_single,
            "non_variant_terms": len(terms_nv),
            "non_variant_terms_needing_new_dim": len(need_nv),
            "share_non_variant_needing_new_dim": round(len(need_nv) / len(terms_nv), 3),
            "top_non_variant_multi_piece": dict(sorted(need_nv.items(), key=lambda kv: -kv[1])[:20]),
        },
        "bag_confusion": {
            "analytes": len(dimsets),
            "exact_collision_groups": collisions,
            "names_in_collisions": sum(len(v) for v in collisions),
            "names_subsumed": len(subsumed),
            "share_subsumed": round(len(subsumed) / len(dimsets), 3),
            "names_cross_subsumed": len(cross),
            "share_cross_subsumed": round(len(cross) / len(dimsets), 3),
            "names_cross_subsumed_non_sars": len(non_sars),
            "cross_subsumed_non_sars": [
                x["name"] + " in " + next(b for b, db in dimsets.items()
                                          if b != x["name"] and dimsets[x["name"]] <= db
                                          and x["name"].lower() not in b.lower())
                for x in non_sars],
            "mean_share_dims_len_le2": round(statistics.mean(short), 3),
            "most_shared_dims": dict(dim_freq.most_common(12)),
        },
    }
    OUT.mkdir(exist_ok=True)
    (OUT / "results.json").write_text("{\n" + ",\n".join(
        f' "{k}": {{\n' + ",\n".join(f'  "{k2}": {json.dumps(v2, ensure_ascii=False)}' for k2, v2 in v.items()) + "\n }"
        for k, v in res.items()) + "\n}\n")
    with open(OUT / "per_name.jsonl", "w") as f:
        for k, v in rows.items():
            for r in v:
                f.write(json.dumps({"set": k, **r}, ensure_ascii=False) + "\n")
    print(json.dumps(summary, indent=1))
    print(json.dumps(res["new_dims"], ensure_ascii=False))
    print(json.dumps(res["bag_confusion"], ensure_ascii=False))


if __name__ == "__main__":
    main()
