"""Arm 1 — direct tag recall from SPARSEUP top-k expansion terms, no training.

For every labelled memory and k in (16, 32, 64): the fraction of its labels that
show up among its top-k expansion terms, under three hit definitions:
  (a) exact-token  — the tag string (lowercased) equals a decoded top-k token
                      after stripping the BPE leading-space marker "Ġ".
  (b) all-parts    — split the tag on "-"/"_"; it hits when every part equals
                      some top-k token. A single-part tag is identical under (a).
  (c) subword      — tokenize the tag WITH the SPARSEUP tokenizer (a leading
                      space, add_special_tokens=False; this is how the tag's
                      characters would actually be sliced if they appeared
                      mid-document, unlike (a)/(b)'s naive string splits) and
                      strip "Ġ" from every resulting piece; it hits when every
                      piece is among the top-k tokens. A single-piece tag is
                      identical to (a) tokenized this way, which is NOT always
                      the same string comparison as (a) itself (raw hyphens/
                      underscores are their own tokens under the real BPE
                      merge, not silently dropped the way (b)'s regex split
                      treats them).

Also: how many of the 325 labels are single-token in the SPARSEUP tokenizer at
all (bounds (a)), how many labels have 1/2/3/4+ subword pieces, and a
literal-mention rate per label (share of memories carrying that label whose
text contains the tag string literally, case insensitive, hyphens also
matching spaces).

Usage:
    python3 arm1_recall.py                 # full run; requires data/topk.json (encode done)
    python3 arm1_recall.py --limit 256      # smoke test: rebuilds a topk from whatever
                                             # data/parts/sparse_*.npz exist, first N rows only
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from common import DATA, load_fixture

SPARSE_MODEL = "Linkup-Platform/linkup-sparseup-embed-v1"
KS = (16, 32, 64)
TOPK_BUILD = 64  # matches encode.py's TOPK
SPLIT_RE = re.compile(r"[-_]+")
RESULTS = Path(__file__).resolve().parent / "results"


def strip_tok(tok: str) -> str:
    """Strip the BPE leading-space marker and lowercase, for token comparison."""
    return tok[1:].lower() if tok.startswith("Ġ") else tok.lower()


def build_topk_from_parts(n_limit: int):
    """Rebuild a topk-like structure from whatever data/parts/sparse_*.npz exist so
    far, for at most the first n_limit rows in fixture order. Mirrors encode.py's
    top-64 decode step (same vocab, same np.argsort(-row.data) ranking)."""
    parts_dir = DATA / "parts"
    part_files = sorted(parts_dir.glob("sparse_*.npz"))
    if not part_files:
        return None, 0
    mats = [sp.load_npz(p) for p in part_files]
    S = sp.vstack(mats).tocsr()
    n_avail = S.shape[0]
    n = min(n_limit, n_avail)
    S = S[:n]
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SPARSE_MODEL)
    vocab = {v: k for k, v in tok.get_vocab().items()}
    topk = []
    for r in range(S.shape[0]):
        row = S.getrow(r)
        order = np.argsort(-row.data)[:TOPK_BUILD]
        topk.append([[vocab[int(row.indices[j])], float(row.data[j])] for j in order])
    return topk, n


def load_topk(limit, fixture):
    ids_fixture = [m["id"] for m in fixture["memories"]]
    if limit:
        topk, n = build_topk_from_parts(limit)
        if topk is None:
            print(f"No data/parts/sparse_*.npz found; cannot smoke-test with --limit {limit}.",
                  file=sys.stderr)
            sys.exit(1)
        return ids_fixture[:n], topk, True
    tj = json.loads((DATA / "topk.json").read_text())
    assert tj["ids"] == ids_fixture, "topk.json id order does not match fixture.json order"
    return tj["ids"], tj["topk"], False


def load_tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(SPARSE_MODEL)


def label_subword_pieces(tok, labels):
    """{label: [stripped, lowercased pieces]} -- the tag tokenized WITH a
    leading space (add_special_tokens=False), i.e. the pieces it would
    actually decompose into inside a document, each with "Ġ" stripped. This
    is the (c) subword hit definition's piece set."""
    out = {}
    for lab in labels:
        ids = tok(" " + lab, add_special_tokens=False)["input_ids"]
        out[lab] = [strip_tok(t) for t in tok.convert_ids_to_tokens(ids)]
    return out


def single_token_bound(tok, labels):
    """How many of the 325 labels tokenize to exactly one token, bare or with a
    leading space (i.e. could in principle equal a single decoded top-k token)."""
    bare, leading, either = [], [], []
    for lab in labels:
        n_bare = len(tok(lab, add_special_tokens=False)["input_ids"])
        n_lead = len(tok(" " + lab, add_special_tokens=False)["input_ids"])
        if n_bare == 1:
            bare.append(lab)
        if n_lead == 1:
            leading.append(lab)
        if n_bare == 1 or n_lead == 1:
            either.append(lab)
    return {
        "n_labels": len(labels),
        "n_single_token_bare": len(bare),
        "n_single_token_leading_space": len(leading),
        "n_single_token_either": len(either),
        "labels_single_token_bare": sorted(bare),
        "labels_single_token_leading_space": sorted(leading),
        "labels_single_token_either": sorted(either),
    }


def literal_mention_rates(labels, id_to_labels, ids_subset, texts_by_id):
    """Per label: share of memories carrying that label whose text contains the
    tag string literally (case-insensitive; hyphens also matched as spaces)."""
    per_label_hits = defaultdict(int)
    per_label_n = defaultdict(int)
    for mid in ids_subset:
        labs = id_to_labels.get(mid) or []
        if not labs:
            continue
        text = (texts_by_id.get(mid) or "").lower()
        for lab in labs:
            per_label_n[lab] += 1
            lab_l = lab.lower()
            variants = {lab_l, lab_l.replace("-", " ")}
            if any(v in text for v in variants):
                per_label_hits[lab] += 1
    return {
        lab: (per_label_hits[lab] / per_label_n[lab] if per_label_n.get(lab) else None)
        for lab in labels
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None,
                     help="smoke-test on the first N fixture rows, rebuilding topk from data/parts")
    args = ap.parse_args()

    fixture = load_fixture()
    id_to_labels = {m["id"]: m["labels"] for m in fixture["memories"]}
    ids, topk, is_smoke = load_topk(args.limit, fixture)
    id_to_topk = dict(zip(ids, topk))
    labelled_ids = [i for i in ids if id_to_labels.get(i)]

    print(f"{'[smoke test] ' if is_smoke else ''}{len(ids)} rows available, "
          f"{len(labelled_ids)} labelled", file=sys.stderr)

    # tokenizer-derived pieces (independent of --limit; needs the tokenizer only)
    tok = load_tokenizer()
    stb = single_token_bound(tok, fixture["labels"])
    pieces_map = label_subword_pieces(tok, fixture["labels"])
    pieces_bucket = {"1": 0, "2": 0, "3": 0, "4+": 0}
    for pieces in pieces_map.values():
        key = str(len(pieces)) if len(pieces) < 4 else "4+"
        pieces_bucket[key] += 1

    HIT_DEFS = ("exact", "all_parts", "subword")

    # micro (over all label instances) and per-label recall, per hit-def, per k
    micro_hits = {hd: {k: 0 for k in KS} for hd in HIT_DEFS}
    micro_n = {k: 0 for k in KS}
    per_label_hits = {hd: {k: defaultdict(int) for k in KS} for hd in HIT_DEFS}
    per_label_n = {k: defaultdict(int) for k in KS}

    for mid in labelled_ids:
        row = id_to_topk[mid]
        # row is sorted by descending weight already (encode.py's np.argsort(-row.data));
        # slicing the first k preserves rank.
        toks_by_k = {k: {strip_tok(t) for t, _w in row[:k]} for k in KS}
        for label in id_to_labels[mid]:
            parts = [p for p in SPLIT_RE.split(label) if p] or [label]
            sw_pieces = pieces_map.get(label) or [label]
            for k in KS:
                toks_k = toks_by_k[k]
                hit_exact = label in toks_k
                hit_all = all(p in toks_k for p in parts)
                hit_subword = all(p in toks_k for p in sw_pieces)
                micro_n[k] += 1
                per_label_n[k][label] += 1
                if hit_exact:
                    micro_hits["exact"][k] += 1
                    per_label_hits["exact"][k][label] += 1
                if hit_all:
                    micro_hits["all_parts"][k] += 1
                    per_label_hits["all_parts"][k][label] += 1
                if hit_subword:
                    micro_hits["subword"][k] += 1
                    per_label_hits["subword"][k][label] += 1

    micro_recall = {
        hd: {str(k): (micro_hits[hd][k] / micro_n[k] if micro_n[k] else None) for k in KS}
        for hd in HIT_DEFS
    }

    all_labels = sorted({lab for labs in id_to_labels.values() for lab in (labs or [])} |
                         set(fixture.get("labels", [])))
    per_label_recall = {}
    for lab in all_labels:
        entry = {"n": per_label_n[KS[0]].get(lab, 0), "n_subword_pieces": len(pieces_map.get(lab, []))}
        for hd in HIT_DEFS:
            entry[hd] = {
                str(k): (per_label_hits[hd][k][lab] / per_label_n[k][lab]
                         if per_label_n[k].get(lab) else None)
                for k in KS
            }
        per_label_recall[lab] = entry

    # literal mention rate (from data/texts.json, always fully present)
    texts_j = json.loads((DATA / "texts.json").read_text())
    texts_by_id = dict(zip(texts_j["ids"], texts_j["texts"]))
    lit_rates = literal_mention_rates(fixture["labels"], id_to_labels, ids, texts_by_id)
    for lab, rate in lit_rates.items():
        per_label_recall.setdefault(lab, {"n": 0, "exact": {}, "all_parts": {}})
        per_label_recall[lab]["literal_mention_rate"] = rate

    # top/bottom 10 labels by subword recall@32, n >= 20
    K_SPOTLIGHT, N_MIN = 32, 20
    spotlight_candidates = [
        (lab, e["subword"][str(K_SPOTLIGHT)])
        for lab, e in per_label_recall.items()
        if e.get("n", 0) >= N_MIN and e["subword"].get(str(K_SPOTLIGHT)) is not None
    ]
    spotlight_lowest = sorted(spotlight_candidates, key=lambda t: t[1])[:10]
    spotlight_highest = sorted(spotlight_candidates, key=lambda t: -t[1])[:10]

    out = {
        "smoke_test": is_smoke,
        "n_rows_used": len(ids),
        "n_labelled_used": len(labelled_ids),
        "ks": list(KS),
        "micro_recall": micro_recall,
        "micro_n_instances": {str(k): micro_n[k] for k in KS},
        "single_token_bound": stb,
        "subword_pieces_distribution": pieces_bucket,
        "subword_recall_k32_spotlight": {
            "n_min": N_MIN, "k": K_SPOTLIGHT,
            "highest": spotlight_highest, "lowest": spotlight_lowest,
        },
        "per_label": per_label_recall,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "arm1.json").write_text(json.dumps(out, indent=1))

    print(f"\n{'SMOKE TEST — ' if is_smoke else ''}Arm 1: direct tag recall "
          f"({len(labelled_ids)} labelled memories, {len(ids)} rows total)")
    print(f"{'k':>4} {'exact-token':>12} {'all-parts':>12} {'subword':>12}")
    for k in KS:
        e = micro_recall["exact"][str(k)]
        a = micro_recall["all_parts"][str(k)]
        s = micro_recall["subword"][str(k)]
        if e is not None:
            print(f"{k:>4} {e:>12.4f} {a:>12.4f} {s:>12.4f}")
        else:
            print(f"{k:>4} {'n/a':>12} {'n/a':>12} {'n/a':>12}")
    print(f"\nSingle-token labels (bound on exact-token recall): "
          f"{stb['n_single_token_either']}/{stb['n_labels']} "
          f"(bare {stb['n_single_token_bare']}, leading-space {stb['n_single_token_leading_space']})")
    print(f"Subword pieces per label: 1={pieces_bucket['1']} 2={pieces_bucket['2']} "
          f"3={pieces_bucket['3']} 4+={pieces_bucket['4+']} (of {stb['n_labels']})")
    n_rated = sum(1 for v in lit_rates.values() if v is not None)
    mean_lit = (sum(v for v in lit_rates.values() if v is not None) / n_rated) if n_rated else None
    print(f"Mean literal-mention rate across {n_rated} labels: "
          f"{mean_lit:.4f}" if mean_lit is not None else "Mean literal-mention rate: n/a")
    print(f"\nHighest subword recall@{K_SPOTLIGHT} (n>={N_MIN}):")
    for lab, r in spotlight_highest:
        print(f"  {r:.4f}  {lab}")
    print(f"\nLowest subword recall@{K_SPOTLIGHT} (n>={N_MIN}):")
    for lab, r in spotlight_lowest:
        print(f"  {r:.4f}  {lab}")
    print(f"\nWrote {RESULTS / 'arm1.json'}")


if __name__ == "__main__":
    main()
