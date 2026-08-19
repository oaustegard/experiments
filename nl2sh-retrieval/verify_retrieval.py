#!/usr/bin/env python3
"""Adversarial verification of the nl2sh retrieval measurement.

WHY: `eval_retrieval.py` reports low recall (0.28@5 distinct-utility) and reads
that as "lexical retrieval over shell docs does not pick the right utility".
Before that becomes an architecture decision, three things have to be ruled out,
because each one moves the number in a different direction and two of them would
make the tier look *better* than it is:

  1. a wrong gold label (the eval could be scoring against noise),
  2. train/test leakage between tldr example descriptions and NL2Bash prompts
     (both are human-written English about the same commands; verbatim overlap
     would make the reported recall an over-estimate of held-out behaviour),
  3. a k that is large relative to the answer space (recall@20 over a corpus
     whose top-20 chunks span few distinct utilities means something different
     from recall@20 over 4,698 candidates), and no baseline to compare against.

Each check runs standalone (`--check gold`, `--check leak`, ...) and writes its
inputs, code pointer and verdict into results_verify.json. Nothing here is
copied from eval_retrieval's output: every number is recomputed.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from retrieve import Chunk, Index, load_chunks, tokens
import eval_retrieval as ev

HERE = Path(__file__).resolve().parent


def load(a):
    pairs = ev.load_pairs(a.nl2bash / "all.nl", a.nl2bash / "all.cm")
    buckets, freq = ev.bucket_map(pairs)
    sample = random.Random(a.seed).sample(pairs, min(a.sample, len(pairs)))
    return pairs, buckets, freq, sample


# --------------------------------------------------------------- check 1 ----
def check_gold(a, pairs, buckets, freq, sample):
    """Is `gold_utility` right? Spot-check + exhaustive scan for junk labels."""
    spot = random.Random(1234).sample(sample, 30)
    rows = [{"nl": nl[:90], "cm": cm[:110], "gold": g} for nl, cm, g in spot]

    # Exhaustive: which gold labels are not plausibly a utility name?
    junk = Counter()
    for _nl, cm, g in pairs:
        if g in ev.SHELL_CONSTRUCTS:
            junk["shell_construct:" + g] += 1
        elif not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.+-]*", g):
            junk["nonword:" + g] += 1
    # Second, independent parser: take the first token that names a file in a
    # PATH-like set OR is a known-utility-shaped word, WITHOUT the pipeline
    # split, to see whether stage-splitting changes the answer.
    def gold_naive(cm):
        for tok in cm.split():
            tok = tok.strip("()`$\\\"'")
            if not tok or tok in ev.WRAPPERS or tok.startswith("-"):
                continue
            if "=" in tok:
                continue
            return tok.rsplit("/", 1)[-1]
        return ""
    disagree = [(cm, g, gold_naive(cm)) for _n, cm, g in pairs if gold_naive(cm) != g]
    return {
        "spot_check_30": rows,
        "junk_gold_labels": dict(junk.most_common(20)),
        "junk_total": sum(junk.values()),
        "pairs": len(pairs),
        "naive_parser_disagreements": len(disagree),
        "naive_parser_disagreement_examples": [
            {"cm": c[:100], "staged": g, "naive": n} for c, g, n in disagree[:15]],
        "distinct_golds": len(freq),
    }


# --------------------------------------------------------------- check 2 ----
_WS = re.compile(r"[^a-z0-9]+")


def _norm(s):
    return [w for w in _WS.split(s.lower()) if w]


def check_leak(a, pairs, buckets, freq, sample):
    """Overlap between NL2Bash prompts and tldr example descriptions.

    tldr chunk text is "Description\\ncommand"; only the description line is
    English prose that could have been written from the same upstream source as
    an NL2Bash prompt. Measured three ways: exact normalised equality, best
    token-Jaccard per query, and — the number that matters — recall conditioned
    on how well the query matches its *gold utility's* best description.
    """
    chunks = load_chunks(a.chunks)
    descs, desc_owner = [], []
    for c in chunks:
        if c.kind != "tldr_example":
            continue
        d = c.text.split("\n", 1)[0]
        descs.append(set(_norm(d)))
        desc_owner.append(c.utility)
    exact = {" ".join(sorted(s)) for s in descs}

    idx = Index.load(a.chunks)
    by_util = defaultdict(list)
    for s, u in zip(descs, desc_owner):
        by_util[u].append(s)

    n_exact = 0
    jac_all, rows = [], []
    for nl, _cm, g in sample:
        q = set(_norm(nl))
        if " ".join(sorted(q)) in exact:
            n_exact += 1
        # best jaccard against the gold utility's own descriptions (the only
        # ones whose similarity could inflate recall for this query)
        best = 0.0
        for s in by_util.get(g, ()):
            inter = len(q & s)
            if not inter:
                continue
            j = inter / len(q | s)
            if j > best:
                best = j
        jac_all.append(best)
        ranked = idx.rank_utilities(nl, k=20, pool=ev.POOL)
        rows.append((best, g in ranked[:5], g in ranked[:20]))

    def band(lo, hi):
        sel = [r for r in rows if lo <= r[0] < hi]
        if not sel:
            return None
        return {"n": len(sel),
                "recall@5": round(sum(r[1] for r in sel) / len(sel), 4),
                "recall@20": round(sum(r[2] for r in sel) / len(sel), 4)}

    bands = {"0.0-0.2": band(0.0, 0.2), "0.2-0.4": band(0.2, 0.4),
             "0.4-0.6": band(0.4, 0.6), "0.6-1.01": band(0.6, 1.01)}
    return {
        "tldr_descriptions": len(descs),
        "queries": len(sample),
        "exact_normalised_matches": n_exact,
        "best_jaccard_vs_gold_utility_descriptions": {
            "median": round(statistics.median(jac_all), 4),
            "mean": round(statistics.fmean(jac_all), 4),
            "p90": round(sorted(jac_all)[int(0.9 * (len(jac_all) - 1))], 4),
            "frac_ge_0.5": round(sum(j >= 0.5 for j in jac_all) / len(jac_all), 4),
            "frac_ge_0.3": round(sum(j >= 0.3 for j in jac_all) / len(jac_all), 4),
        },
        "recall_by_jaccard_band": bands,
    }


# --------------------------------------------------------------- check 3 ----
def check_k(a, pairs, buckets, freq, sample):
    """How big is the answer space that top-k actually spans, and what does
    chance score at the same k?"""
    idx = Index.load(a.chunks)
    n_utils = len(set(idx.utilities.tolist()))
    spans = {1: [], 5: [], 10: [], 20: []}
    zero_hits = 0
    lat_topk, lat_rank = [], []
    for nl, _cm, _g in sample:
        t0 = time.perf_counter()
        ids = idx.topk(nl, 20)
        lat_topk.append((time.perf_counter() - t0) * 1000)
        t0 = time.perf_counter()
        idx.rank_utilities(nl, k=20, pool=ev.POOL)
        lat_rank.append((time.perf_counter() - t0) * 1000)
        if not ids:
            zero_hits += 1
        us = [idx.chunks[i].utility for i in ids]
        for k in spans:
            spans[k].append(len(set(us[:k])))

    rng = random.Random(7)
    util_arr = list(idx.utilities.tolist())
    # baseline A: k chunks drawn uniformly at random (chunk-count weighted)
    # baseline B: k utilities drawn uniformly at random (unweighted)
    # baseline C: constant list = the k most frequent NL2Bash gold utilities
    #             (a predictor that ignores the query entirely)
    uniq_utils = sorted(set(util_arr))
    const = [u for u, _ in freq.most_common(20)]
    base = {}
    for k in (1, 5, 10, 20):
        ra = rb = rc = 0
        for _nl, _cm, g in sample:
            if g in {rng.choice(util_arr) for _ in range(k)}:
                ra += 1
            if g in {rng.choice(uniq_utils) for _ in range(k)}:
                rb += 1
            if g in const[:k]:
                rc += 1
        n = len(sample)
        base[k] = {"random_chunk": round(ra / n, 4),
                   "random_utility": round(rb / n, 4),
                   "constant_top_frequency": round(rc / n, 4)}
    lat_topk.sort(); lat_rank.sort()
    return {
        "corpus_utilities": n_utils,
        "distinct_utilities_in_topk_chunks": {
            str(k): {"mean": round(statistics.fmean(v), 2),
                     "median": statistics.median(v)} for k, v in spans.items()},
        "queries_with_zero_scoring_chunks": zero_hits,
        "baselines_recall_at_k": {str(k): v for k, v in base.items()},
        "latency_ms": {
            "topk_only_median": round(statistics.median(lat_topk), 3),
            "rank_utilities_pool300_median": round(statistics.median(lat_rank), 3),
            "rank_utilities_pool300_p90": round(lat_rank[int(0.9 * (len(lat_rank) - 1))], 3),
        },
    }


# --------------------------------------------------------------- check 4 ----
def _wilson(k, n, z=1.96):
    if n == 0:
        return [0.0, 0.0]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / d
    return [round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)]


def check_split(a, pairs, buckets, freq, sample):
    """Head/tail disjointness, and whether the tail samples mean anything."""
    head = {u for u, b in buckets.items() if b == "head"}
    tail = {u for u, b in buckets.items() if b == "tail"}
    mid = {u for u, b in buckets.items() if b == "mid"}
    tail_pairs = [p for p in pairs if buckets.get(p[2]) == "tail"]
    tail_sample = tail_pairs if len(tail_pairs) <= a.sample else \
        random.Random(a.seed + 1).sample(tail_pairs, a.sample)
    idx = Index.load(a.chunks)

    # recompute tail recall so the CI is on a number we produced
    hits = {1: 0, 5: 0, 10: 0, 20: 0}
    per_util = defaultdict(lambda: [0, 0])
    for nl, _cm, g in tail_sample:
        r = idx.rank_utilities(nl, k=20, pool=ev.POOL)
        per_util[g][1] += 1
        for k in hits:
            if g in r[:k]:
                hits[k] += 1
                if k == 5:
                    per_util[g][0] += 1
    n = len(tail_sample)
    main_tail = [p for p in sample if buckets.get(p[2]) == "tail"]
    return {
        "head_tail_disjoint": len(head & tail) == 0 and len(head & mid) == 0
                              and len(tail & mid) == 0,
        "utilities": {"head": len(head), "mid": len(mid), "tail": len(tail)},
        "tail_pairs_total": len(tail_pairs),
        "tail_sample_n_pairs": n,
        "tail_sample_distinct_utilities": len(per_util),
        "tail_recall_at_k_recomputed": {str(k): round(v / n, 4) for k, v in hits.items()},
        "tail_recall_at_k_wilson95": {str(k): _wilson(v, n) for k, v in hits.items()},
        "main_sample_tail_n": len(main_tail),
        "main_sample_tail_wilson95_at5": _wilson(
            sum(1 for nl, _c, g in main_tail
                if g in idx.rank_utilities(nl, k=5, pool=ev.POOL)), len(main_tail)),
        "note_effective_n": "per-utility recall@5 varies; the independent unit "
                            "for a claim about 'the tail' is the utility, not the pair",
        "tail_utilities_with_zero_hits_at5": sum(1 for v in per_util.values() if v[0] == 0),
    }


# --------------------------------------------------------------- check 5 ----
def check_other(a, pairs, buckets, freq, sample):
    """Everything else that moves the number: duplicate prompts, the
    leading-utility-only gold, and gold/corpus name mismatch."""
    idx = Index.load(a.chunks)
    corpus = set(idx.utilities.tolist())

    # (a) duplicate NL prompts in the sample / corpus
    nls = Counter(nl.strip().lower() for nl, _c, _g in pairs)
    dup_pairs = sum(c for c in nls.values() if c > 1)

    # (b) leading-utility-only gold understates: score "any utility named in the
    #     command" as an alternative target
    def all_utils(cm):
        out = set()
        for stage in ev._SPLIT.split(ev._mask(cm)):
            g = ev.gold_utility(stage)
            if g:
                out.add(g)
        return out

    lead_hits = {5: 0, 20: 0}
    any_hits = {5: 0, 20: 0}
    for nl, cm, g in sample:
        r = idx.rank_utilities(nl, k=20, pool=ev.POOL)
        au = all_utils(cm) & corpus
        for k in (5, 20):
            if g in r[:k]:
                lead_hits[k] += 1
            if au and (au & set(r[:k])):
                any_hits[k] += 1
    n = len(sample)

    # (c) gold names absent from the corpus — real coverage gap or naming skew?
    missing = Counter(g for _n, _c, g in pairs if g not in corpus)

    # (d) how much of the corpus is not shell-utility documentation at all
    #     (tldr covers many non-POSIX CLIs); measure the share of chunks whose
    #     utility never appears in NL2Bash.
    nl2bash_utils = set(freq)
    off_topic = sum(1 for c in idx.chunks if c.utility not in nl2bash_utils)
    return {
        "duplicate_nl_prompts_pairs": dup_pairs,
        "duplicate_nl_prompts_distinct": sum(1 for c in nls.values() if c > 1),
        "recall_leading_utility": {str(k): round(v / n, 4) for k, v in lead_hits.items()},
        "recall_any_utility_in_command": {str(k): round(v / n, 4) for k, v in any_hits.items()},
        "gold_utilities_missing_from_corpus": len(missing),
        "gold_pairs_missing_from_corpus": sum(missing.values()),
        "top_missing": dict(missing.most_common(15)),
        "chunks_whose_utility_never_appears_in_nl2bash": off_topic,
        "chunk_share_off_topic": round(off_topic / idx.n, 4),
        "corpus_utilities": len(corpus),
        "nl2bash_utilities": len(nl2bash_utils),
    }



# --------------------------------------------------------------- check 6 ----
_TOKSPLIT = re.compile(r"[^A-Za-z0-9_.+-]+")


def check_hint(a, pairs, buckets, freq, sample):
    """Does the prompt already name the utility, and does that carry recall?

    The tier's premise is that the user does NOT know which utility to reach
    for. NL2Bash prompts were written by annotators looking at the command, so
    many name it outright ("Convert *.au files ... using `sox`"). Recall
    measured over those queries is measuring string matching, not the operation
    the tier is supposed to perform, and it flatters the tier.
    """
    idx = Index.load(a.chunks)
    named, unnamed = [], []
    for nl, _cm, g in sample:
        toks = {t.lower() for t in _TOKSPLIT.split(nl) if t}
        (named if g.lower() in toks else unnamed).append((nl, g))

    def score(rows):
        if not rows:
            return None
        h = {1: 0, 5: 0, 20: 0}
        for nl, g in rows:
            r = idx.rank_utilities(nl, k=20, pool=ev.POOL)
            for k in h:
                if g in r[:k]:
                    h[k] += 1
        return {"n": len(rows), **{f"recall@{k}": round(v / len(rows), 4) for k, v in h.items()}}

    # The deployment slice: non-find AND the prompt does not hand over the answer.
    tail_pairs = [p for p in pairs if buckets.get(p[2]) == "tail"]

    def named_p(p):
        return p[2].lower() in {t.lower() for t in _TOKSPLIT.split(p[0]) if t}

    return {
        "prompt_names_gold_utility": score(named),
        "prompt_does_not_name_it": score(unnamed),
        "share_naming_gold": round(len(named) / len(sample), 4),
        "slices": {
            "non_find_named": score([(p[0], p[2]) for p in sample
                                     if p[2] != "find" and named_p(p)]),
            "non_find_unnamed": score([(p[0], p[2]) for p in sample
                                       if p[2] != "find" and not named_p(p)]),
            "tail_named": score([(p[0], p[2]) for p in tail_pairs if named_p(p)]),
            "tail_unnamed": score([(p[0], p[2]) for p in tail_pairs if not named_p(p)]),
        },
    }


# --------------------------------------------------------------- check 7 ----
def check_scoped(a, pairs, buckets, freq, sample):
    """Ceiling if the corpus contained only utilities the eval can ask about.

    90.9% of the chunks belong to utilities NL2Bash never mentions (tldr is
    thick with git subcommands, aws, kubectl, npm; NL2Bash is 2016 POSIX shell).
    Restricting the corpus to the 356 gold utilities is an oracle and NOT a
    deployable configuration - it is the upper bound on how much of the miss
    rate is 'wrong candidate pool' rather than 'wrong ranking'.
    """
    keep = set(freq)
    chunks = [c for c in load_chunks(a.chunks) if c.utility in keep]
    idx = Index(chunks)
    tail_pairs = [p for p in pairs if buckets.get(p[2]) == "tail"]
    out = {"chunks": idx.n, "utilities": len(set(idx.utilities.tolist()))}
    for name, rows in (("sample_600", sample), ("tail_all", tail_pairs)):
        h = {1: 0, 5: 0, 20: 0}
        hn = {1: 0, 5: 0, 20: 0}
        nf = 0
        for nl, _cm, g in rows:
            r = idx.rank_utilities(nl, k=20, pool=ev.POOL)
            isnf = g != "find"
            nf += isnf
            for k in h:
                if g in r[:k]:
                    h[k] += 1
                    if isnf:
                        hn[k] += 1
        out[name] = {"n": len(rows),
                     "recall": {str(k): round(v / len(rows), 4) for k, v in h.items()},
                     "recall_non_find": {str(k): round(v / nf, 4) for k, v in hn.items()} if nf else None}
    return out


# --------------------------------------------------------------- check 8 ----
def check_baseline_slices(a, pairs, buckets, freq, sample):
    """The constant-frequency-prior baseline, per slice, prior fitted out-of-sample.

    check_k fitted the prior on all of NL2Bash including the eval sample, which
    is an oracle. Here the prior comes from the pairs NOT in the sample, so it
    is an honest query-independent predictor. Reported per slice because the
    prior is strong exactly where BM25 does not need to help (head) and empty
    where it does (tail).
    """
    idx = Index.load(a.chunks)
    ids = {id(p) for p in sample}
    held = [p for p in pairs if id(p) not in ids]
    prior = [u for u, _ in Counter(g for _n, _c, g in held).most_common(20)]
    slices = {"all": lambda g: True, "non_find": lambda g: g != "find",
              "head": lambda g: buckets.get(g) == "head",
              "tail": lambda g: buckets.get(g) == "tail"}
    # tail is evaluated on all 369 of its pairs, not the 18 that landed in the sample
    tail_pairs = [p for p in pairs if buckets.get(p[2]) == "tail"]
    cache = {}
    out = {"prior_top20": prior, "prior_fitted_on_pairs": len(held)}
    for sl, keep in slices.items():
        sel = [p for p in (tail_pairs if sl == "tail" else sample) if keep(p[2])]
        if not sel:
            continue
        b = {1: 0, 5: 0, 20: 0}
        m = {1: 0, 5: 0, 20: 0}
        for nl, _cm, g in sel:
            r = cache.get(nl)
            if r is None:
                r = cache[nl] = idx.rank_utilities(nl, k=20, pool=ev.POOL)
            for k in b:
                if g in prior[:k]:
                    b[k] += 1
                if g in r[:k]:
                    m[k] += 1
        n = len(sel)
        out[sl] = {"n": n,
                   "bm25": {str(k): round(v / n, 4) for k, v in m.items()},
                   "constant_prior": {str(k): round(v / n, 4) for k, v in b.items()}}
    return out


CHECKS = {"gold": check_gold, "leak": check_leak, "k": check_k,
          "split": check_split, "other": check_other, "hint": check_hint,
          "scoped": check_scoped, "baseline_slices": check_baseline_slices}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=Path, default=HERE / "data" / "chunks.jsonl")
    ap.add_argument("--nl2bash", type=Path, default=HERE / "data" / "nl2bash")
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--check", action="append", choices=list(CHECKS) + ["all"],
                    default=None)
    ap.add_argument("--out", type=Path, default=HERE / "verify_parts")
    args = ap.parse_args()
    which = args.check or ["all"]
    if "all" in which:
        which = list(CHECKS)
    pairs, buckets, freq, sample = load(args)
    args.out.mkdir(exist_ok=True)
    for name in which:
        t0 = time.perf_counter()
        r = CHECKS[name](args, pairs, buckets, freq, sample)
        r["_seconds"] = round(time.perf_counter() - t0, 2)
        (args.out / f"{name}.json").write_text(json.dumps(r, indent=2) + "\n")
        print(f"[{name}] {time.perf_counter()-t0:.1f}s -> {args.out/name}.json")
        print(json.dumps(r, indent=2)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
