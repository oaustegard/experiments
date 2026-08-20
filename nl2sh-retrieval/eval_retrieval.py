#!/usr/bin/env python3
"""Does lexical retrieval over shell docs actually pick the right utility?

This is the measurement that decides the architecture. `nl2sh-scoping` found a
thin head and a long tail (377 distinct leading utilities, 176 used exactly
once), which argues for a tier that *narrows candidates* rather than one that
answers. That argument is only worth anything if the narrowing works — and
specifically if it works on the tail, because the tail is where a small model
needs the help and where tldr coverage falls to ~50%.

Ground truth is NL2Bash: for each (nl, cm) pair the gold label is the leading
utility of the command. That is a weaker target than "produce the command", on
purpose — the retrieval tier is not being asked to write anything, only to put
the right utility in front of the model.

Reported:
  * recall@{1,5,10,20}: is the gold utility among the utilities of the top-k
    *chunks*. Because one utility often owns several top chunks, a
    distinct-utility variant (top-k utilities from a deeper chunk pool) is
    reported alongside — the two answer different questions and the second is
    the one a shortlist of size k actually delivers.
  * The same, split head (top-50 utilities by NL2Bash usage) vs tail (used <= 5
    times). The tail number is the finding.
  * A ceiling: the fraction of sampled gold utilities that have any chunk in the
    corpus at all. Recall cannot exceed it, and reporting recall without it
    hides whether a miss was retrieval's fault or coverage's.
  * Median query latency.
  * Ablation on chunk source: tldr only vs man only vs both, at k=5. The
    scoping README predicts man pages carry the tail. NOTE: only 60 man pages
    are installed on this container, so the man arm here measures the *parser
    and the ranker*, not the coverage claim.

    python3 eval_retrieval.py --nl2bash <dir with all.nl/all.cm>
"""
from __future__ import annotations

import argparse
import json
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from retrieve import Chunk, Index, load_chunks

HERE = Path(__file__).resolve().parent

KS = (1, 5, 10, 20)
POOL = 300  # chunk pool depth for the distinct-utility ranking

# Wrappers that prefix a real command without being one.
WRAPPERS = {"sudo", "time", "nohup", "command", "env", "exec", "xargs", "!",
            "\\time", "builtin"}
# Shell grammar that can lead a command line. These are not utilities and no
# documentation corpus is keyed by them; they are counted and reported
# separately rather than silently dropped.
SHELL_CONSTRUCTS = {"for", "while", "until", "if", "case", "do", "done", "then",
                    "else", "fi", "esac", "function", "{", "(", "[", "[[",
                    "select", "trap", "return", "break", "continue", "in"}
_SPLIT = re.compile(r"\|\||&&|\||;")


def _mask(cmd: str) -> str:
    """Blank quoted spans so pipeline separators inside strings do not split."""
    return re.sub(r'"[^"]*"|\'[^\']*\'', '""', cmd)


def gold_utility(cmd: str) -> str:
    """Leading utility of the first pipeline stage, wrappers stripped.

    `sudo find / -name x | head` -> find
    `/usr/bin/awk '{print}'`     -> awk   (path prefix dropped)
    `LANG=C sort f`              -> sort  (VAR=value prefix skipped)
    """
    stages = [s.strip() for s in _SPLIT.split(_mask(cmd)) if s.strip()]
    if not stages:
        return ""
    for tok in stages[0].split():
        tok = tok.strip("()`$\\")
        if not tok:
            continue
        if tok in WRAPPERS:
            continue
        if "=" in tok and not tok.startswith("-"):
            continue  # VAR=value prefix
        if tok.startswith("-"):
            continue
        if "/" in tok:
            tok = tok.rsplit("/", 1)[-1]  # /usr/bin/awk -> awk
        return tok
    return ""


def load_pairs(nl_path: Path, cm_path: Path) -> list[tuple[str, str, str]]:
    nls = nl_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cms = cm_path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(nls) != len(cms):
        raise SystemExit(f"misaligned corpus: {len(nls)} nl vs {len(cms)} cm")
    out = []
    for nl, cm in zip(nls, cms):
        nl, cm = nl.strip(), cm.strip()
        if not nl or not cm:
            continue
        g = gold_utility(cm)
        if g:
            out.append((nl, cm, g))
    return out


def bucket_map(pairs, head_n: int = 50, tail_max: int = 5) -> tuple[dict[str, str], Counter]:
    """Label every utility head / mid / tail by its NL2Bash usage frequency."""
    freq = Counter(g for _, _, g in pairs)
    head = {u for u, _ in freq.most_common(head_n)}
    out = {}
    for u, c in freq.items():
        out[u] = "head" if u in head else ("tail" if c <= tail_max else "mid")
    return out, freq


def evaluate(idx: Index, sample, buckets: dict[str, str], ks=KS,
             measure_latency: bool = False) -> dict:
    """Recall@k over a sample of (nl, cm, gold) triples.

    Each pair contributes to several overlapping slices: `all`, its frequency
    bucket (head/mid/tail), and `find` vs `non_find`. The last split is not
    optional — 60.3% of NL2Bash leads with `find`, so an undifferentiated `all`
    number is mostly a measurement of one utility (nl2sh-scoping README).
    """
    corpus_utils = set(idx.utilities.tolist())
    hits_chunk = {k: Counter() for k in ks}       # slice -> hits
    hits_util = {k: Counter() for k in ks}
    hits_cov = {k: Counter() for k in ks}   # hits restricted to covered golds
    totals: Counter = Counter()
    in_corpus: Counter = Counter()
    lat: list[float] = []
    kmax = max(ks)
    for nl, _cm, gold in sample:
        slices = ["all", buckets.get(gold, "mid"), "find" if gold == "find" else "non_find"]
        for sl in slices:
            totals[sl] += 1
            if gold in corpus_utils:
                in_corpus[sl] += 1
        t0 = time.perf_counter()
        chunk_ids = idx.topk(nl, kmax)
        if measure_latency:
            lat.append((time.perf_counter() - t0) * 1000.0)
        chunk_utils = [idx.chunks[i].utility for i in chunk_ids]
        # distinct-utility ranking from a deeper chunk pool
        ranked = idx.rank_utilities(nl, k=kmax, pool=POOL)
        covered = gold in corpus_utils
        for k in ks:
            in_chunks = gold in set(chunk_utils[:k])
            in_ranked = gold in ranked[:k]
            for sl in slices:
                if in_chunks:
                    hits_chunk[k][sl] += 1
                if in_ranked:
                    hits_util[k][sl] += 1
                    if covered:
                        hits_cov[k][sl] += 1

    order = ("all", "non_find", "find", "head", "mid", "tail")

    def table(h, denom=None):
        d = denom or totals
        return {sl: {str(k): round(h[k][sl] / d[sl], 4) for k in ks}
                for sl in order if d[sl]}

    out = {
        "n": totals["all"],
        "n_by_slice": {sl: totals[sl] for sl in order if totals[sl]},
        "gold_in_corpus": {sl: round(in_corpus[sl] / totals[sl], 4)
                           for sl in order if totals[sl]},
        "recall_at_k_chunks": table(hits_chunk),
        "recall_at_k_utilities": table(hits_util),
        # Recall among only those pairs whose gold utility has a chunk at all —
        # separates a ranking failure from a coverage failure.
        "recall_at_k_utilities_given_covered": table(hits_cov, in_corpus),
    }
    if measure_latency and lat:
        lat.sort()
        out["latency_ms"] = {
            "median": round(statistics.median(lat), 3),
            "p90": round(lat[int(0.9 * (len(lat) - 1))], 3),
            "mean": round(statistics.fmean(lat), 3),
        }
    return out


def utility_documents(chunks) -> list[Chunk]:
    """One document per utility: all of its chunks concatenated.

    Chunk-level retrieval asks "which example matches"; this asks "which
    utility's documentation matches", which is the question the tier actually
    has to answer. It also removes BM25's length normalisation from a corpus of
    ~19-token chunks, where a single shared word is a large fraction of a
    document.
    """
    agg: dict[str, list[str]] = defaultdict(list)
    for c in chunks:
        agg[c.utility].append(c.text)
    return [Chunk(id=f"utility:{u}", utility=u, kind="utility_page",
                  text="\n".join(t), runnable=False)
            for u, t in agg.items()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chunks", type=Path, default=HERE / "data" / "chunks.jsonl")
    ap.add_argument("--nl2bash", type=Path, default=HERE / "data" / "nl2bash",
                    help="directory containing all.nl and all.cm")
    ap.add_argument("--sample", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=HERE / "results_retrieval.json")
    a = ap.parse_args()

    if not (a.nl2bash / "all.nl").exists():
        raise SystemExit(
            f"no NL2Bash corpus at {a.nl2bash}. Clone it and point --nl2bash at "
            "its data/bash directory:\n"
            "  git clone --depth 1 https://github.com/TellinaTool/nl2bash.git\n"
            "  python3 eval_retrieval.py --nl2bash nl2bash/data/bash")
    pairs = load_pairs(a.nl2bash / "all.nl", a.nl2bash / "all.cm")
    buckets, freq = bucket_map(pairs)
    rng = random.Random(a.seed)
    sample = rng.sample(pairs, min(a.sample, len(pairs)))
    sample_no_shell = [p for p in sample if p[2] not in SHELL_CONSTRUCTS]

    t0 = time.perf_counter()
    idx = Index.load(a.chunks)
    build_s = time.perf_counter() - t0
    print(f"index: {idx.n} chunks, {len(idx.postings)} terms, built in {build_s:.2f}s")

    res = {
        "corpus": {
            "chunks": idx.n,
            "terms": len(idx.postings),
            "utilities": len(set(idx.utilities.tolist())),
            "build_seconds": round(build_s, 2),
            "kind_counts": dict(Counter(c.kind for c in idx.chunks)),
        },
        "eval": {
            "source": "NL2Bash all.nl / all.cm",
            "pairs_total": len(pairs),
            "sample": len(sample),
            "seed": a.seed,
            "distinct_gold_utilities": len(freq),
            "head_definition": "top 50 utilities by NL2Bash leading-utility count",
            "tail_definition": "utilities with <= 5 occurrences",
            "shell_construct_golds_in_sample": len(sample) - len(sample_no_shell),
        },
        "main": evaluate(idx, sample, buckets, measure_latency=True),
        "excluding_shell_constructs": evaluate(idx, sample_no_shell, buckets),
    }

    # --- stratified tail: n=18 in a random 600 is not a measurement --------- #
    tail_pairs = [p for p in pairs if buckets.get(p[2]) == "tail"]
    tail_sample = tail_pairs if len(tail_pairs) <= a.sample else \
        random.Random(a.seed + 1).sample(tail_pairs, a.sample)
    res["tail_stratified"] = {
        "note": "every NL2Bash pair whose gold utility occurs <= 5 times "
                "(capped at --sample), evaluated on the main index",
        "pairs_available": len(tail_pairs),
        **evaluate(idx, tail_sample, buckets),
    }

    # --- ablation: chunk source -------------------------------------------- #
    arms = {
        "tldr_only": {"tldr_example"},
        "man_only": {"man_option", "man_example"},
        "both": None,
    }
    abl = {}
    for name, kinds in arms.items():
        chunks = load_chunks(a.chunks, kinds)
        if not chunks:
            abl[name] = {"error": "no chunks"}
            continue
        sub = Index(chunks)
        r = evaluate(sub, sample, buckets, ks=(5,))
        rt = evaluate(sub, tail_sample, buckets, ks=(5,))
        abl[name] = {
            "chunks": sub.n,
            "utilities": len(set(sub.utilities.tolist())),
            "gold_in_corpus": r["gold_in_corpus"],
            "recall_at_5_chunks": {b: v["5"] for b, v in r["recall_at_k_chunks"].items()},
            "recall_at_5_utilities": {b: v["5"] for b, v in r["recall_at_k_utilities"].items()},
            "tail_stratified": {
                "n": rt["n"],
                "gold_in_corpus": rt["gold_in_corpus"]["all"],
                "recall_at_5_utilities": rt["recall_at_k_utilities"]["all"]["5"],
                "recall_at_5_given_covered":
                    rt["recall_at_k_utilities_given_covered"]["all"]["5"]
                    if "all" in rt["recall_at_k_utilities_given_covered"] else None,
            },
        }
    res["ablation_source"] = abl

    # --- ablation: chunk-level vs utility-level documents ------------------- #
    upage = Index(utility_documents(idx.chunks))
    res["ablation_granularity"] = {
        "note": "one document per utility (all its chunks concatenated)",
        "documents": upage.n,
        "utility_level": evaluate(upage, sample, buckets),
    }

    # --- ablation: light stemming ------------------------------------------ #
    stemmed = Index.load(a.chunks, stem=True)
    res["ablation_stemming"] = {
        "note": "light suffix stripper, not Porter; see retrieve._stem",
        "stemmed": evaluate(stemmed, sample, buckets),
    }

    a.out.write_text(json.dumps(res, indent=2) + "\n")

    # --- console report ----------------------------------------------------- #
    m = res["main"]
    print(f"\nsample {m['n']} pairs  slices {m['n_by_slice']}")
    print(f"gold utility present in corpus: {m['gold_in_corpus']}")
    for label, key in (("chunk-membership", "recall_at_k_chunks"),
                       ("distinct-utility", "recall_at_k_utilities")):
        print(f"\nrecall@k ({label})")
        print(f"  {'bucket':<8}" + "".join(f"{'@'+str(k):>9}" for k in KS))
        for b, row in m[key].items():
            print(f"  {b:<8}" + "".join(f"{row[str(k)]:>9.3f}" for k in KS))
    print(f"\nlatency ms: {m['latency_ms']}")
    t = res["tail_stratified"]
    print(f"\ntail-stratified (n={t['n']} of {t['pairs_available']} available), "
          f"gold in corpus {t['gold_in_corpus']['all']:.3f}")
    tu, tc = t["recall_at_k_utilities"]["all"], t["recall_at_k_utilities_given_covered"]["all"]
    print("  distinct-utility  " + "  ".join(f"@{k}={tu[str(k)]:.3f}" for k in KS))
    print("  given covered     " + "  ".join(f"@{k}={tc[str(k)]:.3f}" for k in KS))
    print("\nablation @5 (chunk-membership / distinct-utility)")
    for name, v in abl.items():
        if "error" in v:
            print(f"  {name}: {v['error']}")
            continue
        c, u, t = v["recall_at_5_chunks"], v["recall_at_5_utilities"], v["tail_stratified"]
        print(f"  {name:<10} chunks={v['chunks']:<6} utils={v['utilities']:<5} "
              f"cover={v['gold_in_corpus']['all']:.3f} all={c['all']:.3f}/{u['all']:.3f} "
              f"| tail n={t['n']} cover={t['gold_in_corpus']:.3f} "
              f"r@5={t['recall_at_5_utilities']:.3f}")
    print("\nother arms (distinct-utility recall@k, all / non_find / tail)")
    arms_out = {
        "chunk-level (main)": m["recall_at_k_utilities"],
        "utility-level": res["ablation_granularity"]["utility_level"]["recall_at_k_utilities"],
        "stemmed": res["ablation_stemming"]["stemmed"]["recall_at_k_utilities"],
    }
    for name, tbl in arms_out.items():
        cells = "  ".join(
            f"{sl}@{k}={tbl[sl][str(k)]:.3f}"
            for sl in ("all", "non_find", "tail") if sl in tbl for k in (5, 20))
        print(f"  {name:<20} {cells}")
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
