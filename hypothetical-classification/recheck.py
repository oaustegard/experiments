#!/usr/bin/env python3
"""recheck — re-derive every number in RESULTS.md from the saved artifacts.

Every input is on disk, including the 250 sampled memory rows: the Muninn corpus is live
and a seeded sample over a mutable population is not a fixture (ERRORS.md #6).

Runs in well under a minute and makes no API call: the model outputs are all on disk
(artifacts_lite.json, register_hall.json, haiku_arms.json, muninn_tags*_hall.json), so
only the snapping and scoring are recomputed. Fails loudly on any number that has drifted
from the prose.

    python3 recheck.py            # tfidf arms only, no download
    python3 recheck.py --minilm   # also the MiniLM arms (needs sentence-transformers)
"""
import argparse, json, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, "/home/user/muninn-utilities")

TOL = 0.001
failures, checked = [], 0


def check(name, got, want, tol=TOL):
    global checked
    checked += 1
    if got is None or abs(got - want) > tol:
        failures.append(f"{name}: RESULTS.md says {want:.3f}, recomputed {got!r}")


def snapper(labels, backend):
    from muninn_utils.hypothetical_classifier import Vocabulary
    return Vocabulary(labels, backend=backend)


def score_single(V, texts, gold_idx, labels):
    hits = V.snap(texts, k=3)
    idx = np.array([[labels.index(l) for l, _ in r] for r in hits])
    return (float(np.mean([g == p[0] for p, g in zip(idx, gold_idx)])),
            float(np.mean([g in p for p, g in zip(idx, gold_idx)])))


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--minilm", action="store_true")
    a = ap.parse_args(argv)
    backends = ["tfidf"] + (["minilm"] if a.minilm else [])

    A = json.load(open(HERE / "artifacts_lite.json"))
    vocab, queries, goldn = A["vocab"], A["queries"], A["gold"]
    gold = np.array([vocab.index(c) for c in goldn])
    reg = json.load(open(HERE / "register_hall.json"))
    assert len(reg) == len(queries) == 468, f"expected 468 queries, got {len(queries)}"
    assert len(vocab) == 860, f"expected 860 labels, got {len(vocab)}"

    WANDS = {  # (arm, backend) -> (acc@1, acc@3) as claimed in RESULTS.md
        ("query", "tfidf"): (0.316, 0.453), ("query", "minilm"): (0.417, 0.564),
        ("novelty", "tfidf"): (0.457, 0.556), ("novelty", "minilm"): (0.489, 0.613),
        ("register", "tfidf"): (0.528, 0.620), ("register", "minilm"): (0.564, 0.690),
    }
    texts = {"query": queries, "novelty": A["hall"], "register": reg}
    for bk in backends:
        V = snapper(vocab, bk)
        for arm in ("query", "novelty", "register"):
            a1, a3 = score_single(V, texts[arm], gold, vocab)
            w1, w3 = WANDS[(arm, bk)]
            check(f"WANDS {bk}/{arm} acc@1", a1, w1)
            check(f"WANDS {bk}/{arm} acc@3", a3, w3)

    # Haiku arms, first 40 queries
    H = json.load(open(HERE / "haiku_arms.json"))
    D = json.load(open(HERE / "h40.json"))
    g40 = np.array([vocab.index(c) for c in D["g"]])
    HAIKU = {("novelty", "tfidf"): (0.150, 0.225), ("novelty", "minilm"): (0.100, 0.275),
             ("register", "tfidf"): (0.400, 0.550), ("register", "minilm"): (0.525, 0.750)}
    for bk in backends:
        V = snapper(vocab, bk)
        for arm in ("novelty", "register"):
            assert len(H[arm]) == 40, f"haiku {arm}: expected 40 labels"
            a1, a3 = score_single(V, H[arm], g40, vocab)
            w1, w3 = HAIKU[(arm, bk)]
            check(f"Haiku {bk}/{arm} acc@1", a1, w1)
            check(f"Haiku {bk}/{arm} acc@3", a3, w3)


    # Tiny-model arms (n=40). Generated text is on disk; only snapping is recomputed.
    T = json.load(open(HERE / "tiny_arms.json"))
    TINY = {("Monad-fewshot", "tfidf"): (0.250, 0.375),
            ("Monad-fewshot", "minilm"): (0.425, 0.500),
            ("Baguettotron-fewshot", "tfidf"): (0.225, 0.325),
            ("Baguettotron-fewshot", "minilm"): (0.400, 0.475)}
    for bk in backends:
        V = snapper(vocab, bk)
        for arm in ("Monad-fewshot", "Baguettotron-fewshot"):
            assert len(T[arm]) == 40, f"{arm}: expected 40 labels"
            a1, a3 = score_single(V, T[arm], g40, vocab)
            w1, w3 = TINY[(arm, bk)]
            check(f"tiny {bk}/{arm} acc@1", a1, w1)
            check(f"tiny {bk}/{arm} acc@3", a3, w3)

    # Reranker and encoder tables are stored results, not re-derivable without the
    # models; assert the files agree with the prose instead.
    RR = json.load(open(HERE / "tiny_rerank_results.json"))
    for key, want in (("minilm/embedder acc@1", 0.500), ("minilm/ceiling recall@10", 0.825),
                      ("minilm/Monad rerank acc@1", 0.325),
                      ("minilm/Baguettotron rerank acc@1", 0.350),
                      ("tfidf/embedder acc@1", 0.275), ("tfidf/ceiling recall@10", 0.675)):
        check(f"rerank {key}", RR.get(key), want)
    BE = json.load(open(HERE / "browser_embedders.json"))
    for hf, want1, want3 in (("sentence-transformers/all-MiniLM-L6-v2", 0.417, 0.564),
                             ("thenlper/gte-small", 0.455, 0.594),
                             ("BAAI/bge-base-en-v1.5", 0.462, 0.630)):
        check(f"encoder {hf} acc@1", BE[hf]["query_acc1"], want1)
        check(f"encoder {hf} acc@3", BE[hf]["query_acc3"], want3)

    # Muninn tag arms — from the pinned fixture, never resampled (ERRORS.md #6)
    F = json.load(open(HERE / "muninn_tags_fixture.json"))
    labels = F["vocab"]
    S = [r["summary"] for r in F["rows"]]
    G = [set(r["gold"]) for r in F["rows"]]
    assert len(S) == 250, f"fixture has {len(S)} rows, expected 250"
    assert len(labels) == 1273, f"fixture vocab {len(labels)}, expected 1273"
    TAGS = {("control", "tfidf"): (0.416, 0.628, 0.712),
            ("novelty", "tfidf"): (0.208, 0.352, 0.424),
            ("register", "tfidf"): (0.508, 0.700, 0.792),
            ("union", "tfidf"): (0.672, 0.852, 0.888),
            ("control", "minilm"): (0.356, 0.584, 0.656),
            ("novelty", "minilm"): (0.244, 0.380, 0.460),
            ("register", "minilm"): (0.464, 0.676, 0.792),
            ("union", "minilm"): (0.640, 0.836, 0.888)}
    for bk in backends:
        V = snapper(labels, bk)
        ctl = V.snap(S, k=5)
        per = {}
        for arm in ("novelty", "register"):
            flat = [t for tags in F[arm] for t in tags]
            snapped = V.snap(flat, k=1)
            out, c = [], 0
            for tags in F[arm]:
                out.append([snapped[c + i][0][0] for i in range(len(tags))]); c += len(tags)
            per[arm] = out
        for k, i in ((1, 0), (3, 1), (5, 2)):
            check(f"tags {bk}/control @{k}",
                  float(np.mean([any(l in g for l, _ in r[:k]) for r, g in zip(ctl, G)])),
                  TAGS[("control", bk)][i])
            for arm in ("novelty", "register"):
                check(f"tags {bk}/{arm} @{k}",
                      float(np.mean([any(l in g for l in p[:k]) for p, g in zip(per[arm], G)])),
                      TAGS[(arm, bk)][i])
            check(f"tags {bk}/union @{k}",
                  float(np.mean([any(l in g for l, _ in r[:k]) or any(l in g for l in p[:k])
                                 for r, p, g in zip(ctl, per["register"], G)])),
                  TAGS[("union", bk)][i])

    # prose invariants that do not need recomputation
    txt = (HERE / "RESULTS.md").read_text()
    for needle in ("32,539", "860", "468", "1,273", "0.701", "5,265",
               "35 MB", "33 MB", "gte-small", "0.825"):
        if needle not in txt:
            failures.append(f"RESULTS.md no longer mentions {needle}")

    print(f"\nchecked {checked} numbers across backends {backends}")
    if failures:
        print(f"\n{len(failures)} DRIFTED:")
        for f in failures: print("  " + f)
        return 1
    print("all clear — RESULTS.md matches the artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
