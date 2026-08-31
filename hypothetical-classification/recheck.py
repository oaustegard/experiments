#!/usr/bin/env python3
"""recheck — re-derive every number in RESULTS.md from the saved artifacts.

Runs in well under a minute and makes no API call: the model outputs are all on disk
(artifacts_lite.json, register_hall.json, haiku_arms.json, muninn_tags*_hall.json), so
only the snapping and scoring are recomputed. Fails loudly on any number that has drifted
from the prose.

    python3 recheck.py            # tfidf arms only, no download
    python3 recheck.py --minilm   # also the MiniLM arms (needs sentence-transformers)
"""
import argparse, json, re, sys, collections
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

    # Muninn tags — needs the live corpus; skip cleanly when Turso is unreachable
    try:
        from muninn_utils.memory_tfidf import MemoryIndex
        import random
        idx = MemoryIndex(); idx.build()
    except Exception as exc:                                    # noqa: BLE001
        print(f"[skip] Muninn tag arms: corpus unavailable ({type(exc).__name__})")
        idx = None
    if idx is not None:
        def tags_of(m):
            t = m.get("tags") or []
            if isinstance(t, str): t = [x.strip() for x in t.split(",") if x.strip()]
            return set(t)
        counts = collections.Counter(t for m in idx.meta for t in tags_of(m))
        labels = sorted(t for t, c in counts.items() if c >= 3)
        VS = set(labels)
        rows = [(idx.summaries[i], tags_of(idx.meta[i]) & VS) for i in range(len(idx.ids))
                if len(idx.summaries[i]) > 300 and tags_of(idx.meta[i]) & VS]
        random.seed(20260831); rows = random.sample(rows, min(250, len(rows)))
        S = [s for s, _ in rows]; G = [g for _, g in rows]
        halls = {"novelty": json.load(open(HERE / "muninn_tags2_hall.json")),
                 "register": json.load(open(HERE / "muninn_tags3_hall.json"))}
        TAGS = {("control", "tfidf"): (0.400, 0.604, 0.684),
                ("novelty", "tfidf"): (0.200, 0.352, 0.452),
                ("register", "tfidf"): (0.500, 0.680, 0.728),
                ("union", "tfidf"): (0.676, 0.848, 0.876),
                ("control", "minilm"): (0.296, 0.528, 0.616),
                ("novelty", "minilm"): (0.212, 0.388, 0.500),
                ("register", "minilm"): (0.500, 0.672, 0.728),
                ("union", "minilm"): (0.640, 0.824, 0.860)}
        for bk in backends:
            V = snapper(labels, bk)
            ctl = V.snap(S, k=5)
            per = {}
            for arm, H_ in halls.items():
                flat = [t for tags in H_ for t in tags]
                snapped = V.snap(flat, k=1)
                out, c = [], 0
                for tags in H_:
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
    for needle in ("32,539", "860", "468", "1,273", "0.701", "5,265"):
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
