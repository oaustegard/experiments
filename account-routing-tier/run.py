"""Can a small always-loaded summary index route a query to the right repo?

A whole-account index partitioned per repo only works if a client can decide
*which* partitions to fetch without downloading them all. That means a coarse
tier: one small index of per-repo summaries, always loaded, that picks the top-k
repos; the fine partitions for those repos are then fetched on demand.

The failure mode is unforgiving. If the coarse tier picks wrong, the answer is
unreachable no matter how good the partition is — unlike a flat index, there is
no graceful degradation, just a confident wrong answer from the wrong repo.

## Design

**Oracle:** a flat search over every chunk of every repo. Whichever repo owns
the top-scoring chunk is the repo the query "wanted". This needs no hand
labelling, which matters because hand-labelled gold has broken three times in
this line of work already.

**Routing recall@k:** how often the oracle's repo appears in the coarse tier's
top-k. That is the number that decides whether partitioning is viable.

**Queries deliberately avoid the coarse tier's own content.** The repo card is
built from README / CLAUDE.md / AGENTS.md plus the directory listing. If queries
were written from those, routing would score well for a trivial reason. They are
instead about implementation details that live deep in the fine tier
(`ascii_fold`, `GRID_VERSION`, the CSR builder, NVFP4 dequant), so the coarse
tier has to route on topical similarity to a summary it does not contain the
answer to. That is the hard case and the realistic one.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO / "hybrid-code-index"))
sys.path.insert(0, str(REPO / "repo-index"))
sys.path.insert(0, "/home/user/remex")
import hcindex as H  # noqa: E402
from ask import Encoder  # noqa: E402

ROOTS = [Path("/home/user") / n for n in (
    "experiments", "remax", "remex", "remax_kb", "muninn-utilities",
    "claude-workspace", "claude-skills", "claude-container-layers",
    "sklearn-bench")]

# README comes in several spellings and a general-purpose tool cannot assume
# markdown. Measured: sklearn-bench ships README.rst, so an md-only list gave it
# a card of AGENTS.md + a directory listing and every sklearn query mis-routed --
# 4 of 8 misses on the first run, from one missing extension.
CARD_FILES = {"README.md", "README.rst", "README.txt", "README",
              "CLAUDE.md", "AGENTS.md", "SKILL.md", "pyproject.toml"}


def content_card(name: str, chunks: list[H.Chunk], df: dict, n_repos: int,
                 top: int = 400) -> list[H.Chunk]:
    """A card built from what the repo *contains*, not what its README says it is.

    Front matter turned out to be the wrong source. scikit-learn's README.rst
    mentions none of "gradient boosting", "one-hot", "cross validation",
    "sparse" or "estimator" -- it is badges, install instructions and links --
    while its tree has 296 files matching "sparse" and 311 matching "estimator".
    A README describes identity; routing needs inventory.

    So: terms ranked by repo-level tf-idf (frequent here, rare across the other
    repos), plus module path components. Distinctive rather than merely common,
    which is what keeps `import`/`self`/`return` out of every card.
    """
    from collections import Counter
    import math
    tf = Counter()
    paths = Counter()
    for c in chunks:
        tf.update(H.tokens(c.text))
        for part in Path(c.f).parts:
            paths[part.replace(".py", "").replace(".md", "")] += 1
    scored = sorted(tf, key=lambda w: -(tf[w] * math.log(n_repos / max(1, df.get(w, 1)))))
    terms = [w for w in scored if len(w) > 2][:top]
    mods = [w for w, _ in paths.most_common(60)]
    body = (f"repository {name}\ndistinctive terms: " + " ".join(terms) +
            "\nmodules: " + " ".join(mods))
    return [H.Chunk(name, 0, body[i:i + 2000]) for i in range(0, len(body), 2000)]


def repo_card(root: Path, cfg: dict) -> list[H.Chunk]:
    """The always-loaded summary for one repo: front matter + shape.

    Root-level README/CLAUDE.md/AGENTS.md, plus a listing of top-level entries
    and second-level directories. This is what a client holds for every repo, so
    it has to stay small -- a few chunks each.
    """
    out: list[H.Chunk] = []
    name = root.name
    for fn in sorted(CARD_FILES):
        p = root / fn
        if p.exists() and p.is_file():
            body = p.read_text(errors="ignore")[:6000]
            for i in range(0, len(body), cfg["max_chars"]):
                out.append(H.Chunk(name, 0,
                                   f"repository {name}\n{fn}\n{body[i:i+cfg['max_chars']]}"))
    tops = sorted({p.name for p in root.iterdir()
                   if not p.name.startswith(".")})[:60]
    subs = sorted({f"{p.name}/{c.name}" for p in root.iterdir()
                   if p.is_dir() and not p.name.startswith(".")
                   for c in list(p.iterdir())[:40] if c.is_dir()})[:80]
    out.append(H.Chunk(name, 0, f"repository {name}\ncontents: " +
                       ", ".join(tops) + "\nsubdirectories: " + ", ".join(subs)))
    return out


# Queries about *internals*, not front matter. Grouped only for reporting.
QUERIES = [
    "what does ascii_fold do",
    "why is the grid cache keyed on a version constant",
    "Lloyd-Max codebook boundaries and centroids",
    "where do spoke checkouts resolve from",
    "reciprocal rank fusion k parameter",
    "how was the CSR sparse matrix builder implemented for the count-sketch path",
    "how was NVFP4 dequantization performed before encoding",
    "what did the tests for the sparse postings list assert",
    "why is the number of RHT rounds floored at two",
    "stacked sign bit quantizer with k independent rotations",
    "popcount XOR scan over bit-packed codes",
    "Haar rotation via explicit Householder QR reflector convention",
    "ADC search with a lookup table over quantized indices",
    "Matryoshka nested codebooks via right shift",
    "IVF coarse index multi-probe by Hamming distance",
    "tombstones and soft deletes in the knowledge base format",
    "BM25 plus RRF fusion inside the kb reader",
    "srht projection matrix construction for remax_kb",
    "turso memory recall with FTS5 and vector search",
    "boot sequence that materializes utility scripts from memories",
    "blog post publish and bluesky announce helper",
    "grapheme cap check for bluesky posts",
    "container layer composition and cached restore from releases",
    "SessionStart hook that fetches skills tarballs",
    "skill frontmatter name and description validation rules",
    "gerund naming convention for skills",
    "sample weights in gradient boosting loss functions",
    "sparse matrix CSR indptr handling in the estimator",
    "cross validation splitter with stratification",
    "one-hot encoder handling of unknown categories",
]


def main() -> None:
    cfg = dict(H.DEFAULT_CFG)
    # .json measured inert for the fused arm and is 79% of one repo's corpus;
    # dropping it here keeps the encode tractable without changing the question
    cfg["extensions"] = [e for e in cfg["extensions"] if e != ".json"]

    enc = Encoder()
    fine: dict[str, list[H.Chunk]] = {}
    cards: list[H.Chunk] = []
    for root in ROOTS:
        if not root.exists():
            print(f"  (missing {root})")
            continue
        c = dict(cfg)
        c.update(H.load_cfg(root))
        c["extensions"] = cfg["extensions"]
        # This harness lists all 30 queries verbatim and lives inside one of the
        # indexed repos, so leaving it in biases the oracle toward `experiments`
        # and invalidates the fine-tier cache on every edit to it. Fourth
        # instance of this shape here, after code-index-duplication (where the
        # harness retrieved itself for 4 of 9 NL queries) and hybrid-code-index.
        c["exclude"] = list(c.get("exclude", [])) + ["account-routing-tier/*"]
        fine[root.name] = H.build_corpus(root, c)
        cards += repo_card(root, c)
    for n, ch in fine.items():
        print(f"  {n:26s} {len(ch):6d} chunks")
    total = sum(len(v) for v in fine.values())
    print(f"  {'TOTAL fine':26s} {total:6d} chunks")
    print(f"  {'coarse (repo cards)':26s} {len(cards):6d} chunks "
          f"= {100*len(cards)/total:.2f}% of fine\n", flush=True)

    # encode
    flat_chunks, flat_repo = [], []
    for n, ch in fine.items():
        flat_chunks += ch
        flat_repo += [n] * len(ch)
    flat_repo = np.array(flat_repo)
    # cache the fine tier: it is 26k chunks / ~18 min and does not change when
    # only the card construction is being iterated on
    import hashlib
    key = hashlib.sha256("".join(c.text for c in flat_chunks).encode()).hexdigest()[:16]
    cache = Path(f"/tmp/routing_fine_{key}.npy")
    if cache.exists():
        print(f"reusing cached fine tier {cache.name}", flush=True)
        flat_mat = np.load(cache)
    else:
        print("encoding fine tier ...", flush=True)
        flat_mat = enc([c.text for c in flat_chunks], batch=16)
        np.save(cache, flat_mat)
    print("encoding coarse tier ...", flush=True)
    card_mat = enc([c.text for c in cards], batch=16)
    card_repo = np.array([c.f for c in cards])

    flat_bm = H.BM25().fit(flat_chunks)
    card_bm = H.BM25().fit(cards)

    def rank_repos_coarse(q: str) -> list[str]:
        qv = enc([q])[0]
        d, b = card_mat @ qv, card_bm.score(q)
        # rank repos by RRF over the two card rankings, best card per repo
        def best(scores):
            order, seen = np.argsort(-scores), []
            for i in order:
                if card_repo[i] not in seen:
                    seen.append(card_repo[i])
            return seen
        return H.rrf([best(d), best(b)])

    def oracle_repo(q: str) -> str:
        qv = enc([q])[0]
        d, b = flat_mat @ qv, flat_bm.score(q)
        top_d = flat_repo[int(np.argmax(d))]
        # RRF between the two arms at chunk level, then take the winning repo
        dr = [f"{flat_repo[i]}#{i}" for i in np.argsort(-d)[:50]]
        br = [f"{flat_repo[i]}#{i}" for i in np.argsort(-b)[:50]]
        fused = H.rrf([dr, br])
        return fused[0].split("#")[0] if fused else top_d

    # document frequency across repos, for the content-card tf-idf
    from collections import Counter
    df: Counter = Counter()
    for n, ch in fine.items():
        df.update({w for c in ch for w in H.tokens(c.text)})
    ccards = []
    for n, ch in fine.items():
        ccards += content_card(n, ch, df, len(fine))

    print("card chunks per repo:", {n: sum(1 for c in cards if c.f == n)
                                    for n in fine}, flush=True)
    n = len(QUERIES)
    oracles = [oracle_repo(q) for q in QUERIES]
    allrows = {}
    print(f"\n{'card source':28s} {'@1':>6s} {'@2':>6s} {'@3':>6s} {'@5':>6s}  chunks")
    print("-" * 60)
    for label, cc in (("front matter (README etc)", cards),
                      ("content (tf-idf + modules)", ccards),
                      ("both", cards + ccards)):
        cm = enc([c.text for c in cc], batch=16)
        cr = np.array([c.f for c in cc])
        cb = H.BM25().fit(cc)

        def route(q, cm=cm, cr=cr, cb=cb):
            qv = enc([q])[0]
            def best(scores):
                seen = []
                for i in np.argsort(-scores):
                    if cr[i] not in seen:
                        seen.append(cr[i])
                return seen
            return H.rrf([best(cm @ qv), best(cb.score(q))])

        hits, rows = {1: 0, 2: 0, 3: 0, 5: 0}, []
        for q, want in zip(QUERIES, oracles):
            got = route(q)
            pos = got.index(want) + 1 if want in got else 99
            for k in hits:
                hits[k] += pos <= k
            rows.append({"q": q, "oracle": want, "routed": got[:3], "rank": pos})
        print(f"{label:28s} " + " ".join(f"{100*hits[k]/n:5.0f}%" for k in (1,2,3,5))
              + f"  {len(cc)}")
        allrows[label] = rows
    print("\nmisses at k=3, content card:")
    for r in allrows["content (tf-idf + modules)"]:
        if r["rank"] > 3:
            print(f"  want {r['oracle']:22s} got {[str(x) for x in r['routed']]}  <- {r['q'][:44]}")
    json.dump(allrows, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
