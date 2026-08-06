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

    print("card chunks per repo:", {n: sum(1 for c in cards if c.f == n)
                                    for n in fine}, flush=True)
    hits = {1: 0, 2: 0, 3: 0, 5: 0}
    rows = []
    for q in QUERIES:
        want = oracle_repo(q)
        got = rank_repos_coarse(q)
        pos = got.index(want) + 1 if want in got else 99
        for k in hits:
            hits[k] += pos <= k
        rows.append({"q": q, "oracle": want, "routed": got[:3], "rank": pos})

    n = len(QUERIES)
    print(f"routing recall over {n} queries, {len(fine)} repos:")
    for k in (1, 2, 3, 5):
        print(f"  recall@{k}  {hits[k]:2d}/{n}  ({100*hits[k]/n:.0f}%)")
    print("\nmisses at k=3:")
    for r in rows:
        if r["rank"] > 3:
            print(f"  want {r['oracle']:22s} got {r['routed']}  <- {r['q'][:48]}")
    json.dump(rows, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
