#!/usr/bin/env python3
"""query string -> ranked shell-utility search results. No model involved.

This is the "no model" core traced from the code that already exists in
`nl2sh-dense/` and `nl2sh-retrieval/` (issue experiments#48/#52). It imports
those modules rather than reimplementing BM25, the dense arm, or fusion:

    retrieve.Index          nl2sh-retrieval/retrieve.py   BM25 over chunk text
    dense_index.DenseArm    nl2sh-dense/dense_index.py    cosine over cached vectors
    dense_index.rank_utilities / .rrf / .wsum              chunk-score -> utility fusion
    encoders.LeafMTEncoder  nl2sh-dense/encoders.py        the ONNX query/doc encoder
    enrich.render           nl2sh-dense/enrich.py          corpus-rewrite text builder
    pleias_gate.load_tldr   nl2sh-retrieval/pleias_gate.py raw tldr example parser

The one new thing here is the plumbing that connects them into `search()` and
redirects every cache outside the repo, to ~/.cache/nl2sh/ (or $NL2SH_CACHE).

WHAT THIS DOES CARRY, and matches RESULTS.md exactly:

* The **enriched corpus** (nl2sh-dense/RESULTS.md sect. 7): one document per
  tldr/man PAGE (dense_index.page_chunks), rendered through enrich.render()
  with the summary / category / "use when you want to" intents / "not for"
  disambiguation prepended and the original tldr text kept verbatim last.
  `data/cards.jsonl` already holds all 6,397 pages' generated cards (6,396
  enriched, 1 refusal kept as plain text, per RESULTS.md sect. 7) — this
  script reads that file and calls the *same* `enrich.render()` function the
  original enrichment run used. It does NOT call the Gemini enrichment model;
  there is nothing left to generate.
* **BM25** over that corpus, via `retrieve.Index` unmodified.
* **The dense arm**, via `dense_index.DenseArm` unmodified, with the
  shipped-recommendation encoder **leaf-mt-int8** (`MongoDB/mdbr-leaf-mt`,
  ONNX int8, 1024-d, ~25.6 MB — RESULTS.md sect. 1: "matches a 164.5 MB one").
  Its ONNX asset was not present anywhere on this box or in the repo's
  releases, so this script fetched it fresh from
  https://huggingface.co/MongoDB/mdbr-leaf-mt (onnx/model_quantized.onnx{,_data},
  tokenizer.json, 2_Dense/model.safetensors) into $NL2SH_DENSE_MODELS.
  `nl2sh-dense/pack_kb.py`'s `LeafMTKBEmbedder.release_sha256` is `None` in
  the checked-in code despite that file's own docstring claiming the asset is
  "SHA-pinned" — only the URL is actually pinned. sha256 of what this script
  fetched is recorded in the probe report, not in this file, since a source
  file should not hardcode a hash nobody in the repo asked it to verify against.
* **Page granularity** (RESULTS.md sect. 2: the single largest lift measured,
  "the largest single lift in this writeup: 0.262 -> 0.390").
* **Fusion = RRF(BM25, dense)**, both arms scored on identical document text
  (`f"{utility} {text}"`) as `dense_index.doc_text` requires. This is the one
  place RESULTS.md is not fully unambiguous: the whole-stack table's "+ dense
  arm, page-level index" row (0.390) traces to the page-level RRF number in
  sect. 2, and the reproduce section's own end-to-end command is
  `--retriever rrf:bm25+leaf-mt-int8 --granularity page`; both point at RRF.
  Section 4 separately recommends switching to the *weighted sum* — but only
  for building a confidence/abstention gate on top ("RRF is the wrong
  substrate for an abstention gate... If the gate ships, the retriever under
  it should be the weighted sum"), a different concern than which fusion
  produced 0.555. This script follows the reproduce command: RRF.
* **The `source_line` shape the generator prompt actually expects.** Every
  place in this repo that builds a generator prompt — `gemma_arm.make_sources`,
  `gemma_fullsystem.retrieved_sources`, `fullsystem_dense.py`'s
  `render(..., source_form="example")` (the shipped default), and
  `nl2sh-instantiate/run_gen.py`'s `make_sources` — constructs the exact same
  string: `f"{utility} — {description}: {command}"`, where `(description,
  command)` is a utility's FIRST tldr example, parsed straight from the raw
  tldr pages by `pleias_gate.load_tldr` / `parse_tldr`. Critically, that text
  is NOT drawn from the enriched corpus or from cards.jsonl — the enriched
  corpus and cards.jsonl only make RETRIEVAL better (they change which
  utilities get ranked highly); the text handed to the generator is always
  built fresh from the plain tldr page, regardless of which retriever ranked
  it there. `run_gen.py`'s `make_sources` builds this same string directly
  from an oracle utility list (gold + distractors) rather than from a real
  ranking, which is the only difference from what this probe does — this
  probe's `source_line` is built from a REAL ranking's output utility, using
  the identical string format. `search()` below reproduces that shape exactly
  via `pleias_gate.load_tldr(TLDR_DIR)`.

WHAT THIS DOES NOT CARRY, and what that costs:

* **The trained query adapter** (RESULTS.md sect. 5, `adapter.py`). No `.npz`
  adapter weights exist anywhere in the repo or on this box, and this script
  does not train one. Even if it did: `pack_kb.py`'s own docstring says why
  not to ship it — "the adapter's gain was entirely on the 207 [NL2Bash]
  utilities it trained on — the least portable component. Dropping it is the
  right call for an artifact, not a limitation to apologise for." RESULTS.md
  sect. 5 measures the same thing quantitatively: +0.184 gold-in-sources on
  the utilities it saw in training, -0.039 on utilities it never saw. So this
  probe's ceiling against the measured stack is the **"enriched corpus, no
  adapter"** row of the sect. 7 whole-stack table — **0.506 gold-in-sources /
  0.226 end-to-end routing** — not the with-adapter headline of 0.555 / 0.250.
* **The packed `.kbi` artifact** (`pack_kb.py`, `remax_kb`). `remax_kb` is not
  installed on this box and is not on PyPI under either `remax-kb` or
  `remax_kb` (`pip install` returns "No matching distribution found" for
  both, probed once as instructed). This script therefore builds the BM25 +
  dense indices directly in memory via `retrieve.Index` / `dense_index.DenseArm`
  rather than reading a `.kbi` file. Retrieval quality is identical (same
  arms, same corpus, same fusion) — what's lost is the .kbi's packaging
  convenience: a single 7.3 MB + 6.1 MB file and quantization down to 4 bits
  (measured in RESULTS.md at a 0.014 gold-in-sources cost from the round trip).
* **funceq / flag correctness.** Every score anywhere in this pipeline,
  including this probe's, is "is the right utility in the results" — never
  "are the flags right". RESULTS.md's caveats section says so explicitly and
  it applies here unchanged.
* **PATH scoping** (`nl2sh-retrieval/nlsh.py`'s `path_utilities`). This probe
  searches the full 4,698-utility corpus, not just what is installed on this
  machine's `$PATH` — RESULTS.md's own caveat says every number in that
  document is therefore "the unscoped floor", and the same is true here.

Cache layout, all under $NL2SH_CACHE (default ~/.cache/nl2sh/), never in the
repo:
    models/leaf-mt/...        the encoder's ONNX + tokenizer + Dense head
    vectors/*.npy             cached page embeddings (dense_index.build_vectors,
                               redirected here by reassigning dense_index.CACHE
                               at runtime — no file in the repo is touched)

Usage:
    python3 retrieval_probe.py "recover the password for backup.zip"
    python3 retrieval_probe.py -k 8 "find every .log file under /var/log"
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

# Threads: whatever the machine has, unless the caller says otherwise. The
# encoder and BLAS both read these at import time, so they are set before any
# of it is imported. NL2SH_THREADS overrides for a shared or contended box.
THREADS = int(os.environ.get("NL2SH_THREADS", 0)) or (os.cpu_count() or 4)
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, str(THREADS))

CACHE_ROOT = Path(os.environ.get("NL2SH_CACHE", str(Path.home() / ".cache" / "nl2sh")))
# encoders.py reads this at IMPORT time, so it must be set before `import encoders`
# (which happens transitively via `import dense_index as D` below).
os.environ.setdefault("NL2SH_DENSE_MODELS", str(CACHE_ROOT / "models"))
VECTOR_CACHE = CACHE_ROOT / "vectors"

HERE = Path(__file__).resolve().parent
# This module reads the corpus and the retrieval code out of the sibling
# experiment directories. That makes it runnable from a checkout and NOT
# runnable from a bare `pip install`, which `status()` reports rather than
# discovering at query time. Packaging the corpus is the open item; see README.
EXPERIMENTS_ROOT = Path(os.environ.get("NL2SH_EXPERIMENTS", HERE.parent.parent))
sys.path.insert(0, str(EXPERIMENTS_ROOT))
try:
    from _lib.paths import experiment  # noqa: E402
except ImportError as _e:                                   # pragma: no cover
    raise ImportError(
        f"nl2sh.search needs the oaustegard/experiments checkout "
        f"(looked in {EXPERIMENTS_ROOT}); set NL2SH_EXPERIMENTS to point at it"
    ) from _e

DENSE_DIR = experiment("nl2sh-dense")
RETRIEVAL_DIR = experiment("nl2sh-retrieval")
sys.path.insert(0, str(RETRIEVAL_DIR))
sys.path.insert(0, str(DENSE_DIR))

import retrieve as R       # noqa: E402  nl2sh-retrieval/retrieve.py   (BM25)
import dense_index as D    # noqa: E402  nl2sh-dense/dense_index.py   (dense + fusion)
import enrich as ENR       # noqa: E402  nl2sh-dense/enrich.py        (corpus render)
import pleias_gate as PG   # noqa: E402  nl2sh-retrieval/pleias_gate.py (tldr examples)

# Redirect dense_index's vector cache outside the repo. dense_index.py hardcodes
# `CACHE = <module dir>/cache`; reassigning the module attribute at runtime does
# not touch any file on disk, and every function that writes there (`build_vectors`,
# `cache_path`) reads `CACHE` through the module namespace, so this redirection
# is honored by code we did not modify.
VECTOR_CACHE.mkdir(parents=True, exist_ok=True)
D.CACHE = VECTOR_CACHE

MODEL = "leaf-mt-int8"      # RESULTS.md sect. 1 / sect. 4: the shipped-recommendation encoder
GRANULARITY = "page"        # RESULTS.md sect. 2: the largest single lift measured
POOL = 400                  # dense_index.rank_utilities' own default

TLDR_DIR = Path(os.environ.get("NL2SH_TLDR_DIR", "/home/user/corpora/tldr/pages"))
CARDS_PATH = DENSE_DIR / "data" / "cards.jsonl"
CHUNKS_PATH = RETRIEVAL_DIR / "data" / "chunks.jsonl"
# A synthetic corpus path, only ever used for its .stem by dense_index.cache_path
# to key the vector cache filename distinctly from the plain (non-enriched)
# corpus's cache entry. Never read or written as a file.
_ENRICHED_CORPUS_KEY = VECTOR_CACHE / "cards_enriched_pages.jsonl"


class _State:
    __slots__ = ("index", "dense", "tldr", "util_cards")


_state: _State | None = None


def _build_enriched_pages():
    """One `retrieve.Chunk` per tldr/man page, text rewritten by `enrich.render`.

    Reproduces exactly what `enrich.py --render-only` would write to
    `chunks_enriched.jsonl`, from the cards that are already on disk — no LLM
    call happens here.
    """
    cards: dict[str, dict | None] = {}
    with CARDS_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cards[rec["id"]] = rec.get("card")

    pages = D.page_chunks(R.load_chunks(CHUNKS_PATH))
    enriched = [
        R.Chunk(p.id, p.utility, p.kind, ENR.render(p, cards.get(p.id)), p.runnable)
        for p in pages
    ]

    util_cards: dict[str, dict] = {}
    for p in pages:
        c = cards.get(p.id)
        if c and p.utility not in util_cards:
            util_cards[p.utility] = c
    return enriched, util_cards


def _load() -> _State:
    global _state
    if _state is not None:
        return _state

    enriched, util_cards = _build_enriched_pages()

    index = R.Index(enriched)

    vectors = D.build_vectors(
        MODEL, enriched, utility_prefix=True, threads=THREADS,
        granularity=GRANULARITY, corpus=_ENRICHED_CORPUS_KEY,
    )
    dense = D.DenseArm(MODEL, enriched, vectors, threads=THREADS, adapter=None)

    tldr = PG.load_tldr(TLDR_DIR)

    st = _State()
    st.index, st.dense, st.tldr, st.util_cards = index, dense, tldr, util_cards
    _state = st
    return st


def _source_line(utility: str, tldr: dict) -> tuple[str, str]:
    """(runnable example command, generator-ready source line).

    Exact shape used everywhere a prompt is built in this repo:
    `gemma_arm.make_sources`, `gemma_fullsystem.retrieved_sources`,
    `fullsystem_dense.render(source_form="example")`, and
    `nl2sh-instantiate/run_gen.make_sources` all emit
    `f"{utility} — {description}: {command}"` from a utility's first tldr
    example. A utility that only has man-page chunks (no tldr page at all —
    real for some of the 4,698 in this corpus) has no example to offer here;
    it degrades to the bare utility name, matching `fullsystem_dense.py`'s
    own `source_form="name"` control condition rather than inventing text.
    """
    examples = tldr.get(utility)
    if not examples:
        return "", utility
    desc, cmd = examples[0]
    return cmd, f"{utility} — {desc}: {cmd}"


def search(query: str, k: int = 5) -> list[dict]:
    """Rank shell utilities for `query`. No model, no network call, no state
    beyond the process-local index built (and cached) on first call.

    Each result dict carries:
        utility      the command name
        score        the RRF fusion score (not comparable across queries)
        example      a runnable example command for this utility, or "" if
                     the utility has no tldr page in this corpus
        source_line  f"{utility} — {description}: {command}", the exact
                     string shape the fine-tuned generator's prompt expects
                     (see module docstring)
        summary      one-line enriched summary, if this utility's page was
                     successfully enriched (bonus field, not in the spec)
        rank         1-based rank in this result list
    """
    st = _load()

    bm_scores = st.index.scores(query)
    bm_ranked = D.rank_utilities(bm_scores, st.index.utilities, pool=POOL, positive_only=True)

    dn_scores = st.dense.scores(query)
    dn_ranked = D.rank_utilities(dn_scores, st.dense.utilities, pool=POOL)

    fused = D.rrf(bm_ranked, dn_ranked)

    out = []
    for rank, (utility, score) in enumerate(fused[:k], 1):
        example, source_line = _source_line(utility, st.tldr)
        card = st.util_cards.get(utility)
        out.append({
            "utility": utility,
            "score": float(score),
            "example": example,
            "source_line": source_line,
            "summary": (card or {}).get("summary", ""),
            "rank": rank,
        })
    return out


def _main(argv: list[str]) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("query", nargs="+")
    ap.add_argument("-k", type=int, default=5)
    a = ap.parse_args(argv)
    query = " ".join(a.query)

    t0 = time.time()
    hits = search(query, k=a.k)
    t_cold = time.time() - t0

    print(f"query: {query!r}")
    print(f"[cold: {t_cold * 1000:.0f} ms, includes index build + encoder load]\n")
    for h in hits:
        print(f"{h['rank']}. {h['utility']:<16} {h['score']:.4f}")
        if h["summary"]:
            print(f"   {h['summary']}")
        if h["example"]:
            print(f"   $ {h['example']}")
        print(f"   source_line: {h['source_line']!r}")
        print()

    # warm timing: same query, then a couple of process-local repeats
    warm_times = []
    for _ in range(3):
        t0 = time.time()
        search(query, k=a.k)
        warm_times.append(time.time() - t0)
    avg_warm = sum(warm_times) / len(warm_times)
    print(f"[warm: {avg_warm * 1000:.1f} ms avg over {len(warm_times)} repeat "
          f"queries, {[round(t * 1000, 1) for t in warm_times]}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))


def status() -> str:
    """One line for `nl2sh doctor`: what is present and what a query would cost.

    Reports readiness without building anything, so `doctor` stays instant on a
    machine where the first query would otherwise spend three minutes encoding.
    """
    bits = []
    for label, path in (("corpus", CARDS_PATH), ("chunks", CHUNKS_PATH),
                        ("tldr", TLDR_DIR)):
        bits.append(f"{label}={'ok' if path.exists() else 'MISSING'}")
    vecs = sorted(VECTOR_CACHE.glob("*.npy")) if VECTOR_CACHE.exists() else []
    enc = (CACHE_ROOT / "models" / MODEL)
    bits.append(f"encoder={'cached' if enc.exists() else 'will download ~26 MB'}")
    if vecs:
        bits.append("vectors=cached (first query ~3 s)")
    else:
        bits.append("vectors=absent (first query encodes 6,397 pages, ~2-3 min)")
    bits.append(f"threads={THREADS}")
    return "  ".join(bits)
