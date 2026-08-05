"""Does indexing *deleted* code add anything over a repo that documents its rejections?

A current-state index structurally cannot contain code that no longer exists.
That sounds like an obvious win for indexing history — but it is only a win if
the knowledge actually left with the code. `remax` is the hard case: its
CLAUDE.md carries an explicit convention —

    "A measured rejection is an asset — delete the driver, never the record."

— so when apparatus is removed, a prose writeup stays behind in
`bench/results/*.md`. If that convention works, a tombstone index is redundant
there, and the honest result is a negative one.

So the question is not "can history find deleted things" (trivially yes) but
**what is the marginal value of tombstones over a disciplined working tree**.

Two query classes, because they should behave differently:

  existence : "was X ever tried, and what happened?" The surviving record is a
              complete answer. Prediction: current-state already wins.
  mechanism : "how was X implemented?" The record is prose about a verdict; the
              implementation left with the file. Prediction: only tombstones.

Gold is asymmetric on purpose. For `existence`, either the record or the deleted
file answers the question. For `mechanism`, only the deleted file does — a
writeup saying "a CSR builder was built" does not tell you how it was built.

Retrieval is the configuration `hybrid-code-index` measured best: RRF over dense
+ stored BM25.
"""
from __future__ import annotations

import json
import subprocess
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

TARGET = Path("/home/user/remax")


def git(*a: str) -> str:
    return subprocess.run(["git", "-C", str(TARGET), *a],
                          capture_output=True, text=True).stdout


def tombstones(cfg) -> list[H.Chunk]:
    """Content of every file deleted and never restored, at its last living revision.

    The chunk header carries the removing commit and its subject line, because
    the message is often the highest-signal part — remax's own read
    "Delete the ten Nemotron/NVFP4 bench drivers, keep every conclusion".
    """
    dead = [f for f in sorted(set(git("log", "--diff-filter=D", "--name-only",
                                      "--format=").split()))
            if f and not (TARGET / f).exists() and Path(f).suffix in set(cfg["extensions"])]

    # A relocated file looks deleted at its old path. Indexing it as a tombstone
    # would put content that still exists into the "gone" corpus and inflate the
    # measured benefit. Detected by content, not by name: remax moved five files
    # out of src/remax/bench/ into bench/ ("Stop shipping the benchmark harness
    # inside the wheel") and they were 27% of the tombstone corpus on the first
    # run -- one of them satisfied a mechanism query from the working tree, which
    # is what exposed this.
    live_lines = {}
    for lp in TARGET.rglob("*"):
        if lp.is_file() and ".git" not in lp.parts and lp.suffix in set(cfg["extensions"]):
            try:
                live_lines[lp] = set(lp.read_text(errors="ignore").split("\n"))
            except OSError:
                pass

    def relocated(body: str) -> Path | None:
        want = {ln for ln in body.split("\n") if len(ln.strip()) > 20}
        if not want:
            return None
        for lp, have in live_lines.items():
            if len(want & have) / len(want) > 0.5:
                return lp
        return None

    out: list[H.Chunk] = []
    moved: list[str] = []
    for f in dead:
        c = git("log", "--diff-filter=D", "--format=%H", "-1", "--", f).strip()
        if not c:
            continue
        subj = git("log", "-1", "--format=%s", c).strip()
        date = git("log", "-1", "--format=%ad", "--date=short", c).strip()
        body = git("show", f"{c}^:{f}")
        if not body.strip():
            continue
        alive_at = relocated(body)
        if alive_at is not None:
            moved.append(f"{f} -> {alive_at.relative_to(TARGET)}")
            continue
        head = f"{f} [DELETED {date} in {c[:8]}: {subj}]"
        lines = body.split("\n")
        for s in range(0, max(1, len(lines)), cfg["stride"]):
            seg = "\n".join(lines[s:s + cfg["win"]]).strip()
            if len(seg) >= cfg["min_chars"]:
                out.append(H.Chunk(f"[deleted] {f}", s + 1,
                                   f"{head}\n{seg[:cfg['max_chars']]}"))
            if s + cfg["win"] >= len(lines):
                break
    if moved:
        print(f"  excluded {len(moved)} relocated files (content still present):")
        for m in moved:
            print(f"    {m}")
    return out


EXISTENCE = [
    ("Has anyone tried a BM25 or sparse-vector retrieval path in this library?",
     ["BM25_SKETCH", "bm25.py", "sparse.py"]),
    ("Was a learned or data-dependent rotation ever evaluated against the random one?",
     ["LEARNED_ROTATION"]),
    ("Did anyone test structured rotations like a Hadamard transform for SimHash here?",
     ["ROTATION_LSH"]),
    ("Were Nemotron embeddings ever benchmarked in this repo?",
     ["NEMOTRON", "nemotron"]),
    ("Was a benchmark harness ever shipped inside the installable package?",
     ["CROSSOVER", "remax/bench"]),
    ("Has NVFP4 quantization been evaluated?", ["nvfp4", "NEMOTRON"]),
]

MECHANISM = [
    ("How was the CSR sparse matrix builder implemented for the count-sketch path?",
     ["sparse.py"]),
    ("What was the encoder function for BM25-weighted sparse vectors and its signature?",
     ["bm25.py"]),
    ("How did the Nemotron one-bit evaluation script batch and load its embeddings?",
     ["eval_nemotron_1bit.py"]),
    ("How did the Nemotron latency benchmark time and report its query timings?",
     ["latency_nemotron.py"]),
    ("How was NVFP4 dequantization performed before encoding?",
     ["nvfp4_dequant_encode.py"]),
    ("What did the tests for the sparse postings list assert?",
     ["test_sparse_postings.py"]),
]


def main() -> None:
    cfg = H.load_cfg(TARGET)
    live = H.build_corpus(TARGET, cfg)
    dead = tombstones(cfg)
    print(f"remax working tree : {len(live)} chunks / "
          f"{len({c.f for c in live})} files")
    print(f"tombstones         : {len(dead)} chunks / "
          f"{len({c.f for c in dead})} deleted files "
          f"(+{100*len(dead)/len(live):.0f}% corpus)", flush=True)

    enc = Encoder()
    corpora = {"current-only": live, "tombstone-only": dead, "current+tombstone": live + dead}
    cache: dict[int, tuple] = {}
    for name, chunks in corpora.items():
        cache[id(chunks)] = (enc([c.text for c in chunks]), H.BM25().fit(chunks))
    print("encoded\n", flush=True)

    rows = {}
    hdr = f"{'corpus':20s} {'existence':>11s} {'mechanism':>11s} {'TOTAL':>8s}"
    print(hdr); print("-" * len(hdr))
    for name, chunks in corpora.items():
        mat, bm = cache[id(chunks)]

        def top5(q):
            d = H.to_files(mat @ enc([q])[0], chunks, k=20)
            b = H.to_files(bm.score(q), chunks, k=20)
            return H.rrf([d, b])[:5]

        scores = []
        for cls in (EXISTENCE, MECHANISM):
            scores.append(sum(any(any(g.lower() in f.lower() for g in gold)
                                  for f in top5(q)) for q, gold in cls))
        a, b = scores
        print(f"{name:20s} {a:>9d}/6 {b:>9d}/6 {a+b:>6d}/12")
        rows[name] = {"existence": a, "mechanism": b, "total": a + b}

    json.dump(rows, open(HERE / "results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
