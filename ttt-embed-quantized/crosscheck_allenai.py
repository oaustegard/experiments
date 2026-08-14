#!/usr/bin/env python3
"""Is BEIR SciFact reconstructible from the upstream AllenAI release? Partly.

A second independent encode of this artifact (PR #35) was run in a container
where **every** Hugging Face host answers 403 at the egress proxy, so it rebuilt
the corpus from the AllenAI release BEIR wraps
(`scifact.s3-us-west-2.amazonaws.com`) instead of from `BeIR/scifact`. It
verified the reconstruction against BEIR's four published cardinalities
(5183 docs / 1109 claims / 300 test queries / 339 test qrels) and all four
matched -- then registered, correctly, that counts pin the *split* and not the
document *strings*, and asked whoever could reach HF to run the per-document
diff.

This is that diff. The answer:

    exact doc-string match : 4128/5183 (79.64%)
    differing doc strings  : 1055     (20.36%)
    title differs          : 0

**Cause.** AllenAI's `abstract` is a list of sentences, and at structured-abstract
section boundaries those sentences carry trailing whitespace that BEIR
normalised away. `" ".join(abstract)` preserves it:

    BEIR     ...'detected. RESULTS We propose an empirical Bayesian model...'
    AllenAI  ...'detected.   \\n RESULTS We propose an empirical Bayesian model...'

**Why it matters.** Both encodes landed inside the pre-registered 0.60-0.72
nDCG@10 sanity band (0.7152 from BEIR text, 0.7067 from the reconstruction), so
the band did not catch it -- exactly as that PR's own `ANCHORS.md` row predicted
a loose band would not. A count check plus a wide metric band is not a
substitute for comparing the strings.

Run:  python3 crosscheck_allenai.py
      python3 crosscheck_allenai.py --other-dm PATH/Dm.npy --other-meta PATH/meta.json
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from encode import RAW, SOURCES, fetch  # noqa: E402

ALLENAI_URL = "https://scifact.s3-us-west-2.amazonaws.com/release/latest/data.tar.gz"


def load_allenai() -> dict[str, tuple[str, list[str]]]:
    tgz = RAW / "scifact_allenai.tar.gz"
    root = RAW / "allenai"
    if not (root / "data" / "corpus.jsonl").exists():
        fetch(ALLENAI_URL, tgz)
        root.mkdir(parents=True, exist_ok=True)
        with tarfile.open(tgz) as t:
            t.extractall(root)
    out = {}
    with (root / "data" / "corpus.jsonl").open() as f:
        for line in f:
            d = json.loads(line)
            out[str(d["doc_id"])] = (d["title"], d["abstract"])
    return out


def load_beir() -> dict[str, tuple[str, str]]:
    import pyarrow.parquet as pq

    fetch(SOURCES["corpus.parquet"], RAW / "corpus.parquet")
    return {
        str(r["_id"]): (r["title"], r["text"])
        for r in pq.read_table(RAW / "corpus.parquet").to_pylist()
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--other-dm", type=Path, help="a second Dm.npy to compare vectors against")
    ap.add_argument("--other-meta", type=Path, help="that artifact's meta.json (for doc_id order)")
    args = ap.parse_args()

    allen, beir = load_allenai(), load_beir()
    meta = json.loads((HERE / "data" / "meta.json").read_text())
    doc_ids = meta["doc_ids"]

    print(f"allenai docs {len(allen)}  beir docs {len(beir)}  same id set {set(allen) == set(beir)}")

    def rebuilt(d):  # the reconstruction PR #35 used
        t, a = allen[d]
        return f'{t}. {" ".join(a)}'

    def truth(d):  # what BeIR/scifact actually ships
        t, x = beir[d]
        return f"{t}. {x}"

    differs = np.array([rebuilt(d) != truth(d) for d in doc_ids])
    same = int((~differs).sum())
    print(f"\nexact doc-string match : {same}/{len(doc_ids)} ({100 * same / len(doc_ids):.2f}%)")
    print(f"differing doc strings  : {int(differs.sum())} ({100 * differs.mean():.2f}%)")
    print(f"title differs          : {sum(allen[d][0] != beir[d][0] for d in doc_ids)}")

    shown = 0
    for i, d in enumerate(doc_ids):
        if differs[i] and shown < 3:
            a, b = truth(d), rebuilt(d)
            j = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), 0)
            print(f"\ndoc {d}")
            print(f"  BEIR    ...{a[max(0, j - 55):j + 55]!r}")
            print(f"  AllenAI ...{b[max(0, j - 55):j + 55]!r}")
            shown += 1

    ok = same == 4128 and int(differs.sum()) == 1055
    print(f"\n{'PASS' if ok else 'FAIL'}  reproduces the documented 4128 / 1055 split")

    if args.other_dm and args.other_meta:
        other = np.load(args.other_dm)
        om = json.loads(args.other_meta.read_text())
        pos = {d: i for i, d in enumerate(om["doc_ids"])}
        mine = np.load(HERE / "data" / "Dm.npy")
        cos = (other[[pos[d] for d in doc_ids]] * mine).sum(1)
        low = cos < 0.9999
        print("\n=== does the text difference explain the vector difference? ===")
        print(f"  text differs & vector differs : {int((differs & low).sum())}")
        print(f"  text differs & vector same    : {int((differs & ~low).sum())}")
        print(f"  text same    & vector differs : {int((~differs & low).sum())}  <- must be 0")
        print(f"  text same    & vector same    : {int((~differs & ~low).sum())}")
        print(f"  per-doc cosine: mean {cos.mean():.6f}  min {cos.min():.6f}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
