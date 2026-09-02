"""Build the inversion corpus: short English strings across a length range,
embedded once with bekko-a8m (float). Everything downstream reads data/.

Sources (both pulled by the driver via huggingface_hub, gitignored):
  - sentence-transformers/natural-questions  pair/  -> NQ questions + sentences
    split from the Wikipedia answer passages
  - sentence-transformers/msmarco            queries/ -> MS MARCO queries

The mix is deliberate: questions are 4-12 words, wiki sentences run to 32, so
the length axis is populated for the capacity curve. Splits are by string,
deduped, seeded.
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from encoder import BekkoEncoder, SignBits

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

_SENT = re.compile(r"(?<=[.!?])\s+(?=[A-Z])")


def wiki_sentences(passages, lo=6, hi=32, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for p in passages:
        for s in _SENT.split(p):
            s = s.strip()
            n = len(s.split())
            if lo <= n <= hi and s[0].isupper() and s[-1] in ".!?" and "[" not in s and "(" not in s:
                out.append(s)
    rng.shuffle(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=40000)
    ap.add_argument("--n-dev", type=int, default=1000)
    ap.add_argument("--n-test", type=int, default=1000)
    ap.add_argument("--max-tokens", type=int, default=40, help="bekko token cap")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wiki-parquet", nargs="*", default=[],
                    help="extra sentence sources: parquet files with a `sentence` column "
                         "(sentence-transformers/wikipedia-en-sentences, 7.8M rows in two files)")
    ap.add_argument("--questions-frac", type=float, default=0.5, help="share of the corpus that is questions")
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    nq = pq.read_table(DATA / "pair/train-00000-of-00001.parquet").to_pandas()
    ms = pq.read_table(DATA / "queries/train-00000-of-00001.parquet").to_pandas()
    total = a.n_train + a.n_dev + a.n_test
    half = int(total * a.questions_frac)

    seen = set()
    questions = []
    for q in [q.strip() for q in nq["query"].tolist()] + [q.strip() for q in ms["query"].tolist()]:
        if q.lower() not in seen:
            seen.add(q.lower()); questions.append(q)
    questions = [q for q in questions if 3 <= len(q.split()) <= 32 and len(q) < 200]
    rng.shuffle(questions)
    sents = wiki_sentences(nq["answer"].tolist(), seed=a.seed)
    for wp in a.wiki_parquet:  # already one sentence per row; apply the same length/shape filter
        extra = pq.read_table(wp, columns=["sentence"]).column("sentence").to_pylist()
        extra = [x.strip() for x in extra if x and 6 <= len(x.split()) <= 32 and x[0].isupper()
                 and x.strip()[-1] in ".!?" and "[" not in x and "(" not in x]
        rng.shuffle(extra)
        sents += extra[: total]  # more than enough after dedupe; keeps memory bounded
        print(f"{wp}: {len(extra)} usable sentences", flush=True)
    seen, uniq = set(), []
    for x in sents:
        if x.lower() not in seen:
            seen.add(x.lower()); uniq.append(x)
    sents = uniq

    enc = BekkoEncoder()
    def take(cands, n):
        cands = cands[: int(n * 1.5)]
        ntok = enc.n_tokens(cands)
        return [t for t, k in zip(cands, ntok) if k <= a.max_tokens][:n]
    qs = take(questions, half)
    ss = take(sents, total - half)
    texts = qs + ss
    rng.shuffle(texts)
    assert len(texts) == total, (len(texts), total)

    splits = {
        "train": texts[: a.n_train],
        "dev": texts[a.n_train: a.n_train + a.n_dev],
        "test": texts[a.n_train + a.n_dev:],
    }
    (DATA / "splits.json").write_text(json.dumps(splits, ensure_ascii=False))
    print(f"corpus built: {len(questions)} questions, {len(sents)} sentences available; "
          f"{ {k: len(v) for k, v in splits.items()} } ({time.time()-t0:.0f}s)", flush=True)

    for k, v in splits.items():
        t = time.time()
        e = enc.encode(v, batch_size=64)
        np.save(DATA / f"emb_{k}.npy", e)
        print(f"embedded {k}: {e.shape} in {time.time()-t:.0f}s "
              f"({len(v)/(time.time()-t):.0f} texts/s)", flush=True)

    sb = SignBits.fit(np.load(DATA / "emb_train.npy"))
    sb.save(DATA / "signbits_mu.npy")
    wl = np.array([len(t.split()) for t in splits["test"]])
    meta = {
        "n": {k: len(v) for k, v in splits.items()},
        "test_words": {"median": float(np.median(wl)), "p10": float(np.percentile(wl, 10)),
                       "p90": float(np.percentile(wl, 90)), "max": int(wl.max())},
        "test_bekko_tokens_median": float(np.median(enc.n_tokens(splits["test"]))),
        "encoder_output": enc.out_name,
        "seed": a.seed,
    }
    (DATA / "meta.json").write_text(json.dumps(meta, indent=1))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
