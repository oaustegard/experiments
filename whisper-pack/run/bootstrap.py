"""Paired bootstrap of WER difference (A - B) between two schemes on one set.

usage: bootstrap.py results.jsonl SCHEME_A SCHEME_B SET [--model M]
"""
import argparse, json
import numpy as np


def load(path, model=None):
    recs = {}
    for line in open(path):
        if line.strip():
            r = json.loads(line)
            if model and r["model"] != model:
                continue
            recs[(r["scheme"], r["set"])] = r
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl"); ap.add_argument("a"); ap.add_argument("b"); ap.add_argument("set")
    ap.add_argument("--model", default=None)
    ap.add_argument("--resamples", type=int, default=10000)
    x = ap.parse_args()
    recs = load(x.jsonl, x.model)
    ra, rb = recs[(x.a, x.set)], recs[(x.b, x.set)]
    assert ra["ids"] == rb["ids"], "utterance ids differ between schemes"
    ea, eb = np.array(ra["utt_errors"], float), np.array(rb["utt_errors"], float)
    nw = np.array(ra["utt_ref_words"], float)
    n = len(nw)
    idx = np.random.RandomState(0).randint(0, n, size=(x.resamples, n))
    den = nw[idx].sum(1)
    diffs = (ea[idx].sum(1) - eb[idx].sum(1)) / den
    d = ea.sum() / nw.sum() - eb.sum() / nw.sum()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    print(f"{x.set}: {x.a} WER {ea.sum()/nw.sum()*100:.2f}%  {x.b} WER {eb.sum()/nw.sum()*100:.2f}%")
    print(f"diff ({x.a} - {x.b}) = {d*100:+.2f} pts, 95% paired bootstrap CI [{lo*100:+.2f}, {hi*100:+.2f}] "
          f"({x.resamples} resamples, seed 0, n={n})")


if __name__ == "__main__":
    main()
