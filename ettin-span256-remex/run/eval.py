"""Step 3: remex codes on the 768-d outputs, on the folded 256-d outputs, and with remex folded in.

Usage: python3 -I run/eval.py

Corpus: the 61,686 Bluesky posts. Queries: synth (1,498 Haiku queries, each written from one
held-out post) and real (1,000 MS MARCO tech/AI queries). Ground truth: exact inner-product
top 10 on the shipped model's fp32 768-d vectors.

Spaces (docs and queries embedded the same way in each):
  full768        the model's output as shipped; bits=2 centred is the live index
  span256        the exact 256-d fold (fold.py), remex applying its own rotation
  prefix256      the first 256 of the 768 coordinates, renormalized (control)
  span256_norot  the 256 SVD coordinates quantized with no rotation (control: is the
                 rotation still needed once the head is folded?)
  integrated256  the head to deploy: remex's rotation folded into an int8 256 -> 256 head, so
                 the model's output is already rotated and a code is a boundary lookup
Each at 1, 2, 3, 4 and 8 bits with remex Quantizer(d, bits, seed=42), centred on the corpus
mean of that space; full768 and span256 also uncentred. Codes are scored as search_adc scores
them (float query against decoded directions, times scale, plus bias), as one matmul; every
arm checks 20 queries against search_adc itself.

Metrics: recall@10 against the ground truth on both sets, hit@10 (the source post in the top
10) on synth, each with a bootstrap 95% interval, and the paired difference to the live
index (full768 centred 2-bit). Bytes per vector = d * bits / 8 plus the stored floats
(scale and bias, 8 B centred; scale, 4 B uncentred). Five remex seeds for the arms the
answer rests on. Writes results/recall.json.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import remex

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import OUT, RES, SETS, K, boot, hits, load, recall, topk, unit

BITS = (1, 2, 3, 4, 8)
LIVE = "full768_c_b2"


def code_arm(Xd, Xq, bits, centred, rotation=None, seed=42):
    d = Xd.shape[1]
    mean = Xd.mean(0).astype(np.float32) if centred else None
    kw = {} if rotation is None else {"rotation": rotation}
    qz = remex.Quantizer(d, bits=bits, seed=seed, mean=mean, **kw)
    cv = qz.encode(Xd)
    dirs = qz.centroids.astype(np.float32)[cv.indices]
    scale = qz._score_norms(cv).astype(np.float32)
    bias = qz._row_bias(cv)
    bias = 0.0 if bias is None else np.asarray(bias, np.float32)

    def score(Q):
        return (qz._rotate_rows(np.asarray(Q, np.float32)).astype(np.float32) @ dirs.T) * scale + bias

    tops = {k: topk(score, Xq[k]) for k in SETS}
    adc = np.stack([qz.search_adc(cv, q, k=K)[0] for q in Xq["real"][:20]])
    meta = {"d": d, "bits": bits, "centred": centred, "rotation": qz.rotation,
            "bytes": d * bits // 8 + (8 if centred else 4),
            "adc_check_top10_agree": float(recall(adc, tops["real"][:20]).mean())}
    return tops, meta, cv.indices


def main():
    import warnings
    warnings.simplefilter("ignore")  # remex warns on uncentred low-bit encodes of anisotropic data; that arm is deliberate
    X, _, _, src = load()
    Y = {k: np.load(OUT / "vec" / f"{k}.y.npy") for k in X}
    YQ = {k: np.load(OUT / "vec" / f"{k}.yq.npy") for k in X}
    truth = {k: topk(lambda q: q @ X["docs"].T, X[k]) for k in SETS}
    spaces = {
        "full768": (X, None), "span256": (Y, None),
        "prefix256": ({k: unit(X[k][:, :256]) for k in X}, None),
        "span256_norot": (Y, "none"), "integrated256": (YQ, "none"),
    }

    def measure(tops):
        r = {f"recall_at_10_{k}": recall(tops[k], truth[k]) for k in SETS}
        r["hit_at_10"] = hits(tops["synth"], src)
        return r

    per_q, meta = {}, {}
    for name, bytes_ in (("fp32_768", 3072), ("fp32_256", 1024)):
        S = X if name == "fp32_768" else Y
        per_q[name] = measure({k: topk(lambda q, D=S["docs"]: q @ D.T, S[k]) for k in SETS})
        meta[name] = {"d": S["docs"].shape[1], "bits": 32, "bytes": bytes_}
    t0 = time.time()
    for name, (S, rot) in spaces.items():
        for centred in ((True, False) if name in ("full768", "span256") else (True,)):
            for b in BITS:
                arm = f"{name}_{'c' if centred else 'u'}_b{b}"
                tops, meta[arm], _ = code_arm(S["docs"], S, b, centred, rot)
                per_q[arm] = measure(tops)
                print(arm, meta[arm]["bytes"], {k: round(float(v.mean()), 4) for k, v in per_q[arm].items()},
                      f"{time.time() - t0:.0f}s", flush=True)

    res = {"n_docs": len(X["docs"]), "n_queries": {k: len(X[k]) for k in SETS}, "k": K, "live_arm": LIVE,
           "remex": remex.__version__, "arms": {}}
    for arm, r in per_q.items():
        row = dict(meta[arm])
        for m, v in r.items():
            row[m] = boot(v)
            if arm != LIVE:
                row[f"{m}_minus_live"] = boot(v - per_q[LIVE][m])
        res["arms"][arm] = row

    seeds = {}
    for arm, (space, centred, b) in {"full768_c_b1": ("full768", True, 1), "full768_c_b2": ("full768", True, 2),
                                     "full768_c_b3": ("full768", True, 3), "span256_c_b3": ("span256", True, 3),
                                     "span256_c_b4": ("span256", True, 4), "span256_c_b8": ("span256", True, 8)}.items():
        S = spaces[space][0]
        runs = []
        for seed in range(5):
            tops, _, _ = code_arm(S["docs"], S, b, centred, None, seed)
            runs.append({m: float(v.mean()) for m, v in measure(tops).items()})
        seeds[arm] = {m: {"mean": float(np.mean([r[m] for r in runs])), "sd": float(np.std([r[m] for r in runs], ddof=1)),
                          "min": float(min(r[m] for r in runs)), "max": float(max(r[m] for r in runs))} for m in runs[0]}
        print("seeds", arm, {m: (round(v["mean"], 4), round(v["sd"], 4)) for m, v in seeds[arm].items()}, flush=True)
    res["seeds"] = seeds
    (RES / "recall.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
