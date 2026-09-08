"""Evaluate every (model arm x decoder) pair on the MSLR-WEB10K test slates.

Reports the PLAN.md diagnostics (G1 collisions, G2 disagreement, G3 rank-1
energy) before the ranking metrics, because if M is rank-1 or collision-free
then D1 == D2 by construction and the metric comparison measures nothing.
"""
import json
from pathlib import Path

import numpy as np
import torch

from decoders import (
    DECODERS,
    d1_hungarian,
    d2_row_greedy,
    kendall_tau_vs,
    mrr,
    ndcg,
    rank1_energy,
    recall_at_k,
    row_argmax_collisions,
)
from train import Student

HERE = Path(__file__).resolve().parent
SEED = 20260908
BOOT = 10000


def load(arm, ep):
    p = HERE / "checkpoints" / f"{arm}_ep{ep}.pt"
    c = torch.load(p, weights_only=False)
    m = Student(c["head"], d_in=c["d_in"], K=c["K"])
    m.load_state_dict(c["state"]); m.eval()
    return m


@torch.no_grad()
def infer(model, X, batch=128):
    return np.concatenate([model(X[i:i + batch]).numpy()
                           for i in range(0, len(X), batch)])


def score_rankings(labels, rankings, teacher_perm):
    """Per-slate metric arrays for one decoder."""
    out = {k: np.empty(len(labels)) for k in
           ("ndcg1", "ndcg10", "ndcg50", "recall1", "recall10", "mrr", "tau")}
    for s, (L, r, tp) in enumerate(zip(labels, rankings, teacher_perm)):
        out["ndcg1"][s] = ndcg(L, r, 1)
        out["ndcg10"][s] = ndcg(L, r, 10)
        out["ndcg50"][s] = ndcg(L, r, 50)
        out["recall1"][s] = recall_at_k(L, r, 1)
        out["recall10"][s] = recall_at_k(L, r, 10)
        out["mrr"][s] = mrr(L, r)
        out["tau"][s] = kendall_tau_vs(r, tp)
    return out


def paired_boot(a, b, rng, n=BOOT):
    """95% CI on mean(a) - mean(b), resampling slates (paired)."""
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    idx = rng.integers(0, a.size, size=(n, a.size))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float(np.mean(a) - np.mean(b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main():
    rng = np.random.default_rng(SEED)
    torch.set_num_threads(4)
    z = np.load(HERE / "data" / "slates.npz")
    X = torch.from_numpy((z["Xte"] - z["mu"]) / z["sd"])
    Y = z["yte"]; TP = z["teacher_perm_te"].astype(np.int64)
    n_slates, N, _ = X.shape
    print(f"test slates {n_slates}  N={N}\n")

    per_slate, summary, diag = {}, {}, {}

    # ---- reference rows -------------------------------------------------
    ident = np.tile(np.arange(N), (n_slates, 1))          # BM25 first-stage order
    per_slate["bm25_firststage"] = score_rankings(Y, ident, TP)
    per_slate["teacher_gbt"] = score_rankings(Y, TP, TP)

    # ---- students -------------------------------------------------------
    for arm in ("M1", "M2", "M3"):
        model = load(arm, 20)
        raw = infer(model, X)
        if arm == "M3":
            rk = np.argsort(-raw, axis=1, kind="stable")
            per_slate[f"{arm}:sort"] = score_rankings(Y, rk, TP)
            continue
        coll = np.array([row_argmax_collisions(M)[0] for M in raw])
        r1 = np.array([rank1_energy(M) for M in raw])
        d1 = np.stack([d1_hungarian(M) for M in raw])
        d2 = np.stack([d2_row_greedy(M) for M in raw])
        diag[arm] = {
            "G1_mean_colliding_items": float(coll.mean()),
            "G1_frac_slates_with_any_collision": float((coll > 0).mean()),
            "G3_rank1_energy_mean": float(r1.mean()),
            "G3_rank1_energy_p5_p95": [float(np.percentile(r1, 5)), float(np.percentile(r1, 95))],
            "G2_frac_slates_D1_ne_D2": float(np.mean([not np.array_equal(x, y) for x, y in zip(d1, d2)])),
        }
        for dname, f in DECODERS.items():
            rk = np.stack([f(M) for M in raw])
            per_slate[f"{arm}:{dname}"] = score_rankings(Y, rk, TP)

    for k, v in per_slate.items():
        summary[k] = {m: float(np.nanmean(a)) for m, a in v.items()}

    # ---- pre-registered comparisons -------------------------------------
    comps = {}
    for arm in ("M1", "M2"):
        base = f"{arm}:D1_hungarian"
        for d in ("D2_row_greedy", "D3_col_greedy", "D4_expected_pos", "D5_col0"):
            for met in ("ndcg10", "ndcg1", "ndcg50", "recall10", "tau"):
                comps[f"{base} - {arm}:{d} [{met}]"] = paired_boot(
                    per_slate[base][met], per_slate[f"{arm}:{d}"][met], rng)
    for met in ("ndcg10", "ndcg1", "ndcg50", "recall10", "tau"):
        comps[f"M1:D1_hungarian - M3:sort [{met}]"] = paired_boot(
            per_slate["M1:D1_hungarian"][met], per_slate["M3:sort"][met], rng)
        comps[f"M1:D1_hungarian - teacher_gbt [{met}]"] = paired_boot(
            per_slate["M1:D1_hungarian"][met], per_slate["teacher_gbt"][met], rng)

    # ---- degradation sweep (control 3) ----------------------------------
    sweep = []
    for ep in (1, 2, 5, 10, 20):
        model = load("M1", ep)
        raw = infer(model, X)
        s1 = score_rankings(Y, np.stack([d1_hungarian(M) for M in raw]), TP)
        s2 = score_rankings(Y, np.stack([d2_row_greedy(M) for M in raw]), TP)
        coll = np.array([row_argmax_collisions(M)[0] for M in raw])
        r1 = np.array([rank1_energy(M) for M in raw])
        gap, lo, hi = paired_boot(s1["ndcg10"], s2["ndcg10"], rng, 2000)
        sweep.append({"epoch": ep, "ndcg10_D1": float(np.nanmean(s1["ndcg10"])),
                      "ndcg10_D2": float(np.nanmean(s2["ndcg10"])),
                      "gap": gap, "ci": [lo, hi],
                      "collisions": float(coll.mean()),
                      "rank1_energy": float(r1.mean())})

    # ---- report ---------------------------------------------------------
    print("DIAGNOSTICS (PLAN.md G1/G2/G3)")
    for arm, d in diag.items():
        print(f"  {arm}: {json.dumps(d)}")
    print("\nMEAN METRICS")
    hdr = ("ndcg1", "ndcg10", "ndcg50", "recall1", "recall10", "mrr", "tau")
    print(f"  {'arm:decoder':<26}" + "".join(f"{h:>10}" for h in hdr))
    for k in sorted(summary):
        print(f"  {k:<26}" + "".join(f"{summary[k][h]:10.4f}" for h in hdr))
    print("\nPAIRED BOOTSTRAP (mean diff [95% CI], 10k resamples over slates)")
    for k, (m, lo, hi) in comps.items():
        flag = "" if lo <= 0 <= hi else "  *"
        print(f"  {k:<52} {m:+.5f} [{lo:+.5f}, {hi:+.5f}]{flag}")
    print("\nDEGRADATION SWEEP (M1, D1 vs D2, NDCG@10)")
    for r in sweep:
        print(f"  ep{r['epoch']:<3} D1 {r['ndcg10_D1']:.4f}  D2 {r['ndcg10_D2']:.4f}  "
              f"gap {r['gap']:+.5f} [{r['ci'][0]:+.5f},{r['ci'][1]:+.5f}]  "
              f"collide {r['collisions']:5.2f}  rank1E {r['rank1_energy']:.4f}")

    (HERE / "out").mkdir(exist_ok=True)
    with open(HERE / "out" / "eval.json", "w") as fh:
        json.dump({"n_slates": int(n_slates), "N": int(N), "diagnostics": diag,
                   "summary": summary, "comparisons": comps, "sweep": sweep},
                  fh, indent=2)


if __name__ == "__main__":
    main()
