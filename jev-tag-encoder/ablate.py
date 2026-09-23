"""Step 5 (phrasing ablation, first 50 arXiv test papers) and step 6 (determinism, 30 mixed docs)."""
import numpy as np

import jev
from common import DATA, arxiv, boot, ci, f1s, fmt, jsonl, label_scores, save

VARIANTS = ["about", "mentions", "substantially"]


def phrasing():
    ids, Y, _ = arxiv("test")
    ids, Y = ids[:50], Y[:50]
    P = {v: jev.matrix("arxiv", v, ids) for v in VARIANTS}
    ok = np.all([~np.isnan(P[v]).any(1) for v in VARIANTS], axis=0)
    print(f"phrasing: {ok.sum()}/50 papers encoded under all three phrasings")
    P = {v: m[ok] for v, m in P.items()}
    Y = Y[ok]
    n = len(Y)
    out = {"n": n}
    tags = np.array(jev.TAGS, dtype=object)
    for v in VARIANTS:
        act = (P[v] >= 0.5).sum(1)
        S = label_scores(P[v])
        f = f1s(Y, S >= 0.5)
        bs = boot(lambda i, S=S: f1s(Y[i], S[i] >= 0.5), n, seed=3)
        c = ci(bs)
        # mapped-label tags only: how often does a gold label's tag fire (recall-side)?
        gold_hit = float((S >= 0.5)[Y].mean())
        # off-label firing: fraction of docs with >= 1 non-gold label firing
        off = float(((S >= 0.5) & ~Y).any(1).mean())
        act_ci = ci(boot(lambda i, act=act: act[i].mean(), n, seed=3))[0]
        out[v] = {"active_mean": float(act.mean()), "active_ci": act_ci,
                  "mean_prob": float(P[v].mean()), "micro_f1": f[0], "macro_f1": f[1], "f1_ci": c,
                  "gold_label_recall": gold_hit, "docs_with_offlabel_fire": off, "_boot": bs}
        print(f"{v:<14} active/doc {act.mean():.2f}  gold-label recall {gold_hit:.3f}  off-label docs {off:.2f}  "
              f"micro-F1 {fmt(f[0], c[0])}  macro-F1 {fmt(f[1], c[1])}")
    for v in VARIANTS[1:]:
        d = out[v]["_boot"][:, 0] - out["about"]["_boot"][:, 0]
        dc = ci(d[:, None])[0]
        out[v]["delta_micro_f1_vs_about"] = [out[v]["micro_f1"] - out["about"]["micro_f1"]] + dc
        print(f"  {v} - about micro-F1: {out[v]['micro_f1'] - out['about']['micro_f1']:+.3f} [{dc[0]:+.3f}, {dc[1]:+.3f}]")
        shift = (P[v] >= 0.5).mean(0) - (P["about"] >= 0.5).mean(0)
        top = np.argsort(-shift)[:8]
        out[v]["top_fire_shift"] = [(tags[i], round(float(shift[i]), 3)) for i in top]
        print(f"  largest fire-rate gains under '{v}': {out[v]['top_fire_shift']}")
    for v in VARIANTS:
        out[v].pop("_boot")
    return out


def determinism():
    ids = [r["id"] for r in jsonl(DATA / "mixed.jsonl")][:30]
    A, B = jev.matrix("mixed", "about", ids), jev.matrix("mixed_rerun", "about", ids)
    ok = ~np.isnan(A).any(1) & ~np.isnan(B).any(1)
    A, B = A[ok], B[ok]
    d = np.abs(A - B)
    flips = (A >= 0.5) != (B >= 0.5)
    rerun = jev.load("mixed_rerun", "about")
    out = {"n": int(ok.sum()), "mean_abs_delta": float(d.mean()), "max_abs_delta": float(d.max()),
           "frac_identical": float((d == 0).mean()), "bit_flips_total": int(flips.sum()),
           "docs_with_flip": int(flips.any(1).sum()), "values_changed": int((d > 0).sum()),
           "cache_status": sorted({str(r.get("cache")) for r in rerun.values()})}
    if flips.any():
        out["flip_examples"] = [(jev.TAGS[j], float(A[i, j]), float(B[i, j])) for i, j in zip(*np.where(flips))][:10]
    print("determinism:", out)
    return out


if __name__ == "__main__":
    save("step5_phrasing", phrasing())
    save("step6_determinism", determinism())
