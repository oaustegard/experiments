"""Per-query arrays and out-of-domain checks requested on the embed-coord board (#104).

Usage: python3 -I run/board_eval.py

1. results/<arm>_perquery.json for EG2, the A1 map, the A3 students (int8 ONNX) and A4 at
   each K: synth_hit10 (1,498 bools, synth_test file order) and real_overlap10 (1,000 floats,
   real_test order). ASYM: query encoder varies, docs are EG2 q4's committed codes.
2. results/ood.json: each A4 pruned vocabulary on real_test and on 2,000 general MS MARCO
   queries held out from every set here (seed 1, outside real_test and msmarco_train):
   share of queries tokenized intact, top-10 overlap with EG2, cosine to EG2 (mean, p05, min),
   and the five lowest-cosine queries.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pandas as pd
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_linear import fit  # noqa: E402
from common import OUT, QUERY, Index, Teacher, boot_ci, overlap  # noqa: E402
from final_eval import onnx_student  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "out" / "queries"
RES = ROOT / "results"


def rows(t):
    return [json.loads(l) for l in (Q / f"{t}.jsonl").open()]


def pruned(T, k):
    d = OUT / "a4" / f"k{k}"
    P = Teacher.__new__(Teacher)
    P.sess = ort.InferenceSession(str(d / "model_q4.onnx"), providers=["CPUExecutionProvider"])
    P.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
    P.specials = T.specials
    return P


def general_queries(n=2000):
    d = pd.read_parquet(ROOT / "out" / "msmarco_queries.parquet")
    used = {r["id"] for t in ("real_test", "msmarco_train") for r in rows(t)}
    pool = d[~d._id.astype(str).isin(used)]
    pick = pool.sample(n, random_state=1)
    return [{"id": str(i), "q": q} for i, q in zip(pick._id, pick.text)]


def main():
    ix, T = Index(), Teacher()
    data = {t: rows(t) for t in ("real_test", "synth_test")}
    E = {t: np.load(Q / f"{t}.eg2.npy") for t in data}
    top_t = {t: ix.topk(E[t]) for t in data}
    src = np.array([r["post"] for r in data["synth_test"]])

    def arrays(P):
        top_s, top_r = ix.topk(P["synth_test"]), ix.topk(P["real_test"])
        return {"synth_hit10": [bool(x) for x in (top_s == src[:, None]).any(1)],
                "real_overlap10": [float(x) for x in overlap(top_r, top_t["real_test"])]}

    arms = {"eg2_q4": E}
    S = OUT / "student" / "mdbr-leaf-mt"
    f = fit(np.load(S / "synth_train.npy"), np.load(Q / "synth_train.eg2.npy"), 1e-3)
    arms["a1_leaf_ridge"] = {t: f(np.load(S / f"{t}.npy")) for t in data}
    for ck, r in json.loads((RES / "export.json").read_text()).items():
        p = ROOT / ck
        arms[f"a3_{p.parent.name}_{p.name}_int8"] = {
            t: onnx_student(p, [x["q"] for x in data[t]], r["prompt"]) for t in data}
    ks = list(json.loads((RES / "a4.json").read_text())["k"])
    for k in ks:
        P = pruned(T, k)
        arms[f"a4_k{k}"] = {t: P.embed([x["q"] for x in data[t]]) for t in data}
    for name, P in arms.items():
        a = arrays(P)
        (RES / f"{name}_perquery.json").write_text(json.dumps(
            {"metric": "ASYM: query encoder varies, docs = EG2 q4 committed 2-bit codes", **a}))
        print(name, round(np.mean(a["synth_hit10"]), 4), round(np.mean(a["real_overlap10"]), 4), flush=True)

    gen = general_queries()
    Eg = T.embed([g["q"] for g in gen])
    top_g = ix.topk(Eg)
    sets = {"real_test": ([x["q"] for x in data["real_test"]], E["real_test"], top_t["real_test"]),
            "general_msmarco_2000": ([g["q"] for g in gen], Eg, top_g)}
    ood = {"general_msmarco_2000_ids": [g["id"] for g in gen], "k": {}}
    for k in ks:
        P = pruned(T, k)
        keep = set(P.tok.get_vocab().keys())
        r = {}
        for name, (qs, Ref, top_ref) in sets.items():
            V = P.embed(qs)
            cos = (V * Ref).sum(1)
            intact = np.mean([all(T.tok.id_to_token(i) in keep for i in T.tok.encode(QUERY + T.clean(q)).ids)
                              for q in qs])
            low = np.argsort(cos)[:5]
            r[name] = {"intact_share": float(intact), "overlap@10": boot_ci(overlap(ix.topk(V), top_ref)),
                       "cos_mean": float(cos.mean()), "cos_p05": float(np.quantile(cos, 0.05)),
                       "cos_min": float(cos.min()),
                       "lowest": [{"q": qs[i], "cos": round(float(cos[i]), 4)} for i in low]}
        ood["k"][k] = r
        print(k, json.dumps({n: {kk: (v if not isinstance(v, list) or kk == "overlap@10" else "...")
                                 for kk, v in x.items() if kk != "lowest"} for n, x in r.items()}), flush=True)
    (RES / "ood.json").write_text(json.dumps(ood, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
