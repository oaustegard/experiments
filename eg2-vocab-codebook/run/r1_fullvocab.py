"""Round 1: does the full vocabulary help the k=4 cut with no retraining, float table?

Usage: python3 -I run/r1_fullvocab.py
Arms: cut tokenizer + trained 41,741-row table (as shipped, float); full tokenizer + full table
(EG2 q4 rows, trained rows written over the kept ids); full tokenizer + EG2's own rows everywhere.
Writes results/r1_fullvocab.json and per-query arrays in .work/r1_*.npz.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, LS, WORK, Benches, cut_rows, cut_student, full_student, teacher_table  # noqa: E402

torch.set_num_threads(4)


def main():
    B = Benches()
    sd = torch.load(LS / ".work/mac/out/l1/mac_k4_v32k/final.pt", map_location="cpu")
    T = teacher_table()
    rows = cut_rows()
    tr = sd["tm.embed_tokens.weight"].numpy()
    t0 = np.asarray(T[rows])
    cos = (tr * t0).sum(1) / np.linalg.norm(tr, axis=1) / np.linalg.norm(t0, axis=1)
    res = {"trained_vs_eg2_row_cos": {"mean": float(cos.mean()), "p01": float(np.quantile(cos, .01))}}
    print(res, flush=True)
    arms = {
        "cut": lambda: (cut_student(sd), "cut"),
        "full": lambda: (full_student(sd), "full"),
        "full_eg2rows": lambda: (full_student(sd, table=np.array(T)), "full"),
    }
    for name, f in arms.items():
        m, v = f()
        out, per = B.run(m, v, per=True)
        res[name] = out
        np.savez(WORK / f"r1_{name}.npz", **per)
        print(name, out, flush=True)
        del m
    (HERE / "results" / "r1_fullvocab.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
