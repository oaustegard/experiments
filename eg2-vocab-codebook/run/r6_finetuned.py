"""Round 6: the run C-style caption fine-tune with each vocabulary, scored in its shipping codecs.

vc_trim: 41,741-row trimmed vocabulary (as shipped), fine-tuned from mac_k4.
vc_v64k: 67,347-row vocabulary (r5 v64k_mix), table initialised with learned rows, same recipe.
Usage: python3 -I run/r6_finetuned.py  -> results/r6_finetuned.json (+ per-query arrays in .work/r6_*.npz)
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, LS, WORK, Benches, rows_student  # noqa: E402
import rhq  # noqa: E402
from tcodecs import int_g32  # noqa: E402

torch.set_num_threads(4)
BUDGET = 10.0e6


def rhq_mix(X, f, n3):
    order = np.argsort(-f, kind="stable")
    R = np.empty_like(X)
    for sel, b in ((order[:n3], 3), (order[n3:], 2)):
        if len(sel):
            c, s = rhq.encode(X[sel], b)
            R[sel] = rhq.decode(c, s, b)
    return R, n3 * 194 + (len(X) - n3) * 130


def main():
    B = Benches()
    freq = np.load(WORK / "freq.npy")
    arms = {
        "vc_trim": (LS / "out/vocab/k32000", "cut"),
        "vc_v64k": (WORK / "vocab/v64k_mix", None),
    }
    p = HERE / "results" / "r6_finetuned.json"
    out = json.loads(p.read_text()) if p.exists() else {}
    for tag, (vd, vocab) in arms.items():
        ck = LS / "out/l1" / tag / "final.pt"
        if not ck.exists():
            print("missing", ck)
            continue
        sd = torch.load(ck, map_location="cpu")
        X = sd.pop("tm.embed_tokens.weight").numpy()
        rows = np.array(json.loads((vd / "rows.json").read_text()))
        tok = vocab or (tag, Tokenizer.from_file(str(vd / "tokenizer.json")))
        f = freq[rows].astype(np.float64)
        n3 = int((BUDGET - len(rows) * 130) // 64)
        codecs = {"float": lambda: (X, X.size * 2), "int3_g32": lambda: int_g32(X, 3), "rhq3": lambda: rhq_mix(X, f, len(X)),
                  "rhq_mix10": lambda: rhq_mix(X, f, max(0, min(len(X), n3))), "rhq2": lambda: rhq_mix(X, f, 0)}
        for cname, fn in codecs.items():
            key = f"{tag}/{cname}"
            if key in out:
                continue
            R, nb = fn()
            sd2 = dict(sd)
            sd2["tm.embed_tokens.weight"] = torch.from_numpy(np.ascontiguousarray(R, np.float32))
            m = rows_student({k: v for k, v in sd2.items() if k != "tm.embed_tokens.weight"} | {"tm.embed_tokens.weight": torch.zeros(41741, 512)}, rows, R)
            r, per = B.run(m, tok, per=True)
            r.update({"table_mb": round(nb / 1e6, 3), "n_rows": int(len(rows))})
            out[key] = r
            np.savez(WORK / f"r6_{tag}_{cname}.npz", **per)
            print(key, r, flush=True)
            p.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
