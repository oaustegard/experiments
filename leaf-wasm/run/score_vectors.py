"""Score vectors a browser page computed (out/gpu/) on the eg2-native-small Bluesky task.

Usage: python3 -I run/score_vectors.py NAME     (reads out/gpu/NAME.docs.f32, NAME.queries.f32)

docs.f32 = 61,686 x 768 float32 (post order, no prompt), queries.f32 = 1,498 x 768 (synth_test
order, leaf-ir query prompt). SYM hit@10 like eg2-native-small/run/native.py: fp32 exact cosine
and the arm's own remex 2-bit codes (rx2_topk), paired against leaf-ir-wasm-i8 (the same int8
weights through ORT) and EG2. Writes results/<NAME>.json and results/<NAME>_perquery.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NAT = ROOT.parent / "eg2-native-small"
sys.path.insert(0, str(NAT / "run"))
from native import boot_ci, paired, rx2_topk, topk  # noqa: E402


def main():
    name = sys.argv[1]
    X = np.fromfile(ROOT / "out" / "gpu" / f"{name}.docs.f32", np.float32).reshape(-1, 768)
    Q = np.fromfile(ROOT / "out" / "gpu" / f"{name}.queries.f32", np.float32).reshape(-1, 768)
    synth = [json.loads(l) for l in (NAT / "out" / "q" / "synth_test.jsonl").open()]
    assert X.shape[0] == 61686 and Q.shape[0] == len(synth), (X.shape, Q.shape)
    src = np.array([r["post"] for r in synth])
    ref = json.loads((NAT / "results" / "leaf-ir-wasm-i8_perquery.json").read_text())
    eg2 = np.array(json.loads((NAT / "results" / "EG2_perquery.json").read_text())["EG2 q4 rx2"], float)
    ort_docs = NAT / "out" / "leaf-ir-wasm-i8" / "docs.npy"
    res, per = {"n_docs": len(X), "n_queries": len(Q)}, {"ids": ref["ids"]}
    if ort_docs.exists():  # same weights through ORT: how close are the doc vectors themselves?
        c = (X * np.load(ort_docs)).sum(1)
        res["doc_cos_vs_ort_same_weights"] = {"min": float(c.min()), "p001": float(np.quantile(c, 0.001)), "median": float(np.median(c))}
    for rep in ("fp32", "rx2"):
        t = topk(Q @ X.T) if rep == "fp32" else rx2_topk(X, Q)[0]
        h = (t == src[:, None]).any(1).astype(float)
        per[rep] = h.astype(int).tolist()
        res[rep] = {"hit@10": boot_ci(h), "minus_leaf-ir-wasm-i8": paired(h, np.array(ref[rep], float)),
                    "minus_eg2": paired(h, eg2), "discordant_vs_wasm_i8": int((h != np.array(ref[rep])).sum())}
        print(rep, json.dumps(res[rep]))
    (ROOT / "results" / f"{name}.json").write_text(json.dumps(res, indent=1))
    (ROOT / "results" / f"{name}_perquery.json").write_text(json.dumps(per))


if __name__ == "__main__":
    main()
