"""Step 2: fold the 256 -> 768 head into a 256 -> 256 head and check nothing changes.

Usage: python3 -I run/fold.py

The model ends in e = p W (p pooled, 256-d; W the dequantized weight, [256, 768]) and the
caller L2-normalizes. With W = A diag(S) B^T, the head H = A diag(S) gives z = p H = e B.
B has orthonormal columns, so unit(z) and unit(e) have the same inner products.

Checks, on the 61,686 posts and both query sets:
  span       how far the 768-d outputs lie outside B, and the head's singular values
  fp32       score and top-10 agreement between 768-d and folded 256-d float search
  controls   what the folded head is not: the pooled state with no head, and the first 256
             of the 768 coordinates
  int8       the folded head stored the way the package stores weights (int8 codes in +-63,
             one scale per output column), in the SVD basis and with a remex rotation folded in;
             then every int8 head against the checkpoint's fp32 head, which is what the shipped
             768 head was rounded from
  onnx       a real graph with the final MatMul swapped, run by onnxruntime, against the
             numpy head
Writes results/span.json, out/basis.npy, out/vec/*.y.npy (exact fold of the shipped head, SVD
basis), out/vec/*.yq.npy (the head to deploy: remex rotation folded in, int8 +-127 rounded
from the checkpoint; rows are already rotated) and out/model_folded.onnx (that head).
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import remex
from onnx import helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (
    OUT,
    RES,
    SETS,
    boot,
    hits,
    int8_cols,
    load,
    recall,
    svd_head,
    topk,
    unit,
)
from embed import HEAD, POOLED


def agreement(Xd, Xq, Yd, Yq, src):
    """768-d float search against another float space: scores, top 10, hit@10."""
    out = {}
    for k in SETS:
        t0 = topk(lambda q: q @ Xd.T, Xq[k])
        t1 = topk(lambda q: q @ Yd.T, Yq[k])
        r = {"top10_recall": float(recall(t1, t0).mean()),
             "top10_identical_lists": int((t0 == t1).all(1).sum()), "n": len(t0)}
        if k == "synth":
            r["hit_at_10_768"] = float(hits(t0, src).mean())
            r["hit_at_10"] = boot(hits(t1, src))
        out[k] = r
    return out


def main():
    X, P, W, src = load()
    H, B, S = svd_head(W)
    np.save(OUT / "basis.npy", B.astype(np.float32))
    res = {"head": {"shape": list(W.shape), "singular_max": float(S[0]), "singular_min": float(S[-1]),
                    "condition": float(S[0] / S[-1])}}

    # span: the outputs lie in B's column space, so 256 numbers hold all of each vector
    ev = np.linalg.eigvalsh((X["docs"].astype(np.float64).T @ X["docs"]) / len(X["docs"]))[::-1]
    sv = np.sqrt(np.clip(ev, 0, None))
    res["span"] = {"docs_sv_ratio_256": float(sv[255] / sv[0]), "docs_sv_ratio_257": float(sv[256] / sv[0])}
    for k, x in X.items():
        x64 = x.astype(np.float64)
        res["span"][f"outside_basis_max_{k}"] = float(np.linalg.norm(x64 - (x64 @ B) @ B.T, axis=1).max())

    # The pooled states themselves have rank 255: the final LayerNorm has no bias, so it removes
    # each token's mean before scaling, and mean pooling keeps that one linear constraint.
    psv = np.linalg.svd(P["docs"][:8000].astype(np.float64), compute_uv=False)
    res["span"]["pooled_sv_ratio_255"] = float(psv[254] / psv[0])
    res["span"]["pooled_sv_ratio_256"] = float(psv[255] / psv[0])

    # fp32 folded head: y = unit(p H)
    Y = {k: unit(P[k].astype(np.float64) @ H).astype(np.float32) for k in P}
    for k, v in Y.items():
        np.save(OUT / "vec" / f"{k}.y.npy", v)
    res["fp32_folded"] = agreement(X["docs"], X, Y["docs"], Y, src)
    res["fp32_folded"]["max_abs_coord_diff_to_xB"] = float(max(np.abs(Y[k] - X[k] @ B).max() for k in Y))
    d = [np.abs(X[k][s:s + 250] @ X["docs"].T - Y[k][s:s + 250] @ Y["docs"].T).max()
         for k in SETS for s in range(0, len(X[k]), 250)]
    res["fp32_folded"]["max_abs_score_diff"] = float(max(d))

    # controls
    Pn = {k: unit(P[k]) for k in P}
    res["control_pooled_no_head"] = agreement(X["docs"], X, Pn["docs"], Pn, src)
    Xp = {k: unit(X[k][:, :256]) for k in X}
    res["control_prefix256"] = agreement(X["docs"], X, Xp["docs"], Xp, src)

    # int8 folded heads. R is remex's rotation for d=256 at the seed the index uses, folded in.
    R = remex.Quantizer(256, bits=2, seed=42).R.astype(np.float64)
    heads = {"svd_basis": H, "remex_rotated": H @ R.T}
    res["int8_folded"] = {}
    for name, Hx in heads.items():
        q, s = int8_cols(Hx)
        Hq = q.astype(np.float64) * s
        Yf = {k: unit(P[k].astype(np.float64) @ Hx) for k in P}
        Yq = {k: unit(P[k].astype(np.float64) @ Hq).astype(np.float32) for k in P}
        cos = np.concatenate([(Yf[k] * Yq[k]).sum(1) for k in P])
        r = agreement(X["docs"], X, Yq["docs"], Yq, src)
        r["cos_to_fp32_head"] = {"min": float(cos.min()), "median": float(np.median(cos))}
        r["column_scale_ratio_max_over_min"] = float(s.max() / s.min())
        res["int8_folded"][name] = r
    # The shipped 768 head is itself an int8 rounding of the checkpoint's fp32 head. Judge every
    # int8 head against that fp32 head (same pooled states), so a folded head is compared with
    # what it replaces and not charged for the rounding the package already carries.
    from safetensors.numpy import load_file
    W0 = load_file(str(OUT / "rel" / "st" / "epoch1" / "2_Dense" / "model.safetensors"))["linear.weight"].T.astype(np.float64)
    H0, _, _ = svd_head(W0)
    ref = {k: unit(P[k].astype(np.float64) @ W0).astype(np.float32) for k in P}
    cands = {"shipped_int8_768_L63": lambda: X}
    for lv in (63, 127):
        for nm, Hx in (("svd_basis", H0), ("remex_rotated", H0 @ R.T)):
            q, s = int8_cols(Hx, lv)
            cands[f"folded_int8_256_L{lv}_{nm}"] = (lambda q=q, s=s: {k: unit(P[k].astype(np.float64) @ (q.astype(np.float64) * s)).astype(np.float32) for k in P})
    # the head to deploy: rotation folded in, int8 +-127, rounded once from the fp32 checkpoint
    q_rot, s_rot = int8_cols(H0 @ R.T, 127)
    for k, v in cands["folded_int8_256_L127_remex_rotated"]().items():
        np.save(OUT / "vec" / f"{k}.yq.npy", v)
    cands["folded_fp32_256"] = lambda: {k: unit(P[k].astype(np.float64) @ H0).astype(np.float32) for k in P}
    res["vs_fp32_checkpoint_head"] = {n: agreement(ref["docs"], ref, (v := f())["docs"], v, src) for n, f in cands.items()}
    res["head_bytes"] = {"shipped_int8_768": 256 * 768 + 768 * 4, "folded_int8_256": 256 * 256 + 256 * 4,
                         "folded_fp32_256": 256 * 256 * 4}

    # the same head as a real graph: swap the final MatMul's weight, run it, compare
    m = onnx.load(str(OUT / "rel" / "onnx" / "model.onnx"))
    deq = next(n for n in m.graph.node if HEAD in n.output)
    for name in list(deq.input):
        m.graph.initializer.remove(next(i for i in m.graph.initializer if i.name == name))
    m.graph.initializer.extend([numpy_helper.from_array(q_rot, deq.input[0]), numpy_helper.from_array(s_rot, deq.input[1])])
    m.graph.output[0].type.tensor_type.shape.dim[-1].dim_value = 256
    m.graph.output.append(helper.make_tensor_value_info(POOLED, onnx.TensorProto.FLOAT, [None, 256]))
    onnx.checker.check_model(m)
    onnx.save(m, str(OUT / "model_folded.onnx"))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(OUT / "model_folded.onnx"), so, providers=["CPUExecutionProvider"])
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(OUT / "rel" / "onnx" / "tokenizer.json"))
    tok.no_padding()
    tok.enable_truncation(128)
    meta = json.loads((OUT / "rel" / "onnx" / "meta.json").read_text())
    worst = 0.0
    for k in SETS:
        rows = [json.loads(l)["q"] for l in (OUT / "q" / f"{k}_test.jsonl").open()]
        for i in range(0, len(rows), 97):  # every 97th query, one at a time (no padding involved)
            ids = np.array([tok.encode(meta["prompts"]["query"] + rows[i]).ids], np.int64)
            e = sess.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids)})[0]
            assert e.shape == (1, 256)
            worst = max(worst, float(np.abs(unit(e)[0] - np.load(OUT / "vec" / f"{k}.yq.npy")[i]).max()))
    res["onnx_folded"] = {"max_abs_diff_to_numpy_head": worst, "queries_checked": len(range(0, 1498, 97)) + len(range(0, 1000, 97)),
                          "model_bytes": (OUT / "model_folded.onnx").stat().st_size,
                          "shipped_model_bytes": (OUT / "rel" / "onnx" / "model.onnx").stat().st_size}
    assert worst < 1e-5, worst

    (RES / "span.json").write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
