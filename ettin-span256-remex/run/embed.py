"""Step 1: embed the Bluesky corpus and the two query sets with the shipped Ettin-17m ONNX.

Usage: python3 -I run/embed.py

Model: release ettin-native-20261021, onnx-ettin17m-tuned-t4-w8L63.zip (the package the
embedders page runs), unzipped to out/rel/onnx/. The graph ends in mean pooling, then one
bias-free MatMul 256 -> 768 whose int8 weight is dequantized per output column; the caller
L2-normalizes. This script adds the pooled 256-d tensor as a second graph output, so the
head can be replaced afterwards without running the encoder again.

Writes out/vec/{docs,synth,real}.x.npy (unit rows, 768) and .p.npy (pooled, 256),
out/W.npy (the dequantized head weight, [256, 768]) and out/model_pooled.onnx.
"""
import gzip
import json
import time
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "out"
SRC = OUT / "rel" / "onnx"
POOLED = "/st.1/Concat_output_0"
HEAD = "onnx::MatMul_1417"


def head_weight(model):
    """The dequantized 256 x 768 head weight, read from the graph's own initializers."""
    init = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
    last = model.graph.node[-1]
    assert last.op_type == "MatMul" and list(last.input) == [POOLED, HEAD], list(last.input)
    deq = next(n for n in model.graph.node if HEAD in n.output)
    assert deq.op_type == "DequantizeLinear" and deq.attribute[0].i == 1
    q, s = init[deq.input[0]], init[deq.input[1]]
    assert q.dtype == np.int8 and q.shape == (256, 768) and s.shape == (768,)
    return q.astype(np.float32) * s[None, :], q, s


def embed(sess, tok, texts, prompt, batch=64):
    enc = [e.ids for e in tok.encode_batch([prompt + t for t in texts])]
    order = np.argsort([len(e) for e in enc])
    X = np.empty((len(texts), 768), np.float32)
    P = np.empty((len(texts), 256), np.float32)
    for s in range(0, len(order), batch):
        rows = order[s:s + batch]
        S = max(len(enc[r]) for r in rows)
        ids = np.zeros((len(rows), S), np.int64)
        attn = np.zeros((len(rows), S), np.int64)
        for b, r in enumerate(rows):
            ids[b, :len(enc[r])] = enc[r]
            attn[b, :len(enc[r])] = 1
        e, p = sess.run(["sentence_embedding", POOLED], {"input_ids": ids, "attention_mask": attn})
        X[rows] = e / np.linalg.norm(e, axis=1, keepdims=True)
        P[rows] = p
    return X, P


def main():
    model = onnx.load(str(SRC / "model.onnx"))
    W, _, _ = head_weight(model)
    np.save(OUT / "W.npy", W)
    model.graph.output.append(helper.make_tensor_value_info(POOLED, TensorProto.FLOAT, [None, 256]))
    onnx.save(model, str(OUT / "model_pooled.onnx"))

    # ORT's extended optimizations rewrite the DequantizeLinear -> MatMul pairs and the output
    # moves (cosine about 0.9999 to the graph as written). BASIC keeps the graph's own
    # arithmetic, int8 code times column scale, which is also what the WGSL engine computes.
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
    sess = ort.InferenceSession(str(OUT / "model_pooled.onnx"), so, providers=["CPUExecutionProvider"])
    so_all = ort.SessionOptions()
    so_all.intra_op_num_threads = 4
    sess_all = ort.InferenceSession(str(OUT / "model_pooled.onnx"), so_all, providers=["CPUExecutionProvider"])
    tok = Tokenizer.from_file(str(SRC / "tokenizer.json"))
    meta = json.loads((SRC / "meta.json").read_text())
    tok.no_padding()
    tok.enable_truncation(meta["max_seq_length"])
    qp, dp = meta["prompts"]["query"], meta["prompts"]["document"]

    with gzip.open(OUT / "corpus.json.gz") as f:
        corpus = json.load(f)
    texts = [t or "" for t in corpus["texts"]]
    assert len(texts) == 61686
    (OUT / "vec").mkdir(exist_ok=True)
    jobs = [("docs", texts, dp)]
    for name in ("synth", "real"):
        rows = [json.loads(l) for l in (OUT / "q" / f"{name}_test.jsonl").open()]
        jobs.append((name, [r["q"] for r in rows], qp))
    log = {"onnxruntime": ort.__version__, "optimization": "ORT_ENABLE_BASIC", "sets": {}}
    for name, tx, prompt in jobs:
        t0 = time.time()
        X, P = embed(sess, tok, tx, prompt)
        # the pooled output and the head weight must reproduce the model's own output
        E = P @ W
        err = float(np.abs(E / np.linalg.norm(E, axis=1, keepdims=True) - X).max())
        assert err < 1e-5, err
        np.save(OUT / "vec" / f"{name}.x.npy", X)
        np.save(OUT / "vec" / f"{name}.p.npy", P)
        secs = time.time() - t0
        Xa, _ = embed(sess_all, tok, tx[:2000], prompt)
        cos = (Xa * X[:2000]).sum(1)
        log["sets"][name] = {"n": len(tx), "seconds": round(secs), "head_parity_max_abs": err,
                             "cos_to_ORT_ENABLE_ALL": {"n": len(cos), "min": float(cos.min()), "median": float(np.median(cos))}}
        print(name, json.dumps(log["sets"][name]), flush=True)
    (HERE / "results" / "embed.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    main()
