"""One table over the final arms, with paired comparisons and a task-quality metric.

Usage: python3 -I run/final_eval.py

Arms: EG2 (the teacher, reference), A1 = best ridge map (mdbr-leaf-mt fit on synth_train),
A3 = the int8 ONNX of each exported student (results/export.json), A4 = pruned EG2 at each K.
Per arm and test set: top-10 overlap with EG2 (bootstrap CI), and on synth_test the share
of queries whose source post is in the top 10 (hit@10: each synthetic query was written
from one post, so that post is a known relevant result), with its paired difference
against EG2's own hit@10. Writes results/final.json.
"""
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from a1_linear import fit  # noqa: E402
from common import OUT, Index, Teacher, boot_ci, overlap  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
Q = ROOT / "out" / "queries"
TESTS = ["real_test", "synth_test"]


def rows(t):
    return [json.loads(l) for l in (Q / f"{t}.jsonl").open()]


def paired(x, y, n=2000, seed=0):
    d = x - y
    rng = np.random.default_rng(seed)
    m = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return float(d.mean()), float(np.quantile(m, 0.025)), float(np.quantile(m, 0.975))


def onnx_student(ck, texts, prompt):
    sess = ort.InferenceSession(str(ck / "onnx" / "model_int8.onnx"), providers=["CPUExecutionProvider"])
    tok = AutoTokenizer.from_pretrained(str(ck))
    out = []
    for s in range(0, len(texts), 128):
        enc = tok([prompt + t for t in texts[s:s + 128]], padding=True, truncation=True,
                  max_length=128, return_tensors="np")
        out.append(sess.run(["embedding"], {"input_ids": enc["input_ids"].astype(np.int64),
                                            "attention_mask": enc["attention_mask"].astype(np.int64)})[0])
    P = np.concatenate(out).astype(np.float32)
    return P / np.linalg.norm(P, axis=1, keepdims=True)


def main():
    ix = Index()
    data = {t: rows(t) for t in TESTS}
    E = {t: np.load(Q / f"{t}.eg2.npy") for t in TESTS}
    top_t = {t: ix.topk(E[t]) for t in TESTS}
    src = np.array([r["post"] for r in data["synth_test"]])
    hit_t = (top_t["synth_test"] == src[:, None]).any(1).astype(float)
    preds = {}

    S = OUT / "student" / "mdbr-leaf-mt"
    f = fit(np.load(S / "synth_train.npy"), np.load(Q / "synth_train.eg2.npy"), 1e-3)
    preds["A1 leaf-mt ridge"] = ({t: f(np.load(S / f"{t}.npy")) for t in TESTS}, None)
    exp = json.loads((ROOT / "results" / "export.json").read_text())
    for ck, r in exp.items():
        p = ROOT / ck
        preds[f"A3 {p.parent.name} {p.name} int8"] = (
            {t: onnx_student(p, [x["q"] for x in data[t]], r["prompt"]) for t in TESTS},
            r["int8_bytes"] + sum(r["tokenizer_gz_bytes"].values()))
    a4 = json.loads((ROOT / "results" / "a4.json").read_text())
    T = Teacher()
    for k, r in a4["k"].items():
        d = OUT / "a4" / f"k{k}"
        P = Teacher.__new__(Teacher)
        P.sess = ort.InferenceSession(str(d / "model_q4.onnx"), providers=["CPUExecutionProvider"])
        P.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        P.specials = T.specials
        preds[f"A4 pruned EG2 K={k}"] = ({t: P.embed([x["q"] for x in data[t]]) for t in TESTS},
                                         r["onnx_bytes"] + r["tokenizer_gz_bytes"])

    res = {"EG2 (teacher)": {"bytes": a4["full"]["onnx_bytes"] + a4["full"]["tokenizer_gz_bytes"],
                             "synth_test": {"hit@10": boot_ci(hit_t)}}}
    for name, (P, nbytes) in preds.items():
        r = {"bytes": nbytes}
        for t in TESTS:
            top = ix.topk(P[t])
            r[t] = {"overlap@10": boot_ci(overlap(top, top_t[t]))}
            if t == "synth_test":
                h = (top == src[:, None]).any(1).astype(float)
                r[t]["hit@10"] = boot_ci(h)
                r[t]["hit@10_minus_eg2"] = paired(h, hit_t)
        res[name] = r
        print(name, json.dumps(r), flush=True)
    (ROOT / "results" / "final.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
