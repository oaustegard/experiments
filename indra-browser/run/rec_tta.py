"""Recognizer test-time augmentation: average CTC posteriors over perturbed crops.

indra-ocr's recognizers compute a log-softmax over classes per timestep and
export only what decoding needs from it: argmax ids, the peak probability, the
blank log-probability and the top-8 log-probabilities for the n-gram beam
search. `make` copies a model folder and adds that log-softmax to every
rec*.onnx as an output named "logp". `install` patches an OrtOCR's recognizer
sessions so each batch is read under several perturb.CROP variants of the same
crops; the probabilities are averaged per timestep (the crops share their
width, so the timesteps line up) and the six usual outputs are rebuilt from the
average. Greedy decoding, the confidence gate and the beam search then run on
the averaged posterior unchanged. Line boxes, the script classifier and reading
order are untouched.

    python3 run/rec_tta.py make --src /tmp/indra-eval/arms/v110 --out /tmp/indra-eval/arms/v110-logp
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np

LOGP = "logp"


def make(src: Path, out: Path):
    import onnx
    from onnx import TensorProto, helper

    out.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if f.is_file() and f.suffix in {".onnx", ".json", ".npz"}:
            dst = out / f.name
            if not f.name.startswith("rec") or f.suffix != ".onnx":
                shutil.copy2(f, dst)
                continue
            m = onnx.load(str(f))
            ls = [n for n in m.graph.node if n.op_type == "LogSoftmax"]
            if len(ls) != 1:
                raise SystemExit(f"{f.name}: {len(ls)} LogSoftmax nodes, expected 1")
            m.graph.node.append(helper.make_node("Identity", [ls[0].output[0]], [LOGP]))
            m.graph.output.append(helper.make_tensor_value_info(LOGP, TensorProto.FLOAT, None))
            onnx.save(m, str(dst))
            print(f"{f.name}: + {LOGP} <- {ls[0].output[0]}")


def _wrap(sess, fns):
    orig = sess.run

    def run(names, feeds, *a, **kw):
        x, widths = feeds["x"], feeds["widths"]
        acc = None
        lengths = k = None
        for i, fn in enumerate(fns):
            xi = np.ascontiguousarray(fn(x), dtype=np.float32)
            if i == 0:
                lp, lengths, topv = orig([LOGP, "lengths", "topv"], {"x": xi, "widths": widths})
                k = topv.shape[-1]
            else:
                (lp,) = orig([LOGP], {"x": xi, "widths": widths})
            p = np.exp(lp.astype(np.float64))
            acc = p if acc is None else acc + p
        p = acc / len(fns)
        logp = np.log(np.maximum(p, 1e-30))
        ids = logp.argmax(-1).astype(np.int64)
        conf = p.max(-1).astype(np.float32)
        blank = logp[..., 0].astype(np.float32)
        topi = np.argsort(-logp, axis=-1, kind="stable")[..., :k].astype(np.int64)
        topv = np.take_along_axis(logp, topi, -1).astype(np.float32)
        have = {"ids": ids, "conf": conf, "lengths": lengths, "blank": blank,
                "topv": topv, "topi": topi, LOGP: logp.astype(np.float32)}
        if names is None:
            names = [o.name for o in sess.get_outputs()]
        return [have[n] for n in names]

    sess.run = run


def install(ocr, names: list[str]):
    import perturb as PT

    fns = [PT.CROP[n] for n in names]
    sessions = [ocr.rec] + [s for s, _ in getattr(ocr, "groups", {}).values()]
    for s in sessions:
        if LOGP not in {o.name for o in s.get_outputs()}:
            raise SystemExit(f"recognizer without a {LOGP} output: run rec_tta.py make on the models")
        _wrap(s, fns)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("--src", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "make":
        make(a.src, a.out)


if __name__ == "__main__":
    main()
