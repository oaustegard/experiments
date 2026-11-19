"""Inputs and ONNX Runtime outputs of indra-ocr's math reader on real line crops, for the
browser's check of its math nets (web/test/check_math.mjs).

Runs INSIDE scripts/jail.sh (it imports indra_ocr). One page: its line boxes (the package's
detector), the classifier's crops (the recognizer's, height 32) and probabilities, and the
crops of up to 8 flagged lines (height 64) through the encoder, the decoder's init and a
greedy decode of 24 steps (logits of steps 0, 1 and 23, every step's argmax and top
probability).

    sh scripts/jail.sh -C /tmp/indra-eval -e PYTHONPATH=/tmp/jail/indra/.deps -- \
        python3 run/make_math_fixtures.py arms/mix8m16 gray/PAGE.png fixtures/math
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from indra_ocr.line_mask import crop_lines
from indra_ocr.math import BOS
from indra_ocr.pipeline import OrtOCR
from indra_ocr.rec_preproc import pad_batch

models, page, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=True)
ocr = OrtOCR(models, threads=1, furniture=True, math=True)
m = ocr.math
gray = cv2.imread(str(page), cv2.IMREAD_GRAYSCALE)
boxes = ocr.detect(gray)
cases = {}


def save(name, arr, dt=np.float32):
    a = np.ascontiguousarray(arr, dt)
    (out / f"{name}.bin").write_bytes(a.tobytes())
    return {"file": f"{name}.bin", "shape": list(a.shape)}


crops = crop_lines(gray, boxes[:, :4], ocr.height, ocr.mask)
order = sorted(range(len(crops)), key=lambda i: crops[i].shape[1])
pick = order[:: max(1, len(order) // 12)][:12]
arr, widths = pad_batch([crops[j] for j in pick])
p = m.cls.run(None, {"x": arr, "widths": widths})[0]
cases["cls"] = {"x": save("cls_x", arr[:, 0]), "widths": widths.tolist(), "want": save("cls_p", p)}

flags = m.flags(crops)
sel = [j for j, f in enumerate(flags) if f]
sel = sorted(sel, key=lambda j: boxes[j, 2] - boxes[j, 0])[:: max(1, len(sel) // 8)][:8]
mc = crop_lines(gray, boxes[sel, :4], m.height, None)
arr, widths = pad_batch(mc)
mem, mask = m.enc.run(None, {"x": arr, "widths": widths})
ck, cv = m.init.run(None, {"memory": mem})
B = len(sel)
sk = np.zeros((m.layers, B, m.heads, 0, m.head_dim), np.float32)
sv = sk
tok = np.full(B, BOS, np.int64)
toks, tops, logit_steps = [], [], {}
for t in range(24):
    feed = {"token": tok, "pos": np.array(t, np.int64), "self_k": sk, "self_v": sv,
            "cross_k": ck, "cross_v": cv, "mask": mask}
    logits, sk, sv = m.step.run(None, {k: v for k, v in feed.items() if k in m.inputs["math_dec_step.onnx"]})
    if t in (0, 1, 23):
        logit_steps[t] = logits
    tok = logits.argmax(-1).astype(np.int64)
    e = np.exp(logits - logits.max(-1, keepdims=True))
    toks.append(tok.copy())
    tops.append(1.0 / e.sum(-1))
cases["math"] = {"x": save("math_x", arr[:, 0]), "widths": widths.tolist(), "memory": save("math_mem", mem),
                 "mask": save("math_mask", mask), "cross_k": save("math_ck", ck), "cross_v": save("math_cv", cv),
                 "logits0": save("math_l0", logit_steps[0]), "logits1": save("math_l1", logit_steps[1]),
                 "logits23": save("math_l23", logit_steps[23]),
                 "tokens": np.stack(toks).tolist(), "top": save("math_top", np.stack(tops))}
(out / "cases.json").write_text(json.dumps(cases, indent=1))
print("ok", {k: v.get("widths") for k, v in cases.items()}, "B", B, "T", mem.shape[1])
