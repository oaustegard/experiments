"""The bridge job's generation loop, in Python on onnxruntime CPU: the reference the WebGPU run is
checked against, and the definition of what the job does.

Batched greedy decoding over the exported graph (export.py). Prompts are RIGHT-padded:
GroupQueryAttention reads each row's length from the attention mask (seqlens_k = sum - 1), keeps
that row's valid keys at the start of the cache and writes the next token at its own position,
so rows of different lengths share one batch. Finished rows keep decoding pad tokens, ignored.

Stop: EOS (id 2), or "<|im_end|>" appearing in the decoded tail. Decoding uses a per-token piece
table (pieces.json, tok.decode([i]) for every id), exactly as the JS job does.
"""
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort

HERE = Path(__file__).resolve().parent
PAD, EOS, END = 3, 2, "<|im_end|>"


def pieces(tok):
    return [tok.decode([i]) for i in range(tok.vocab_size if hasattr(tok, "vocab_size") else 8192)]


def session(model_dir, threads=4):
    so = ort.SessionOptions()
    so.intra_op_num_threads = threads
    return ort.InferenceSession(str(Path(model_dir) / "model.onnx"), so, providers=["CPUExecutionProvider"])


def generate(sess, prompts, max_new, piece, record_logits=0, stop=True):
    layers = sum(1 for i in sess.get_inputs() if i.name.endswith(".key"))
    B = len(prompts)
    lens = np.array([len(p) for p in prompts], dtype=np.int64)
    T = int(lens.max())
    ids = np.full((B, T), PAD, dtype=np.int64)
    mask = np.zeros((B, T), dtype=np.int64)
    for b, p in enumerate(prompts):
        ids[b, :len(p)] = p
        mask[b, :len(p)] = 1
    feed = {"input_ids": ids, "attention_mask": mask,
            "logits_index": np.arange(B, dtype=np.int64) * T + lens - 1}
    for l in range(layers):
        feed[f"past_key_values.{l}.key"] = np.zeros((B, 4, 0, 64), np.float32)
        feed[f"past_key_values.{l}.value"] = np.zeros((B, 4, 0, 64), np.float32)
    names = [o.name for o in sess.get_outputs()]
    out_toks = [[] for _ in range(B)]
    tails = [""] * B
    done = np.zeros(B, bool)
    stopped = np.zeros(B, bool)
    cur = lens.copy()
    total = T
    kept = []
    t0 = time.perf_counter()
    step_ms = []
    for step in range(max_new + 1):
        ts = time.perf_counter()
        res = dict(zip(names, sess.run(None, feed)))
        step_ms.append((time.perf_counter() - ts) * 1000)
        logits = res["logits"][:, 0, :]
        if step < record_logits:
            kept.append(logits.copy())
        nxt = logits.argmax(-1)
        if step == max_new:
            break
        for b in range(B):
            if done[b]:
                nxt[b] = PAD
                continue
            t = int(nxt[b])
            if stop and t == EOS:
                done[b] = stopped[b] = True
                continue
            out_toks[b].append(t)
            tails[b] = (tails[b] + piece[t])[-32:]
            if stop and END in tails[b]:
                done[b] = stopped[b] = True
        if done.all():
            break
        total += 1
        cur = cur + 1
        mask = (np.arange(total)[None, :] < cur[:, None]).astype(np.int64)
        feed = {"input_ids": nxt[:, None].astype(np.int64), "attention_mask": mask,
                "logits_index": np.arange(B, dtype=np.int64)}
        for l in range(layers):
            feed[f"past_key_values.{l}.key"] = res[f"present.{l}.key"]
            feed[f"past_key_values.{l}.value"] = res[f"present.{l}.value"]
    return {"tokens": out_toks, "stopped": stopped.tolist(), "ms": (time.perf_counter() - t0) * 1000,
            "prefill_ms": step_ms[0], "step_ms": step_ms[1:], "logits": kept}
