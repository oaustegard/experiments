"""Fit each arm's Dense 256->768 to the centred teacher by ridge regression before training.

    python3 -I run/head_init.py   # .work/params/<arm>_h.f32, results/head_init.json

e17c's Dense maps into Arctic-m's space, so at step 0 its outputs are uncorrelated with MedCPT's and the
first few hundred steps rewrite the encoder to fit a random head (smoke tests 1-3). A least-squares head
on the arm's own pooled features, from 32K training rows, starts every arm aligned instead.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import QP, Embedder, from_flat, to_flat  # noqa: E402

W = HERE / ".work"
torch.set_num_threads(4)
ARMS = {"stock": (W / "e17c/epoch1/tokenizer.json", 50368), "ls32": (W / "tok/ls32/tokenizer.json", 32768), "ls50": (W / "tok/ls50/tokenizer.json", 50368)}
N, LAM = 32768, 1e-2


@torch.inference_mode()
def pooled(m, tok, texts, bs=128):
    tok.no_padding(); tok.enable_truncation(128)
    pad = tok.token_to_id("[PAD]")
    out = []
    for i in range(0, len(texts), bs):
        enc = tok.encode_batch(texts[i:i + bs])
        S = max(len(e.ids) for e in enc)
        ids = torch.full((len(enc), S), pad); att = torch.zeros((len(enc), S), dtype=torch.long)
        for b, e in enumerate(enc):
            ids[b, :len(e.ids)] = torch.tensor(e.ids); att[b, :len(e.ids)] = 1
        hs = m.hidden(ids, att)
        a = att.unsqueeze(-1).float()
        out.append(((hs * a).sum(1) / a.sum(1)).numpy())
    return np.concatenate(out)


def main():
    rows = [json.loads(l) for l in (W / "data/train.jsonl").open()]
    idx = np.sort(np.random.default_rng(1).choice(len(rows), N, replace=False))
    texts = [(QP if rows[i]["role"] == "q" else "") + rows[i]["t"] for i in idx]
    T = np.load(W / "teacher/teacher_train.f16.npy", mmap_mode="r")[idx].astype(np.float32) - np.load(W / "teacher/mu_all.npy")
    tr, te = slice(0, N - 2048), slice(N - 2048, N)
    res = {}
    for arm, (tp, V) in ARMS.items():
        flat = np.fromfile(W / f"params/{arm}.f32", np.float32)
        p = from_flat(flat, V)
        m = Embedder(p).eval()
        X = pooled(m, Tokenizer.from_file(str(tp)), texts)
        A = X[tr]
        Wd = np.linalg.solve(A.T @ A + LAM * len(A) * np.eye(A.shape[1]), A.T @ T[tr]).T   # [768, 256]
        cos = lambda Y, Z: float(np.mean(np.sum(Y * Z, 1) / np.linalg.norm(Y, axis=1) / np.linalg.norm(Z, axis=1)))  # noqa: E731
        res[arm] = {"heldout_cos_fit": cos(X[te] @ Wd.T, T[te]), "heldout_cos_e17c_head": cos(X[te] @ p["dense"].numpy().T, T[te])}
        p["dense"] = torch.from_numpy(Wd.astype(np.float32))
        to_flat(Embedder(p)).tofile(W / f"params/{arm}_h.f32")
        print(arm, res[arm], flush=True)
    (HERE / "results/head_init.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
