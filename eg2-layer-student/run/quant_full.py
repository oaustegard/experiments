"""Full EG2 (all 24 layers, 32K vocabulary cut) with its transformer weights quantized below
4 bits: does it keep EG2's space at a smaller download?

Usage: python3 -I run/quant_full.py [--configs rtn:4:32 rtn:3:32 rtn:2:32 rtn:2:16 ...]

Start: the q4 teacher's own weights (run/model.py), so rtn:4:32 should reproduce EG2. Each
config re-quantizes every Linear in the layers, the PLE projection and the output projection
(the token table stays at the teacher's q4) as asymmetric blockwise weights along the input
dimension, simulated in fp32 ("fake quant"), and runs the bench. Method:
  rtn    round to nearest, min/max per block
  gptq   GPTQ (Frantar et al. 2022): column-by-column rounding with the error spread over the
         columns not yet rounded, from each layer's input Hessian on calibration queries
  higgs  HIGGS-style (Malinovskii et al. 2024; the lineage in remex-vs-higgs-ablation): a
         seeded randomized Hadamard rotation along the input dimension, then per block the
         Lloyd-Max levels of N(0,1) times a scale picked by MSE, then the rotation undone.
         It is a download format: decoded to q4 or fp16 at load, so no runtime kernel needed
  gq     both: GPTQ rounding onto the higgs grid, in the rotated basis (W R^T, R H R^T), each
         block's scale fixed by MSE when the block starts (the QuIP# combination)
Bytes for rtn/gptq use onnxruntime MatMulNBits' layout (b bits per weight, an fp32 scale and a
b-bit zero point per block); for higgs b bits per weight and an fp16 scale per block. Plus the
q4 table and tokenizer. Writes results/quant_full.json.
"""
import argparse
import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import TESTS, Bench  # noqa: E402
from l1_distill import train_texts  # noqa: E402
from model import ROOT, Student, batches, embed, tokenize  # noqa: E402
from common import Teacher  # noqa: E402

VOCAB = ROOT / "out" / "vocab" / "k32000"


def targets(m):
    return [(n, mod) for n, mod in m.tm.named_modules()
            if isinstance(mod, torch.nn.Linear) and n != "embed_tokens"]


def rtn(w, bits, block):
    out, cols = w.shape
    W = w.reshape(out, cols // block, block)
    lo, hi = W.min(-1, keepdim=True).values, W.max(-1, keepdim=True).values
    qmax = 2 ** bits - 1
    scale = ((hi - lo) / qmax).clamp(min=1e-12)
    zp = torch.round(-lo / scale).clamp(0, qmax)
    q = torch.clamp(torch.round(W / scale) + zp, 0, qmax)
    return ((q - zp) * scale).reshape(out, cols)


def gptq(w, H, bits, block, damp=0.01):
    """GPTQ with per-block (along columns) asymmetric grids fixed when a block starts."""
    W = w.clone().double()
    cols = W.shape[1]
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    qmax = 2 ** bits - 1
    Q = torch.zeros_like(W)
    for c0 in range(0, cols, block):
        c1 = c0 + block
        Wb = W[:, c0:c1]
        lo, hi = Wb.min(1, keepdim=True).values, Wb.max(1, keepdim=True).values
        scale = ((hi - lo) / qmax).clamp(min=1e-12)
        zp = torch.round(-lo / scale).clamp(0, qmax)
        Err = torch.zeros_like(Wb)
        for j in range(block):
            col = Wb[:, j:j + 1]
            q = (torch.clamp(torch.round(col / scale) + zp, 0, qmax) - zp) * scale
            Q[:, c0 + j] = q[:, 0]
            e = (col - q) / Hinv[c0 + j, c0 + j]
            Wb[:, j:] -= e @ Hinv[c0 + j:c0 + j + 1, c0 + j:c1]
            Err[:, j] = e[:, 0]
        W[:, c1:] -= Err @ Hinv[c0:c1, c1:]
    return Q.float()


def lloyd_max(bits, iters=200):
    """MSE-optimal scalar levels for N(0,1) (Max 1960), by Lloyd iteration on the density."""
    from math import erf, exp, pi, sqrt
    pdf = lambda x: exp(-x * x / 2) / sqrt(2 * pi)  # noqa: E731
    cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731
    n = 2 ** bits
    lv = np.linspace(-2, 2, n)
    for _ in range(iters):
        edges = np.concatenate([[-np.inf], (lv[1:] + lv[:-1]) / 2, [np.inf]])
        lv = np.array([(pdf(a) - pdf(b)) / max(cdf(b) - cdf(a), 1e-12) for a, b in zip(edges[:-1], edges[1:])])
    return torch.tensor(lv, dtype=torch.float32)


def rht(n, seed):
    """Randomized Hadamard: H_n diag(signs) / sqrt(n), n a power of two; orthogonal."""
    H = torch.ones(1, 1)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    signs = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], n)).float()
    return H * signs / n ** 0.5


def higgs(w, bits, block, seed):
    out, cols = w.shape
    R = rht(cols, seed)
    Wr = (w @ R.T).reshape(out, cols // block, block)  # rotated weights, blocks along input
    lv = lloyd_max(bits)
    rms = Wr.pow(2).mean(-1, keepdim=True).sqrt().clamp(min=1e-12)
    best, best_err = None, None
    for f in np.linspace(0.6, 1.4, 17):  # per-block scale by MSE over a small grid
        sc = (rms * f).half().float()  # stored as fp16
        q = lv[(Wr / sc).unsqueeze(-1).sub(lv).abs().argmin(-1)] * sc
        err = (q - Wr).pow(2).sum(-1, keepdim=True)
        if best is None:
            best, best_err = q, err
        else:
            better = err < best_err
            best = torch.where(better, q, best)
            best_err = torch.minimum(err, best_err)
    return best.reshape(out, cols) @ R


def gptq_grid(w, H, bits, block, seed, damp=0.01):
    """GPTQ in the randomized-Hadamard basis with Lloyd-Max levels and an fp16 scale per block."""
    cols = w.shape[1]
    R = rht(cols, seed).double()
    W = (w.double() @ R.T)
    H = R @ H.double() @ R.T
    H += damp * torch.mean(torch.diag(H)) * torch.eye(cols, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    lv = lloyd_max(bits).double()
    Q = torch.zeros_like(W)
    for c0 in range(0, cols, block):
        c1 = c0 + block
        Wb = W[:, c0:c1]
        rms = Wb.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        best_s, best_e = None, None
        for f in np.linspace(0.6, 1.4, 17):
            sc = (rms * f).half().double()
            e = (lv[(Wb / sc).unsqueeze(-1).sub(lv).abs().argmin(-1)] * sc - Wb).pow(2).sum(1, keepdim=True)
            best_s = sc if best_s is None else torch.where(e < best_e, sc, best_s)
            best_e = e if best_e is None else torch.minimum(e, best_e)
        Err = torch.zeros_like(Wb)
        for j in range(block):
            col = Wb[:, j:j + 1]
            q = lv[(col / best_s).unsqueeze(-1).sub(lv).abs().argmin(-1)] * best_s
            Q[:, c0 + j] = q[:, 0]
            e = (col - q) / Hinv[c0 + j, c0 + j]
            Wb[:, j:] -= e @ Hinv[c0 + j:c0 + j + 1, c0 + j:c1]
            Err[:, j] = e[:, 0]
        W[:, c1:] -= Err @ Hinv[c0:c1, c1:]
    return (Q @ R).float()


def hessians(m, enc, batch=64):
    """X^T X of each target Linear's input over the calibration tokens (padding excluded)."""
    H, hooks, mask_now = {}, [], {}
    for n, mod in targets(m):
        H[n] = torch.zeros(mod.in_features, mod.in_features, dtype=torch.float64)

        def hook(mod, inp, out, n=n):
            x = inp[0].reshape(-1, inp[0].shape[-1])[mask_now["m"]].double()
            H[n] += x.T @ x
        hooks.append(mod.register_forward_hook(hook))
    with torch.no_grad():
        for _, ids, mask in batches(enc, batch):
            mask_now["m"] = mask.reshape(-1).bool()
            m(ids, mask)
    for h in hooks:
        h.remove()
    return H


def nbytes(m, bits, block, method="rtn"):
    """rtn/gptq: MatMulNBits bytes (b-bit weights, fp32 scale, b-bit zp per block);
    higgs, gq: b-bit codes and an fp16 scale per block."""
    total = 0
    for _, mod in targets(m):
        n = mod.weight.numel()
        total += n * bits / 8 + n / block * (2 if method in ("higgs", "gq") else 4 + bits / 8)
    return int(total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", nargs="+",
                    default=["rtn:4:32", "rtn:3:32", "rtn:2:32", "rtn:2:16", "gptq:3:32", "gptq:2:32", "gptq:2:16"])
    ap.add_argument("--calib", type=int, default=2048, help="training queries for the GPTQ Hessians")
    a = ap.parse_args()
    torch.set_num_threads(4)
    T, B = Teacher(), Bench()
    rows = json.loads((VOCAB / "rows.json").read_text())
    tok = Tokenizer.from_file(str(VOCAB / "tokenizer.json"))
    m = Student(24, rows)
    m.eval()
    orig = {n: mod.weight.detach().clone() for n, mod in targets(m)}
    enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    out = ROOT / "results" / "quant_full.json"
    res = json.loads(out.read_text()) if out.exists() else {}
    # Fixed sizes from the measured k = 24 q4 file components: the q4 32K table (13.69 MB,
    # results/sizes.json), norms, graph and gzip tokenizer (about 1 MB together).
    sizes = json.loads((ROOT / "results" / "sizes.json").read_text())["l2_k4"]
    fixed = sizes["embed_tokens"]["bytes"] + sizes["tokenizer_gz"]["bytes"] + 1_000_000
    H = None
    if "unquantized (q4 teacher weights)" not in res:
        r, _ = B.score({t: embed(m, enc[t]) for t in TESTS})
        res["unquantized (q4 teacher weights)"] = {"bytes": fixed + nbytes(m, 4, 32), **r}
        out.write_text(json.dumps(res, indent=1))
    for cfg in a.configs:
        if cfg in res:
            continue
        method, bits, block = cfg.split(":")
        bits, block = int(bits), int(block)
        t0 = time.time()
        if method in ("gptq", "gq") and H is None:
            calib, _ = train_texts(["synth_train", "msmarco_train"])
            pick = np.random.default_rng(0).choice(len(calib), a.calib, replace=False)
            for n, mod in targets(m):
                mod.weight.data.copy_(orig[n])
            H = hessians(m, tokenize(T, [calib[i] for i in pick], remap=tok))
        with torch.no_grad():
            for n, mod in targets(m):
                w = orig[n]
                if method == "rtn":
                    q = rtn(w, bits, block)
                elif method == "gptq":
                    q = gptq(w, H[n], bits, block)
                elif method == "gq":
                    q = gptq_grid(w, H[n], bits, block, seed=zlib.crc32(n.encode()))
                else:
                    q = higgs(w, bits, block, seed=zlib.crc32(n.encode()))  # stable across runs, unlike hash()
                mod.weight.data.copy_(q)
        r, _ = B.score({t: embed(m, enc[t]) for t in TESTS})
        nb = fixed + nbytes(m, bits, block, method)
        res[cfg] = {"method": method, "bits": bits, "block": block, "bytes": nb,
                    "mb": round(nb / 1e6, 1), **r, "seconds": round(time.time() - t0)}
        print(cfg, f"{res[cfg]['mb']} MB", json.dumps({t: round(r[t]["overlap@10"][0], 3) for t in TESTS}),
              "hit", round(r["synth_test"]["hit@10"][0], 3), flush=True)
        out.write_text(json.dumps(res, indent=1))
    for n, mod in targets(m):
        mod.weight.data.copy_(orig[n])


if __name__ == "__main__":
    main()
