"""Quantization-aware training of the cut EG2 student's token table (experiments#103 students).

Usage: python3 -I run/qat.py --k 2 --init PATH/final.pt --bits 2 [--steps N] [--tag NAME]
       python3 -I run/qat.py --k 2 --init PATH/final.pt --bits 0 --steps 0   # evaluate only

Starts from an eg2-layer-student checkpoint trained on #102's 32K vocabulary cut (41,741
rows) and continues training on the same 164,588 targets, loss 1 - cos(student, EG2), lr 1e-3
with 100 warmup steps and linear decay, batch 64, batch order seed 2 (neither earlier epoch
replays it). With --bits b > 0 the table is fake-quantized in every forward pass: asymmetric
min/max per group of 32 values (the layout of onnxruntime's GatherBlockQuantized), rounded
with a straight-through estimator, so the fp32 latent table learns rows that survive
rounding. --bits 0 is the control: the same extra training with a float table.

Each finished run is scored on the #103 bench (top-10 overlap with EG2 q4 on real_test and
synth_test, synth hit@10 and its paired difference to EG2) under a grid: table at float /
4 / 3 / 2 bits, transformer linears float or 4-bit round-to-nearest in groups of 32 along the
input dimension (what MatMulNBits stores in the shipped q4 file). Writes results/qat_<tag>.json.
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer
from torch.nn.utils import parametrize

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LS = ROOT.parent / "eg2-layer-student"
sys.path.insert(0, str(LS / "run"))
from bench import TESTS, Bench  # noqa: E402
from l1_distill import SETS, epoch_batches, train_texts  # noqa: E402
from model import Student, batches, embed, tokenize  # noqa: E402
from common import Teacher  # noqa: E402

VOCAB = LS / "out" / "vocab" / "k32000"


def fake_quant(W, bits, g=32):
    """Asymmetric min/max per group of g along each row; returns the dequantized tensor."""
    G = W.reshape(-1, g)
    lo, hi = G.min(1, keepdim=True).values, G.max(1, keepdim=True).values
    s = ((hi - lo) / (2 ** bits - 1)).clamp_min(1e-12)
    return (torch.round((G - lo) / s) * s + lo).reshape(W.shape)


class STEQuant(torch.nn.Module):
    def __init__(self, bits):
        super().__init__()
        self.bits = bits

    def forward(self, W):
        return W + (fake_quant(W, self.bits) - W).detach()


def quantized_copy(m, table_bits, layers_q4):
    q = copy.deepcopy(m)
    with torch.no_grad():
        if table_bits:
            w = q.tm.embed_tokens.weight
            w.copy_(fake_quant(w, table_bits))
        if layers_q4:
            for name, mod in q.tm.named_modules():
                if isinstance(mod, torch.nn.Linear):
                    mod.weight.copy_(fake_quant(mod.weight, 4))
    return q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--init", required=True)
    ap.add_argument("--bits", type=int, default=0)
    ap.add_argument("--steps", type=int, help="default one epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--tag")
    ap.add_argument("--save-every", type=int, default=300)
    a = ap.parse_args()
    tag = a.tag or f"k{a.k}_b{a.bits}"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    T = Teacher()
    rows = json.loads((VOCAB / "rows.json").read_text())
    tok = Tokenizer.from_file(str(VOCAB / "tokenizer.json"))
    m = Student(a.k, rows)
    sd = torch.load(a.init)
    w = sd.get("tm.embed_tokens.weight")
    if w is not None and len(w) != len(rows):
        sd["tm.embed_tokens.weight"] = w[torch.as_tensor(rows)]
    assert not m.load_state_dict(sd, strict=False).unexpected_keys
    assert len(m.tm.embed_tokens.weight) == len(rows)
    B = Bench()
    test_enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    work = ROOT / ".work" / "qat" / tag
    work.mkdir(parents=True, exist_ok=True)
    resp = ROOT / "results" / f"qat_{tag}.json"
    res = {"k": a.k, "init": a.init, "bits": a.bits, "lr": a.lr, "batch": a.batch, "seed": a.seed,
           "table_rows": len(rows), "evals": {}}

    texts, Y = train_texts(SETS)
    enc = tokenize(T, texts, remap=tok)
    spe = (len(texts) + a.batch - 1) // a.batch
    total = spe if a.steps is None else a.steps
    res |= {"n_train": len(texts), "steps": total}
    if a.bits:
        parametrize.register_parametrization(m.tm.embed_tokens, "weight", STEQuant(a.bits))
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / max(total, 1)))
    step = 0
    if (work / "pos.json").exists():
        m.load_state_dict(torch.load(work / "weights.pt"))
        opt.load_state_dict(torch.load(work / "opt.pt"))
        step = json.loads((work / "pos.json").read_text())["step"]
        for _ in range(step):
            sched.step()
        print("resumed at", step, flush=True)

    def save():
        torch.save(m.state_dict(), work / "weights.pt")
        torch.save(opt.state_dict(), work / "opt.pt")
        (work / "pos.json").write_text(json.dumps({"step": step}))

    Yt = torch.from_numpy(Y)
    lens = np.array([len(e) for e in enc])
    plan = epoch_batches(lens, a.batch, a.seed) if total else []
    t0, tot, n = time.time(), 0.0, 0
    while step < total:
        b = plan[step % spe]
        m.train()
        _, ids, mask = next(batches(enc, len(b), order=b))
        loss = (1 - (m(ids, mask) * Yt[torch.as_tensor(b)]).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        tot, n = tot + loss.item(), n + 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {tot / n:.5f} {(time.time() - t0) / n:.2f}s/step", flush=True)
            tot, n, t0 = 0.0, 0, time.time()
        if step % a.save_every == 0:
            save()

    if a.bits:
        parametrize.remove_parametrizations(m.tm.embed_tokens, "weight", leave_parametrized=False)
    for table_bits in [0, 4, 3, 2]:
        for layers_q4 in [False, True]:
            q = quantized_copy(m, table_bits, layers_q4)
            r, per = B.score({t: embed(q, test_enc[t]) for t in TESTS})
            name = f"table{table_bits or 'fp'}_layers{'q4' if layers_q4 else 'fp'}"
            res["evals"][name] = r
            np.savez(work / f"per_{name}.npz", **per)
            resp.write_text(json.dumps(res, indent=1))
            print(name, json.dumps({t: round(r[t]["overlap@10"][0], 4) for t in TESTS}),
                  "hit", round(r["synth_test"]["hit@10"][0], 4), flush=True)
    torch.save({n: t for n, t in m.state_dict().items()}, work / "final.pt")
    print("done", flush=True)


if __name__ == "__main__":
    main()
