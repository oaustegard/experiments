"""How far can LFM3-1.7B be quantized below 4 bits? Builds weight variants and scores each against
the bf16 release.

  python3 lfm-wgsl/run/quant3.py data       # tokenized calibration / eval sets, bf16 chat replies
  python3 lfm-wgsl/run/quant3.py gptq       # per-layer Hessians from bf16 activations, rotated GPTQ at 2/3/4 bits
  python3 lfm-wgsl/run/quant3.py eval [V..] # score variants (default: all) into results/quant3.json

Every stage checkpoints per unit (a layer, a variant) under model3/quant/ and skips finished units.

Formats (all quantize every matrix of the 24 layers; the tied token table / lm_head is 4-bit RTN
g32 in every variant except `onnx-q4`, which is onnx-community's file as shipped):
  rtnB      round to nearest, asymmetric uniform grid, block 32, fp32 scale + B-bit zero point
            (MatMulNBits, what the engine runs today at B = 4)
  gqB       GPTQ in a randomized-Hadamard basis (one 256-wide block rotation, applied to the input
            dimension) onto Lloyd-Max levels for N(0,1), fp16 scale per block of 32
            (METHODS.md: best below 4 bits on EmbeddingGemma 2)
  mixA_B    gqA on the feed-forward matrices (w1, w3, w2: 1.13B of the 1.47B layer weights), gqB elsewhere

Scores, on held-out WikiText-2 test text (8 x 512 tokens) and on the bf16 model's own greedy replies
to 8 voice-assistant prompts: KL(bf16 || variant) per token over the bf16 top 256, top-1 agreement
with bf16, and perplexity of the text.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lfm3 import M3, ROOT, Lfm3, bf16, config, linear_names, onnx_q4, rms  # noqa: E402

torch.set_num_threads(4)
Q = M3 / "quant"
OUT = ROOT / "results" / "quant3.json"
CAL_SEQS, EVAL_SEQS, SEQ = 48, 8, 512
ROT = 256
BLOCK = 32
SYSTEM = "You are a voice assistant. Answer in one to three short spoken sentences. Plain text only."
PROMPTS = ["Why is the sky blue?", "What should I cook for dinner tonight?", "How far away is the moon?",
           "Tell me a fun fact about octopuses.", "What's a good way to fall asleep faster?",
           "Explain what a black hole is.", "Hello there, how are you this evening?",
           "Give me a tip for learning a new language."]


# ---------- quantizers ----------

def hadamard(n):
    H = torch.ones(1, 1, dtype=torch.float64)
    while H.shape[0] < n:
        H = torch.cat([torch.cat([H, H], 1), torch.cat([H, -H], 1)], 0)
    return H


def rotation256(seed=0):
    """The randomized Hadamard block H diag(s) / 16, applied along K in 256-wide blocks (block-diagonal,
    orthogonal). The engine applies it to activations with a 256-point fast Walsh-Hadamard transform."""
    s = torch.from_numpy(np.random.default_rng(seed).choice([-1.0, 1.0], ROT))
    return hadamard(ROT) * s / ROT ** 0.5


def rot_w(W, inverse=False):
    """W [out, K] times R^T (into the rotated basis), or times R (back out) with inverse."""
    R = rotation256().to(W.dtype)
    out, K = W.shape
    Wb = W.reshape(out, K // ROT, ROT)
    return (Wb @ (R if inverse else R.T)).reshape(out, K)


def rot_h(H):
    """R H R^T for the block-diagonal R, block by block."""
    R = rotation256().to(H.dtype)
    K = H.shape[0]; nb = K // ROT
    Hb = H.reshape(nb, ROT, nb, ROT)
    return torch.einsum("ab,ibjc,dc->iajd", R, Hb, R).reshape(K, K)


def lloyd_max(bits, iters=300):
    from math import erf, exp, pi, sqrt
    pdf = lambda x: exp(-x * x / 2) / sqrt(2 * pi)  # noqa: E731
    cdf = lambda x: 0.5 * (1 + erf(x / sqrt(2)))  # noqa: E731
    lv = np.linspace(-2, 2, 2 ** bits)
    for _ in range(iters):
        e = np.concatenate([[-np.inf], (lv[1:] + lv[:-1]) / 2, [np.inf]])
        lv = np.array([(pdf(a) - pdf(b)) / max(cdf(b) - cdf(a), 1e-12) for a, b in zip(e[:-1], e[1:])])
    return torch.tensor(lv, dtype=torch.float64)


def rtn(w, bits):
    out, K = w.shape
    W = w.float().reshape(out, K // BLOCK, BLOCK)
    lo, hi = W.min(-1, keepdim=True).values, W.max(-1, keepdim=True).values
    qmax = 2 ** bits - 1
    s = ((hi - lo) / qmax).clamp(min=1e-12)
    z = torch.round(-lo / s).clamp(0, qmax)
    return ((torch.clamp(torch.round(W / s) + z, 0, qmax) - z) * s).reshape(out, K)


def gq(w, H, bits, damp=0.01):
    """GPTQ in the rotated basis onto Lloyd-Max levels with an fp16 scale per block of 32.
    Returns (codes uint8 [out, K], scales fp16 [out, K/32]); dequantize with deq_gq."""
    K = w.shape[1]
    W = rot_w(w.double())
    H = rot_h(H.double())
    dead = torch.diag(H) == 0
    H[dead, dead] = 1
    H += damp * torch.mean(torch.diag(H)) * torch.eye(K, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    lv = lloyd_max(bits)
    codes = torch.zeros(W.shape, dtype=torch.uint8)
    scales = torch.zeros(W.shape[0], K // BLOCK, dtype=torch.float16)
    for c0 in range(0, K, BLOCK):
        c1 = c0 + BLOCK
        Wb = W[:, c0:c1]
        rmsb = Wb.pow(2).mean(1, keepdim=True).sqrt().clamp(min=1e-12)
        best_s, best_e = None, None
        for f in np.linspace(0.6, 1.6, 21):
            sc = (rmsb * f).half().double()
            e = (lv[(Wb / sc).unsqueeze(-1).sub(lv).abs().argmin(-1)] * sc - Wb).pow(2).sum(1, keepdim=True)
            best_s = sc if best_s is None else torch.where(e < best_e, sc, best_s)
            best_e = e if best_e is None else torch.minimum(e, best_e)
        scales[:, c0 // BLOCK] = best_s[:, 0].half()
        Err = torch.zeros_like(Wb)
        for j in range(BLOCK):
            col = Wb[:, j:j + 1]
            idx = (col / best_s).unsqueeze(-1).sub(lv).abs().argmin(-1)
            codes[:, c0 + j] = idx[:, 0].to(torch.uint8)
            e = (col - lv[idx] * best_s) / Hinv[c0 + j, c0 + j]
            Wb[:, j:] -= e @ Hinv[c0 + j:c0 + j + 1, c0 + j:c1]
            Err[:, j] = e[:, 0]
        W[:, c1:] -= Err @ Hinv[c0:c1, c1:]
    return codes, scales


def deq_gq(codes, scales, bits):
    lv = lloyd_max(bits).float()
    out, K = codes.shape
    Wr = lv[codes.long()].reshape(out, K // BLOCK, BLOCK) * scales.float()[..., None]
    return rot_w(Wr.reshape(out, K), inverse=True)


# ---------- data ----------

def chat_ids(tok, user, reply_ids=()):
    import jinja2
    tpl = (M3 / "bf16" / "chat_template.jinja").read_text()
    env = jinja2.Environment(); env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
    s = env.from_string(tpl).render(messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
                                    add_generation_prompt=True, enable_thinking=False, bos_token="<|startoftext|>")
    return tok.encode(s, add_special_tokens=False).ids + list(reply_ids)


def stage_data():
    import pandas as pd
    from tokenizers import Tokenizer
    Q.mkdir(parents=True, exist_ok=True)
    tok = Tokenizer.from_file(str(M3 / "tokenizer.json"))
    cfg = config()
    out = {}
    for split, n in (("train", CAL_SEQS), ("test", EVAL_SEQS)):
        text = "".join(pd.read_parquet(M3 / "data" / f"wt2-{split}.parquet").text)
        ids = tok.encode(text, add_special_tokens=False).ids
        step = len(ids) // n
        out[split] = [[cfg["bos_token_id"]] + ids[i * step:i * step + SEQ - 1] for i in range(n)]
    p = Q / "chat.json"
    if not p.exists():
        m = Lfm3(bf16())
        eos = cfg["eos_token_id"]
        chats = []
        for u in PROMPTS:
            ids = chat_ids(tok, u)
            n0 = len(ids)
            for _ in range(48):
                nxt = int(m.logits(ids, last=1)[0].argmax())
                if nxt == eos:
                    break
                ids.append(nxt)
            chats.append({"prompt": u, "ids": ids, "reply_start": n0, "reply": tok.decode(ids[n0:])})
            print(u, "->", chats[-1]["reply"], flush=True)
        p.write_text(json.dumps(chats, indent=1))
    (Q / "data.json").write_text(json.dumps(out))
    print({k: len(v) for k, v in out.items()}, "sequences of", SEQ)


# ---------- GPTQ over layers ----------

def stage_gptq(bits_list=(2, 3, 4)):
    cfg = config()
    data = json.loads((Q / "data.json").read_text())["train"]
    W = bf16()
    m = Lfm3(W)
    xs = [W["model.embed_tokens.weight"][torch.tensor(s)].float() for s in data]
    names = linear_names(cfg)
    for i in range(len(cfg["layer_types"])):
        done = Q / f"layer{i}.done"
        layer_names = [(n, t) for n, t in names if n.startswith(f"model.layers.{i}.")]
        if not done.exists():
            t0 = time.time()
            Hs, cnt = {}, {}

            def hook(tag, x):
                x = x.double()
                Hs[tag] = Hs.get(tag, 0) + x.T @ x
                cnt[tag] = cnt.get(tag, 0) + x.shape[0]
            m.hook = hook
            with torch.no_grad():
                for x in xs:
                    m.layer(i, x)
            m.hook = None
            # Linears sharing an input each fire the hook; the mean over calls is the same Hessian
            for t in Hs:
                Hs[t] /= cnt[t]
            for b in bits_list:
                for n, t in layer_names:
                    codes, scales = gq(W[n].float(), Hs[t], b)
                    torch.save({"codes": codes, "scales": scales}, Q / f"gq{b}.{n}.pt")
            done.write_text(f"{time.time() - t0:.0f} s")
            print(f"layer {i}: {time.time() - t0:.0f} s", flush=True)
        with torch.no_grad():
            xs = [m.layer(i, x) for x in xs]


# ---------- evaluation ----------

def variant_weights(name, W):
    """Weights dict for a variant: W (bf16 release) with the layer matrices and the token table replaced."""
    cfg = config()
    if name == "onnx-q4":
        return {k: v.half() for k, v in onnx_q4().items()}, True
    V = dict(W)
    V["model.embed_tokens.weight"] = rtn(W["model.embed_tokens.weight"], 4).half()
    for n, _ in linear_names(cfg):
        ffn = ".feed_forward." in n
        if name == "bf16":
            continue
        if name.startswith("rtn"):
            V[n] = rtn(W[n], int(name[3:])).half()
            continue
        b = int(name[2:]) if name.startswith("gq") else int(name[3:].split("_")[0 if ffn else 1])
        d = torch.load(Q / f"gq{b}.{n}.pt")
        V[n] = deq_gq(d["codes"], d["scales"], b).half()
    if name == "bf16":
        V["model.embed_tokens.weight"] = W["model.embed_tokens.weight"]
    return V, False


def topk_ref(m, seqs, k=256, chat=None):
    out = []
    for j, s in enumerate(seqs):
        lp = F.log_softmax(m.logits(s), -1)
        if chat:
            lp = lp[chat[j] - 1:-1]
        v, idx = lp.topk(k, -1)
        out.append((idx, v, None))
    return out


def score(m, seqs, ref, chat=None):
    kl, agree, nll, n = 0.0, 0, 0.0, 0
    for j, (s, (idx, lpr, _)) in enumerate(zip(seqs, ref)):
        lp = F.log_softmax(m.logits(s), -1)
        tgt = torch.tensor(s[1:])
        if chat:
            lp, tgt = lp[chat[j] - 1:-1], tgt[chat[j] - 1:]
        else:
            lp = lp[:-1]
        idx, lpr = (idx, lpr) if chat else (idx[:-1], lpr[:-1])
        p = lpr.exp()
        kl += (p * (lpr - lp.gather(-1, idx))).sum().item()
        agree += (lp.argmax(-1) == idx[:, 0]).sum().item()
        nll -= lp.gather(-1, tgt[:, None]).sum().item()
        n += lp.shape[0]
    return {"kl": kl / n, "top1": agree / n, "ppl": float(np.exp(nll / n)), "tokens": n}


def bits_of(name, cfg, W):
    """Stored bits per weight over the 24 layers, and total MB including the 4-bit token table."""
    tot_w, tot_b = 0, 0
    for n, _ in linear_names(cfg):
        k = W[n].numel()
        ffn = ".feed_forward." in n
        if name in ("bf16",):
            b = 16
        elif name.startswith("rtn") or name == "onnx-q4":
            bb = 4 if name == "onnx-q4" else int(name[3:])
            b = bb + (32 + bb) / BLOCK
        else:
            bb = int(name[2:]) if name.startswith("gq") else int(name[3:].split("_")[0 if ffn else 1])
            b = bb + 16 / BLOCK
        tot_w += k; tot_b += k * b
    emb = W["model.embed_tokens.weight"].numel() * (16 if name == "bf16" else 4 + 36 / BLOCK)
    return tot_b / tot_w, (tot_b + emb) / 8 / 1e6


def stage_eval(variants):
    cfg = config()
    data = json.loads((Q / "data.json").read_text())
    chats = json.loads((Q / "chat.json").read_text())
    W = bf16()
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    refp = Q / "ref.pt"
    if refp.exists():
        ref = torch.load(refp)
    else:
        m = Lfm3(W)
        ref = {"text": topk_ref(m, data["test"]), "chat": topk_ref(m, [c["ids"] for c in chats], chat=[c["reply_start"] for c in chats])}
        torch.save(ref, refp)
    for v in variants:
        if v in res:
            continue
        t0 = time.time()
        V, ort = variant_weights(v, W)
        m = Lfm3(V, ort=ort)
        bpw, mb = bits_of(v, cfg, W)
        res[v] = {"bits_per_weight": round(bpw, 3), "mb": round(mb, 1),
                  "text": score(m, data["test"], ref["text"]),
                  "chat": score(m, [c["ids"] for c in chats], ref["chat"], chat=[c["reply_start"] for c in chats])}
        print(v, json.dumps(res[v]), f"{time.time() - t0:.0f} s", flush=True)
        OUT.parent.mkdir(exist_ok=True)
        OUT.write_text(json.dumps(res, indent=1))
        del V, m


VARIANTS = ["bf16", "onnx-q4", "rtn4", "gq4", "rtn3", "gq3", "rtn2", "gq2", "mix2_4", "mix3_4", "mix2_3"]

if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "data":
        stage_data()
    elif stage == "gptq":
        stage_gptq()
    elif stage == "eval":
        stage_eval(sys.argv[2:] or VARIANTS)
