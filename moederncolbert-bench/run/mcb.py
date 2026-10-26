#!/usr/bin/env python3
"""MoEdernColBERT forward pass reimplemented from the safetensors weights (no remote code).

Mirrors modeling_modernbert_hash_moe_v10.py (VAGOsolutions) for inference on CPU:
hash embedding -> LayerNorm, then per layer either a gated short conv (115M only) or
RMSNorm-attention (GQA, RoPE) + RMSNorm-MoE (top-2 of 8 GeGLU experts + one shared
expert), final RMSNorm, ColBERT Dense to 128, L2 per token.

Two query/document formats, because the reference pipeline and the model code disagree:
  prefix=False  what sentence-transformers 6.2 emits for this checkpoint: no [Q]/[D] marker,
                query padded to 32 with [MASK] that the other tokens do not attend to.
  prefix=True   PyLate-style: [CLS] [Q] ... [SEP] [MASK]*, and the model's own rule (a query
                row has 30000 at position 1) makes every token attend to every other.
"""
from __future__ import annotations

import json
import string
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from tokenizers import Tokenizer

HERE = Path(__file__).resolve().parent
MODELS = HERE.parent / ".work" / "models"
Q_ID, D_ID, MASK_ID, PAD_ID, CLS_ID, SEP_ID = 30000, 30001, 4, 0, 2, 3


def rms(x, w, eps):
    return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))


class MoEdernColBERT:
    def __init__(self, name: str, query_length=32, document_length=512, device="cpu", root=None):
        d = Path(root or MODELS) / name
        self.dev = device
        self.cfg = json.load(open(d / "config.json"))
        self.w = load_file(d / "model.safetensors", device=device)
        self.dense = load_file(d / "1_Dense" / "model.safetensors", device=device)["linear.weight"]
        self.tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        self.tok.no_padding(); self.tok.no_truncation()
        self.ql, self.dl = query_length, document_length
        c = self.cfg
        self.H, self.nh, self.nkv = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        self.hd = self.H // self.nh
        self.eps = c["norm_eps"]
        inv = 1.0 / (c["rope_theta"] ** (torch.arange(0, self.hd, 2, dtype=torch.float32) / self.hd))
        f = torch.arange(max(512, document_length), dtype=torch.float32)[:, None] * inv[None]
        e = torch.cat([f, f], -1)
        self.cos, self.sin = e.cos().to(device), e.sin().to(device)
        vocab = self.tok.get_vocab()
        self.skip = {vocab[p] for p in string.punctuation if p in vocab}
        conv_i = 0
        self.dil = []
        for t in c["layer_types"]:
            if t == "conv":
                self.dil.append(c["conv_dilations"][conv_i]); conv_i += 1
            else:
                self.dil.append(None)

    # ---------- tokenization ----------
    def ids(self, texts, task, prefix):
        out = []
        for t in texts:
            body = self.tok.encode(t).ids[1:-1]  # strip the template's [CLS]/[SEP]
            if task == "query":
                room = self.ql - 2 - (1 if prefix else 0)
                seq = [CLS_ID] + ([Q_ID] if prefix else []) + body[:room] + [SEP_ID]
            else:
                room = self.dl - 2 - (1 if prefix else 0)
                seq = [CLS_ID] + ([D_ID] if prefix else []) + body[:room] + [SEP_ID]
            out.append(seq)
        return out

    # ---------- model ----------
    def _attn(self, i, x, amask):
        w, p = self.w, f"layers.{i}.attention."
        B, S, H = x.shape
        h = rms(x, w[p + "norm.weight"], self.eps)
        q = (h @ w[p + "Q_proj.weight"].T).view(B, S, self.nh, self.hd).transpose(1, 2)
        k = (h @ w[p + "K_proj.weight"].T).view(B, S, self.nkv, self.hd).transpose(1, 2)
        v = (h @ w[p + "V_proj.weight"].T).view(B, S, self.nkv, self.hd).transpose(1, 2)
        cos, sin = self.cos[:S], self.sin[:S]
        rot = lambda t: torch.cat([-t[..., self.hd // 2:], t[..., : self.hd // 2]], -1)
        q = q * cos + rot(q) * sin
        k = k * cos + rot(k) * sin
        rep = self.nh // self.nkv
        k = k.repeat_interleave(rep, dim=1); v = v.repeat_interleave(rep, dim=1)
        m = amask.bool()[:, None, None, :]
        if S > 513:
            idx = torch.arange(S, device=x.device)
            m = m & ((idx[None, :] - idx[:, None]).abs() <= 256)[None, None]
        o = F.scaled_dot_product_attention(q, k, v, attn_mask=m)
        return x + o.transpose(1, 2).reshape(B, S, H) @ w[p + "O_proj.weight"].T

    def _moe(self, i, x):
        w, p = self.w, f"layers.{i}.ffn."
        B, S, H = x.shape
        t = x.reshape(-1, H)
        probs = F.softmax(t @ w[p + "router.weight"].T, -1)
        tw, ti = probs.topk(self.cfg["top_k"], -1)
        tw = tw / (tw.sum(-1, keepdim=True) + 1e-9)
        out = torch.zeros_like(t)
        for e in range(self.cfg["num_routed_experts"]):
            sel = (ti == e)
            rows = sel.any(-1).nonzero().flatten()
            if rows.numel() == 0:
                continue
            g = (tw * sel).sum(-1)[rows, None]
            xe = t[rows]
            ye = (F.gelu(xe @ w[p + "W_gate"][e]) * (xe @ w[p + "W_up"][e])) @ w[p + "W_down"][e]
            out.index_add_(0, rows, g * ye)
        sh = (F.gelu(t @ w[p + "shared_gate.weight"].T) * (t @ w[p + "shared_up.weight"].T)) @ w[p + "shared_down.weight"].T
        return (out + sh).view(B, S, H)

    def _conv(self, i, x, amask):
        w, p = self.w, f"layers.{i}.mod."
        H = self.H
        h = rms(x, w[p + "in_norm.weight"], self.eps)
        bcx = h @ w[p + "in_proj.weight"].T
        b, c, xt = bcx[..., :H], bcx[..., H:2 * H], bcx[..., 2 * H:]
        b = rms(b, w[p + "gate_norm_b.weight"], self.eps)
        c = rms(c, w[p + "gate_norm_c.weight"], self.eps)
        bx = (b * xt * amask[..., None]).transpose(1, 2)
        dil = self.dil[i]
        y = F.conv1d(bx, w[p + "conv.weight"], padding=dil, dilation=dil, groups=H)[..., : bx.shape[-1]]
        y = c * y.transpose(1, 2)
        return x + y @ w[p + "out_proj.weight"].T

    @torch.no_grad()
    def forward(self, ids, amask):
        return self.forward_grad(ids, amask)

    def forward_grad(self, ids, amask):
        """forward() with autograd on, for run/distill.py."""
        w = self.w
        x = w["embeddings.hash_embeddings.weight"][ids] @ w["embeddings.projection.weight"].T
        x = F.layer_norm(x, (self.H,), w["embeddings.norm.weight"], w["embeddings.norm.bias"], 1e-5)
        for i, t in enumerate(self.cfg["layer_types"]):
            if t == "conv":
                x = self._conv(i, x, amask)
            else:
                x = self._attn(i, x, amask)
                x = x + self._moe(i, rms(x, w[f"layers.{i}.ffn_norm.weight"], self.eps))
        x = rms(x, w["norm.weight"], self.eps)
        return F.normalize(x @ self.dense.T, dim=-1)

    # ---------- ColBERT encode ----------
    def encode(self, texts, task, prefix=True, batch_size=32):
        """List of (n_tokens, 128) float32 arrays, in input order."""
        seqs = self.ids(texts, task, prefix)
        order = np.argsort([len(s) for s in seqs])
        res = [None] * len(seqs)
        for a in range(0, len(order), batch_size):
            idx = order[a:a + batch_size]
            if task == "query":
                S = self.ql
            else:
                S = max(len(seqs[j]) for j in idx)
            ids = torch.full((len(idx), S), MASK_ID if task == "query" else PAD_ID, dtype=torch.long)
            am = torch.zeros((len(idx), S), dtype=torch.float32)
            for r, j in enumerate(idx):
                ids[r, : len(seqs[j])] = torch.tensor(seqs[j]); am[r, : len(seqs[j])] = 1
            eff = torch.ones_like(am) if (task == "query" and prefix) else am
            E = self.forward(ids.to(self.dev), eff.to(self.dev)).float().cpu()
            for r, j in enumerate(idx):
                if task == "query":
                    res[j] = E[r].numpy()
                else:
                    keep = [k for k, tid in enumerate(seqs[j]) if tid not in self.skip]
                    res[j] = E[r, keep].numpy()
        return res


def maxsim(q, D):
    """q (32,128); D list of (n,128) -> scores."""
    return np.array([float((q @ d.T).max(1).sum()) for d in D])
