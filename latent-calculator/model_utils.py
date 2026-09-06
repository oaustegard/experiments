"""Frozen-LM plumbing: loading, injection hooks, trainable heads, calculator.

Injection arms
--------------
residual : encoder vector is added to the residual stream at the OUTPUT of
           decoder layer k (== input of layer k+1) at the query position t.
delayed  : identical, but at position t+1 (first answer token under teacher
           forcing / first decode step during generation).
kv       : one extra key/value slot appended to layer k+1's attention.  K and V
           are produced by running the encoder vector through layer k+1's own
           input_layernorm / k_proj / v_proj.  NO RoPE is applied to the slot
           (choice documented here): the slot is position-agnostic, which keeps
           it identical between the teacher-forced pass and every decode step
           and avoids giving the slot a spurious relative distance to later
           query positions.  The slot is visible only from query positions >= t.

Padding: RIGHT padding with per-row query indices `t` everywhere except
generation, which uses LEFT padding so that every row's prompt ends at the same
absolute index (this keeps the injection index and the kv-slot mask uniform
across the batch and lets positions be derived from the attention mask).
"""

import contextlib
import os

import torch
import torch.nn as nn
import transformers.models.llama.modeling_llama as modeling_llama
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.masking_utils import (ALL_MASK_ATTENTION_FUNCTIONS,
                                         create_causal_mask, eager_mask)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

from data import (BLANK, KINDS, N_DIGIT_CLASSES, N_OPERAND_SLOTS,
                  N_RESULT_SLOTS, OPS)

MODELS = {
    "monad": "PleIAs/Monad",
    "smol": "HuggingFaceTB/SmolLM2-135M",
}

torch.set_num_threads(4)


# --------------------------------------------------------------------------
# injection context
# --------------------------------------------------------------------------
class InjectionContext:
    """Mutable state consulted by the hooks and by the custom attention fn."""

    def __init__(self):
        self.enabled = False
        self.mode = None        # 'residual' | 'delayed' | 'kv'
        self.layer_k = None
        self.vec = None         # [B, H]
        self.t = None           # [B] long, absolute query index per row
        self.q_offset = 0       # absolute index of local position 0
        self.extra_k = None     # [B, n_kv_heads, 1, head_dim]
        self.extra_v = None

    def target_positions(self):
        return self.t + 1 if self.mode == "delayed" else self.t


_CTX = InjectionContext()


def get_ctx():
    return _CTX


@contextlib.contextmanager
def injection(mode, layer_k, vec, t, q_offset=0, extra_k=None, extra_v=None):
    ctx = _CTX
    prev = (ctx.enabled, ctx.mode, ctx.layer_k, ctx.vec, ctx.t, ctx.q_offset,
            ctx.extra_k, ctx.extra_v)
    ctx.enabled, ctx.mode, ctx.layer_k = True, mode, layer_k
    ctx.vec, ctx.t, ctx.q_offset = vec, t, q_offset
    ctx.extra_k, ctx.extra_v = extra_k, extra_v
    try:
        yield ctx
    finally:
        (ctx.enabled, ctx.mode, ctx.layer_k, ctx.vec, ctx.t, ctx.q_offset,
         ctx.extra_k, ctx.extra_v) = prev


# --------------------------------------------------------------------------
# custom attention implementation with the extra latent slot
# --------------------------------------------------------------------------
def latent_slot_attention(module, query, key, value, attention_mask, scaling,
                          dropout=0.0, **kwargs):
    ctx = _CTX
    if (ctx.enabled and ctx.mode == "kv" and ctx.extra_k is not None
            and getattr(module, "layer_idx", None) == ctx.layer_k + 1):
        ek = ctx.extra_k.to(key.dtype)
        ev = ctx.extra_v.to(value.dtype)
        n_real = key.shape[2]
        q_len = query.shape[2]
        key = torch.cat([key, ek], dim=2)
        value = torch.cat([value, ev], dim=2)
        b = query.shape[0]
        if attention_mask is None:
            attention_mask = query.new_zeros(b, 1, q_len, n_real)
        else:
            attention_mask = attention_mask[..., :n_real]
        pos = torch.arange(q_len, device=query.device) + (n_real - q_len)
        visible = pos.view(1, 1, q_len, 1) >= ctx.t.to(query.device).view(-1, 1, 1, 1)
        neg = torch.finfo(attention_mask.dtype).min
        col = torch.where(visible, torch.zeros((), dtype=attention_mask.dtype),
                          torch.tensor(neg, dtype=attention_mask.dtype))
        col = col.expand(b, 1, q_len, 1)
        attention_mask = torch.cat([attention_mask, col], dim=-1)
    return modeling_llama.eager_attention_forward(
        module, query, key, value, attention_mask, scaling, dropout, **kwargs)


if "latent_slot" not in ALL_ATTENTION_FUNCTIONS:
    ALL_ATTENTION_FUNCTIONS.register("latent_slot", latent_slot_attention)
# The mask machinery dispatches on the implementation NAME; without this the
# custom implementation gets a None mask (no padding mask, and causality left to
# sdpa's is_causal), which silently breaks both padding and the slot masking.
if "latent_slot" not in ALL_MASK_ATTENTION_FUNCTIONS:
    ALL_MASK_ATTENTION_FUNCTIONS.register("latent_slot", eager_mask)


def _residual_hook(module, args, output):
    ctx = _CTX
    if not ctx.enabled or ctx.mode not in ("residual", "delayed"):
        return None
    h = output[0] if isinstance(output, tuple) else output
    tgt = ctx.target_positions().to(h.device) - ctx.q_offset
    local = h.shape[1]
    ok = (tgt >= 0) & (tgt < local)
    if not bool(ok.any()):
        return None
    h = h.clone()
    idx = torch.nonzero(ok, as_tuple=False).flatten()
    h[idx, tgt[idx], :] = h[idx, tgt[idx], :] + ctx.vec[idx].to(h.dtype)
    if isinstance(output, tuple):
        return (h,) + output[1:]
    return h


def attach_hook(model, layer_k):
    """Attach the residual/delayed hook to decoder layer `layer_k`.

    layer_k == -1 means the embedding output; that is handled by hooking
    model.model.embed_tokens instead.
    """
    if layer_k < 0:
        return model.model.embed_tokens.register_forward_hook(_residual_hook)
    return model.model.layers[layer_k].register_forward_hook(_residual_hook)


def make_slot_kv(model, layer_k, vec):
    """K/V for the latent slot, from layer k+1's own layernorm and projections."""
    layer = model.model.layers[layer_k + 1]
    attn = layer.self_attn
    h = layer.input_layernorm(vec)
    b = h.shape[0]
    hd = attn.head_dim
    k = attn.k_proj(h).view(b, 1, -1, hd).transpose(1, 2)
    v = attn.v_proj(h).view(b, 1, -1, hd).transpose(1, 2)
    return k, v


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------
def load_model(name, attn="latent_slot"):
    path = MODELS[name]
    tok = AutoTokenizer.from_pretrained(path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)
    model.config._attn_implementation = attn
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, tok


def n_layers(model):
    return len(model.model.layers)


def hidden_size(model):
    return model.config.hidden_size


# --------------------------------------------------------------------------
# batching
# --------------------------------------------------------------------------
def encode_rows(tok, rows, with_answer=True, pad_side="right"):
    """Right-padded batch.  Returns dict with input_ids, attention_mask, t,
    labels (-100 outside answer tokens), n_prompt_tokens."""
    prompts = [tok(r["prompt"])["input_ids"] for r in rows]
    answers = ([tok(r["answer"], add_special_tokens=False)["input_ids"] for r in rows]
               if with_answer else [[] for _ in rows])
    seqs = [p + a for p, a in zip(prompts, answers)]
    tmax = max(len(s) for s in seqs)
    pad = tok.pad_token_id
    ids = torch.full((len(seqs), tmax), pad, dtype=torch.long)
    mask = torch.zeros((len(seqs), tmax), dtype=torch.long)
    labels = torch.full((len(seqs), tmax), -100, dtype=torch.long)
    t = torch.zeros(len(seqs), dtype=torch.long)
    for i, (p, a, s) in enumerate(zip(prompts, answers, seqs)):
        n = len(s)
        if pad_side == "right":
            ids[i, :n] = torch.tensor(s)
            mask[i, :n] = 1
            off = 0
        else:
            ids[i, tmax - n:] = torch.tensor(s)
            mask[i, tmax - n:] = 1
            off = tmax - n
        t[i] = off + len(p) - 1
        for j in range(len(a)):
            labels[i, off + len(p) + j] = a[j]
    return {"input_ids": ids, "attention_mask": mask, "t": t, "labels": labels,
            "n_prompt_tokens": torch.tensor([len(p) for p in prompts])}


def position_ids_from_mask(mask):
    """Positions for LEFT-padded batches (pads get position 0 and are masked).

    For RIGHT-padded batches this equals arange on the real tokens, which is
    what HF's default (arange) gives, so it is safe for both."""
    pos = (mask.cumsum(-1) - 1).clamp(min=0)
    return pos.long()


def build_causal_mask(mask_2d, dtype, q_offset=0):
    """Additive 4D mask [B,1,Tq,Tk] matching HF eager semantics."""
    b, tk = mask_2d.shape
    tq = tk - q_offset
    neg = torch.finfo(dtype).min
    qi = torch.arange(q_offset, tk).view(1, tq, 1)
    ki = torch.arange(tk).view(1, 1, tk)
    allowed = (ki <= qi) & (mask_2d.view(b, 1, tk) > 0)
    out = torch.zeros((b, tq, tk), dtype=dtype)
    out = out.masked_fill(~allowed, neg)
    return out.unsqueeze(1)


def forward_upper(model, hidden, mask_2d, start_layer):
    """Run decoder layers [start_layer:] + norm + lm_head on `hidden`.

    `hidden` must be the output of decoder layer start_layer-1 (or embeddings
    if start_layer == 0) for the full padded sequence.
    """
    mdl = model.model
    position_ids = position_ids_from_mask(mask_2d)
    pos_emb = mdl.rotary_emb(hidden, position_ids=position_ids)
    attn_mask = create_causal_mask(config=model.config, inputs_embeds=hidden,
                                   attention_mask=mask_2d, past_key_values=None,
                                   position_ids=position_ids)
    for layer in mdl.layers[start_layer:]:
        hidden = layer(hidden, attention_mask=attn_mask, position_ids=position_ids,
                       past_key_values=None, use_cache=False,
                       position_embeddings=pos_emb)
        if isinstance(hidden, tuple):
            hidden = hidden[0]
    hidden = mdl.norm(hidden)
    return model.lm_head(hidden)


@torch.no_grad()
def extract_hidden(model, tok, prompts, layers="all", batch_size=32, verbose=False):
    """Hidden states at the query position t for every layer.

    Returns fp16 tensor [N, L+1, H]; index 0 == embeddings (layer -1),
    index k+1 == output of decoder layer k.
    """
    rows = [{"prompt": p, "answer": ""} for p in prompts]
    out = []
    for i in range(0, len(rows), batch_size):
        chunk = rows[i:i + batch_size]
        b = encode_rows(tok, chunk, with_answer=False)
        res = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"],
                    output_hidden_states=True, use_cache=False)
        hs = torch.stack(res.hidden_states, dim=1)          # [B, L+1, T, H]
        idx = b["t"].view(-1, 1, 1, 1).expand(-1, hs.shape[1], 1, hs.shape[3])
        sel = hs.gather(2, idx).squeeze(2)                  # [B, L+1, H]
        out.append(sel.to(torch.float16))
        if verbose and (i // batch_size) % 20 == 0:
            print(f"  extract {i}/{len(rows)}", flush=True)
    res = torch.cat(out, 0)
    if layers != "all":
        res = res[:, [k + 1 for k in layers], :]
    return res


# --------------------------------------------------------------------------
# trainable heads
# --------------------------------------------------------------------------
N_QUERY_OUT = len(OPS) + 2 * N_OPERAND_SLOTS * N_DIGIT_CLASSES


class QueryHead(nn.Module):
    """hidden[t] -> operator logits + digit-slot logits for both operands."""

    def __init__(self, hidden, mlp=512, linear=False):
        super().__init__()
        self.linear = linear
        if linear:
            self.net = nn.Linear(hidden, N_QUERY_OUT)
        else:
            self.net = nn.Sequential(nn.Linear(hidden, mlp), nn.GELU(),
                                     nn.Linear(mlp, N_QUERY_OUT))

    def forward(self, h):
        o = self.net(h)
        op = o[..., :len(OPS)]
        rest = o[..., len(OPS):].view(*o.shape[:-1], 2 * N_OPERAND_SLOTS,
                                      N_DIGIT_CLASSES)
        return op, rest


def query_loss(op_logits, slot_logits, op_t, slot_t):
    lo = nn.functional.cross_entropy(op_logits, op_t)
    ls = nn.functional.cross_entropy(
        slot_logits.reshape(-1, N_DIGIT_CLASSES), slot_t.reshape(-1))
    return lo + ls


# result symbol vocabulary: 0-9, BLANK, sign(+,-), kind(numeric/gt/lt/eq)
SIGN_OFFSET = N_DIGIT_CLASSES            # 11, 12
KIND_OFFSET = N_DIGIT_CLASSES + 2        # 13..16
RESULT_VOCAB = KIND_OFFSET + len(KINDS)  # 17
N_RESULT_TOKENS = N_RESULT_SLOTS + 2     # 12 digits + sign + kind


class ResultEncoder(nn.Module):
    """Structured result -> a single vector in the model's hidden space."""

    def __init__(self, hidden, emb=32, mlp=512):
        super().__init__()
        self.emb = nn.Embedding(RESULT_VOCAB, emb)
        self.pos = nn.Embedding(N_RESULT_TOKENS, emb)
        self.mlp = nn.Sequential(
            nn.Linear(N_RESULT_TOKENS * 2 * emb, mlp), nn.GELU(),
            nn.Linear(mlp, hidden))

    def forward(self, symbols):
        """symbols: [B, N_RESULT_TOKENS] long (already offset-encoded)."""
        b = symbols.shape[0]
        e = self.emb(symbols)
        p = self.pos(torch.arange(N_RESULT_TOKENS, device=symbols.device))
        x = torch.cat([e, p.unsqueeze(0).expand(b, -1, -1)], dim=-1)
        return self.mlp(x.reshape(b, -1))


def result_symbols(result_strings):
    """Encode result strings into [N, N_RESULT_TOKENS] symbol ids."""
    from data import result_target
    out = torch.zeros((len(result_strings), N_RESULT_TOKENS), dtype=torch.long)
    for i, s in enumerate(result_strings):
        tgt = result_target(s)
        out[i, :N_RESULT_SLOTS] = torch.tensor(tgt["slots"])
        out[i, N_RESULT_SLOTS] = SIGN_OFFSET + tgt["sign"]
        out[i, N_RESULT_SLOTS + 1] = KIND_OFFSET + tgt["kind"]
    return out


def count_params(*modules):
    return sum(p.numel() for m in modules for p in m.parameters())


# --------------------------------------------------------------------------
# calculator
# --------------------------------------------------------------------------
def decode_operand(slots):
    """Right-aligned digit slots -> int.  BLANK inside a number counts as 0;
    an all-BLANK operand is 0."""
    nz = [i for i, s in enumerate(slots) if s != BLANK]
    if not nz:
        return 0
    hi = max(nz)
    v = 0
    for i in range(hi, -1, -1):
        d = slots[i]
        v = v * 10 + (0 if d == BLANK else int(d))
    return v


def calculate(op_idx, a_slots, b_slots):
    """Pure-python calculator over the argmaxed query.  Returns result string."""
    op = OPS[int(op_idx)]
    a = decode_operand(list(a_slots))
    b = decode_operand(list(b_slots))
    if op == "add":
        return str(a + b)
    if op == "sub":
        return str(a - b)
    if op == "mul":
        return str(a * b)
    return "greater" if a > b else ("less" if a < b else "equal")


def calculate_from_logits(op_logits, slot_logits):
    ops = op_logits.argmax(-1).tolist()
    slots = slot_logits.argmax(-1).tolist()
    out = []
    for o, s in zip(ops, slots):
        out.append(calculate(o, s[:N_OPERAND_SLOTS], s[N_OPERAND_SLOTS:]))
    return out


def answer_token_loss(logits, labels):
    """Next-token CE on answer tokens only (labels are -100 elsewhere)."""
    lg = logits[:, :-1, :]
    lb = labels[:, 1:]
    return nn.functional.cross_entropy(
        lg.reshape(-1, lg.shape[-1]), lb.reshape(-1), ignore_index=-100)


def answer_token_acc(logits, labels):
    lg = logits[:, :-1, :].argmax(-1)
    lb = labels[:, 1:]
    m = lb != -100
    if m.sum() == 0:
        return float("nan")
    return float((lg[m] == lb[m]).float().mean())


def repo_dir():
    return os.path.dirname(os.path.abspath(__file__))


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p
