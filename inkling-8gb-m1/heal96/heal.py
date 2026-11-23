#!/usr/bin/env python3
"""Inkling 2 Mini: prune experts 128 -> K per layer and heal by distillation.

The released checkpoint stores routed experts as NVFP4 (layers 0-19) and MXFP8
(layers 20-23). This script keeps them packed on the GPU and dequantizes per
expert chunk inside the matmul (forward and backward), so the full teacher fits
in ~20 GB and a 96-expert student in ~16 GB.

Teacher and student run as separate jobs. The teacher writes its top-k
log-probs per token; the student trains LoRA adapters on the kept experts plus a
router delta against them (KL on the top-k distribution plus a tail bucket).

Subcommands
  data     build token shards (train mix + eval sets)
  teacher  full model: routing stats + top-k log-probs for a list of shards
  select   keep sets from routing stats
  train    student KD training on teacher shards
  evalstu  evaluate a student (unhealed or a checkpoint) on the eval shards
  export   write healed bf16 expert + router tensors for the GGUF build
"""
import argparse, glob, json, math, os, re, sys, time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

MODEL_DIR = os.environ.get("INK_MODEL", "/content/ink")
ST_FILE = None  # resolved lazily
TOPK = 32

# ---------------------------------------------------------------- runtime state
class State:
    student = False      # apply keep-mask, router delta and LoRA
    collect = False      # accumulate routing statistics
    chunk = 16           # experts dequantized per chunk


STATE = State()

E2M1 = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
_LUT = {}


def byte_lut(device):
    if device not in _LUT:
        v = torch.tensor(E2M1, device=device, dtype=torch.float32)
        b = torch.arange(256, device=device)
        _LUT[device] = torch.stack((v[b & 0xF], v[b >> 4]), dim=-1)  # [256, 2], low nibble first
    return _LUT[device]


try:
    import triton
    import triton.language as tl

    @triton.jit
    def _nvfp4_kernel(w_ptr, s_ptr, f_ptr, lut_ptr, lut8_ptr, out_ptr, n_bytes, bytes_per_expert, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < n_bytes
        b = tl.load(w_ptr + i, mask=m, other=0).to(tl.int32)
        sb = tl.load(s_ptr + i // 8, mask=m, other=0).to(tl.int32)  # fp8 e4m3 bits, decoded by table (sm80 has no fp8)
        sc = tl.load(lut8_ptr + sb, mask=m, other=0.0)
        fa = tl.load(f_ptr + i // bytes_per_expert, mask=m, other=0.0)
        k = sc * fa
        lo = tl.load(lut_ptr + (b & 0xF), mask=m, other=0.0) * k
        hi = tl.load(lut_ptr + (b >> 4), mask=m, other=0.0) * k
        tl.store(out_ptr + 2 * i, lo.to(tl.bfloat16), mask=m)
        tl.store(out_ptr + 2 * i + 1, hi.to(tl.bfloat16), mask=m)

    @triton.jit
    def _mxfp8_kernel(w_ptr, s_ptr, lut8_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        i = pid * BLOCK + tl.arange(0, BLOCK)
        m = i < n
        b = tl.load(w_ptr + i, mask=m, other=0).to(tl.int32)
        e = tl.load(s_ptr + i // 32, mask=m, other=127).to(tl.float32)
        v = tl.load(lut8_ptr + b, mask=m, other=0.0) * tl.exp2(e - 127.0)
        tl.store(out_ptr + i, v.to(tl.bfloat16), mask=m)

    _TRITON = True
except Exception:  # noqa
    _TRITON = False
_TRITON_OK = None


def _dequant_nvfp4_triton(w, s, fac):
    c, rows, half = w.shape
    out = torch.empty(c, rows, half * 2, dtype=torch.bfloat16, device=w.device)
    key = ("tri", w.device)
    if key not in _LUT:
        _LUT[key] = (torch.tensor(E2M1, device=w.device, dtype=torch.float32),
                     torch.arange(256, dtype=torch.uint8, device=w.device).view(torch.float8_e4m3fn).float())
    lut, lut8 = _LUT[key]
    n = w.numel()
    BLOCK = 2048
    _nvfp4_kernel[(triton.cdiv(n, BLOCK),)](w, s.view(torch.uint8), fac.contiguous(), lut, lut8, out, n, rows * half,
                                            BLOCK=BLOCK)
    return out


def dequant(p, e0, e1, dtype=torch.bfloat16):
    """p: dict(weight, scale, amax|None, den|None) for one projection, experts [e0, e1)."""
    global _TRITON_OK
    if dtype == torch.bfloat16 and p["weight"].is_cuda:
        w = p["weight"][e0:e1]
        if p.get("amax") is not None and _TRITON:
            fac = p["amax"][e0:e1].float() / p["den"].float()
            if _TRITON_OK is None:
                ref = _dequant_torch(p, e0, min(e1, e0 + 2), torch.float32)
                got = _dequant_nvfp4_triton(p["weight"][e0:e0 + ref.shape[0]], p["scale"][e0:e0 + ref.shape[0]],
                                            fac[:ref.shape[0]]).float()
                _TRITON_OK = bool(torch.allclose(got, ref.to(torch.bfloat16).float(), rtol=1e-2, atol=1e-6))
                print("triton nvfp4 dequant:", "ok" if _TRITON_OK else "MISMATCH -> torch path", flush=True)
            if _TRITON_OK:
                return _dequant_nvfp4_triton(w, p["scale"][e0:e1], fac)
        elif p.get("amax") is None and _TRITON and _TRITON_OK:
            key = ("tri", w.device)
            out = torch.empty(w.shape, dtype=torch.bfloat16, device=w.device)
            _mxfp8_kernel[(triton.cdiv(w.numel(), 2048),)](w.view(torch.uint8), p["scale"][e0:e1], _LUT[key][1], out,
                                                           w.numel(), BLOCK=2048)
            return out
        elif p.get("amax") is None:  # MXFP8: power-of-two scale, exact in bf16
            s = torch.exp2(p["scale"][e0:e1].float() - 127).to(torch.bfloat16)
            c, rows = w.shape[:2]
            return (w.to(torch.bfloat16).view(c, rows, s.shape[-1], 32) * s[..., None]).reshape(c, rows, -1)
    return _dequant_torch(p, e0, e1, dtype)


def _dequant_torch(p, e0, e1, dtype=torch.bfloat16):
    w = p["weight"][e0:e1]
    s = p["scale"][e0:e1].float()
    c, rows = w.shape[:2]
    if p.get("amax") is not None:  # NVFP4: two E2M1 codes per byte, fp8 scale per 16, per-expert factor
        vals = byte_lut(w.device)[w.long()].view(c, rows, s.shape[-1], 16) * s[..., None]
        vals = vals * (p["amax"][e0:e1].float() / p["den"].float())[:, None, None, None]
    else:  # MXFP8: fp8 values, power-of-two scale per 32
        vals = w.float().view(c, rows, s.shape[-1], 32) * torch.exp2(s - 127)[..., None]
    return vals.reshape(c, rows, -1).to(dtype)


# ---------------------------------------------------------------- grouped matmul
_GMM = None


def gmm(a, b, offs):
    """a [M, K], b [G, K, N], offs int32 [G] cumulative row ends -> [M, N]."""
    global _GMM
    if _GMM is None:
        _GMM = "loop"
        f = getattr(F, "grouped_mm", None) or getattr(torch, "_grouped_mm", None)
        if f is not None and a.is_cuda:
            try:
              with torch.enable_grad():
                xa = torch.randn(32, 16, device=a.device, dtype=torch.bfloat16, requires_grad=True)
                xb = torch.randn(2, 16, 16, device=a.device, dtype=torch.bfloat16, requires_grad=True)
                t = f(xa, xb.transpose(1, 2), offs=torch.tensor([16, 32], device=a.device, dtype=torch.int32))
                (t.float() * torch.randn_like(t, dtype=torch.float32)).sum().backward()
                ref = torch.cat([xa[:16] @ xb[0].T, xa[16:] @ xb[1].T])
                assert torch.allclose(t.float(), ref.float(), atol=1e-1, rtol=1e-2)
                _GMM = f
            except Exception as ex:  # noqa
                print("grouped_mm unavailable:", ex, flush=True)
        print("gmm impl:", _GMM if _GMM == "loop" else _GMM.__name__, flush=True)
    if _GMM != "loop" and a.dtype == torch.bfloat16 and b.dtype == torch.bfloat16:
        return _GMM(a, b, offs=offs)
    out = a.new_empty(a.shape[0], b.shape[-1])
    ends = offs.tolist()
    start = 0
    for g, end in enumerate(ends):
        if end > start:
            out[start:end] = a[start:end] @ b[g]
        start = end
    return out


class PackedMM(torch.autograd.Function):
    """y = x @ W[e]^T for row groups, W dequantized per expert chunk; W is frozen."""

    @staticmethod
    def forward(ctx, x, offs, packed, n_out):
        ends = offs.tolist()
        y = x.new_empty(x.shape[0], n_out)
        n = len(ends)
        C = STATE.chunk
        for e0 in range(0, n, C):
            e1 = min(n, e0 + C)
            r0 = ends[e0 - 1] if e0 else 0
            r1 = ends[e1 - 1]
            if r1 == r0:
                continue
            W = dequant(packed, e0, e1, x.dtype)  # [c, out, in]
            sub = (offs[e0:e1] - r0).to(torch.int32)
            y[r0:r1] = gmm(x[r0:r1].contiguous(), W.transpose(1, 2), sub)
        ctx.save_for_backward(offs)
        ctx.packed = packed
        ctx.n_in = x.shape[1]
        return y

    @staticmethod
    def backward(ctx, gy):
        (offs,) = ctx.saved_tensors
        ends = offs.tolist()
        gx = gy.new_zeros(gy.shape[0], ctx.n_in)
        n = len(ends)
        C = STATE.chunk
        gy = gy.contiguous()
        for e0 in range(0, n, C):
            e1 = min(n, e0 + C)
            r0 = ends[e0 - 1] if e0 else 0
            r1 = ends[e1 - 1]
            if r1 == r0:
                continue
            W = dequant(ctx.packed, e0, e1, gy.dtype)  # [c, out, in]
            sub = (offs[e0:e1] - r0).to(torch.int32)
            gx[r0:r1] = gmm(gy[r0:r1], W, sub)
        return gx, None, None, None


class LoraMM(torch.autograd.Function):
    """y = (x @ A[e]^T) @ B[e]^T * scale per row group; saves a fixed set of tensors whatever the routing."""

    @staticmethod
    def forward(ctx, x, A, B, offs, scale):
        Ab, Bb = A.to(x.dtype), B.to(x.dtype)
        t = gmm(x, Ab.transpose(1, 2), offs)
        y = gmm(t, Bb.transpose(1, 2), offs) * scale
        ctx.save_for_backward(x, t, A, B, offs)
        ctx.scale = scale
        return y

    @staticmethod
    def backward(ctx, gy):
        x, t, A, B, offs = ctx.saved_tensors
        gy = (gy * ctx.scale).contiguous()
        Ab, Bb = A.to(x.dtype), B.to(x.dtype)
        gt = gmm(gy, Bb, offs)            # [M, r]
        gx = gmm(gt, Ab, offs)            # [M, in]
        gA = torch.zeros_like(A)
        gB = torch.zeros_like(B)
        start = 0
        for g, end in enumerate(offs.tolist()):
            if end > start:
                gB[g] = (gy[start:end].T.float() @ t[start:end].float())
                gA[g] = (gt[start:end].T.float() @ x[start:end].float())
            start = end
        return gx, gA, gB, None, None


# ---------------------------------------------------------------- modules
class PackedExperts(nn.Module):
    """Drop-in for Inkling2LoopExperts holding packed weights for a subset of experts."""

    def __init__(self, config):
        super().__init__()
        import transformers.models.inkling2_loop.modeling_inkling2_loop as M
        self.num_experts = config.n_routed_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.act_fn = M.Inkling2LoopSoftcappedSwiGLU(config)
        self.packed = None  # {"w13": {...}, "w2": {...}}
        self.lora = None
        self.lora_scale = 1.0
        self.layer_idx = -1

    def set_packed(self, packed, experts):
        self.packed = packed
        self.experts = torch.as_tensor(experts, dtype=torch.long)
        slot = torch.full((self.num_experts,), -1, dtype=torch.long)
        slot[self.experts] = torch.arange(len(experts))
        self.register_buffer("slot_of", slot.to(packed["w2"]["weight"].device), persistent=False)
        self.n_local = len(experts)
        self.stats_count = torch.zeros(self.num_experts, dtype=torch.float64, device=self.slot_of.device)
        self.stats_reap = torch.zeros_like(self.stats_count)
        self.stats_gate = torch.zeros_like(self.stats_count)

    def add_lora(self, r, alpha):
        dev = self.slot_of.device
        n, H, I = self.n_local, self.hidden_dim, self.intermediate_dim
        self.lora = nn.ParameterDict({
            "A13": nn.Parameter(torch.empty(n, r, H, device=dev).uniform_(-1, 1) / math.sqrt(H)),
            "B13": nn.Parameter(torch.zeros(n, 2 * I, r, device=dev)),
            "A2": nn.Parameter(torch.empty(n, r, I, device=dev).uniform_(-1, 1) / math.sqrt(I)),
            "B2": nn.Parameter(torch.zeros(n, H, r, device=dev)),
        })
        self.lora_scale = alpha / r

    def _lora(self, x, A, B, offs):
        return LoraMM.apply(x, self.lora[A], self.lora[B], offs, self.lora_scale)

    def forward(self, hidden_states, top_k_index, top_k_weights):
        N, H = hidden_states.shape
        K = top_k_index.shape[1]
        flat = self.slot_of[top_k_index.reshape(-1)]
        if not STATE.student and (flat < 0).any():
            raise RuntimeError("routed to an expert that is not loaded")
        order = torch.argsort(flat, stable=True)
        tok = order // K
        counts = torch.bincount(flat, minlength=self.n_local)
        offs = counts.cumsum(0).to(torch.int32)
        xs = hidden_states[tok]
        use_lora = STATE.student and self.lora is not None
        h = PackedMM.apply(xs, offs, self.packed["w13"], 2 * self.intermediate_dim)
        if use_lora:
            h = h + self._lora(xs, "A13", "B13", offs)
        a = self.act_fn(h[:, 0::2], h[:, 1::2])  # checkpoint rows interleave gate/up
        y = PackedMM.apply(a, offs, self.packed["w2"], H)
        if use_lora:
            y = y + self._lora(a, "A2", "B2", offs)
        w = top_k_weights.reshape(-1)[order].float()
        if STATE.collect:
            with torch.no_grad():
                eid = self.experts.to(flat.device)[flat[order]]
                nrm = y.float().norm(dim=-1)
                self.stats_count.index_add_(0, eid, torch.ones_like(nrm, dtype=torch.float64))
                self.stats_reap.index_add_(0, eid, (nrm * w).double())
                self.stats_gate.index_add_(0, eid, w.double())
        # deterministic combine (index_add_ uses atomics; a recompute that drifts can flip routing)
        yw = torch.empty(N * K, H, dtype=torch.float32, device=hidden_states.device)
        yw[order] = y.float() * w[:, None]
        return yw.view(N, K, H).sum(1).to(hidden_states.dtype)


def router_forward(self, hidden_states):
    hs = hidden_states.float().reshape(-1, self.hidden_dim)
    W = self.weight
    if STATE.student and getattr(self, "delta", None) is not None:
        W = W + self.delta
    logits = F.linear(hs, W)
    scores = self.score_fn(logits)
    sel = scores + self.e_score_correction_bias
    km = getattr(self, "keep_mask", None)
    if STATE.student and km is not None:
        sel = sel.masked_fill(~km, float("-inf"))
    idx = torch.topk(sel, self.top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, idx)
    w = w / (w.sum(dim=-1, keepdim=True) + 1e-20)
    return logits, w * self.routed_scaling_factor, idx


def patch_transformers():
    import transformers.models.inkling2_loop.modeling_inkling2_loop as M
    from transformers import core_model_loading as CML

    M.Inkling2LoopExperts = PackedExperts
    M.Inkling2LoopTopKRouter.forward = router_forward

    def skip_convert(self, input_dict, source_patterns, target_patterns, **kw):
        return {}

    CML.Inkling2LoopDequantizeExperts.convert = skip_convert
    orig_init = M.Inkling2LoopPreTrainedModel._init_weights
    if not getattr(orig_init, "_patched", False):
        def _init_weights(self, module):
            if isinstance(module, PackedExperts):
                return
            return orig_init(self, module)
        _init_weights._patched = True
        M.Inkling2LoopPreTrainedModel._init_weights = _init_weights


def st_file():
    return sorted(glob.glob(os.path.join(MODEL_DIR, "model-*.safetensors")))[0]


def load_model(keep=None, device="cuda"):
    """keep: None (all experts) or {layer: [expert ids]}."""
    patch_transformers()
    from transformers import AutoConfig
    from transformers.models.inkling2_loop.modeling_inkling2_loop import Inkling2LoopForCausalLM
    from safetensors import safe_open

    t0 = time.time()
    model = Inkling2LoopForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map=device)
    model.config.use_cache = False
    print(f"base loaded {time.time()-t0:.0f}s, mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    experts = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.mlp\.experts$", name)
        if m and isinstance(mod, PackedExperts):
            experts[int(m.group(1))] = mod
        m = re.search(r"layers\.(\d+)\.mlp\.gate$", name)
        if m:
            mod.layer_idx = int(m.group(1))
    assert len(experts) == 24, len(experts)
    rts = routers(model)
    with safe_open(st_file(), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for li, r in rts.items():  # module index must match checkpoint layer index
            ref = f.get_tensor(f"model.llm.layers.{li}.mlp.gate.weight").float().to(r.weight.device)
            assert torch.allclose(r.weight.float(), ref), f"router {li} mismatch"
        for li, mod in sorted(experts.items()):
            ids = list(range(128)) if keep is None else sorted(keep[li])
            sel = torch.tensor(ids)
            packed = {}
            for proj in ("w13", "w2"):
                base = f"model.llm.layers.{li}.mlp.experts.{proj}_weight"
                d = {"weight": f.get_tensor(base)[sel].to(device),
                     "scale": f.get_tensor(base + "_scale")[sel].to(device)}
                if base + "_amax" in keys:
                    d["amax"] = f.get_tensor(base + "_amax")[sel].to(device)
                    d["den"] = f.get_tensor(base + "_outer_scale_denominator").to(device)
                packed[proj] = d
            mod.set_packed(packed, ids)
            mod.layer_idx = li
    for p in model.parameters():
        p.requires_grad_(False)
    # pick the grouped-matmul implementation now: its probe runs a backward pass, which must not happen
    # inside a checkpointed layer (the recompute would save a different number of tensors)
    gmm(torch.zeros(16, 16, device=device, dtype=torch.bfloat16), torch.zeros(1, 16, 16, device=device,
        dtype=torch.bfloat16), torch.tensor([16], device=device, dtype=torch.int32))
    print(f"experts loaded {time.time()-t0:.0f}s, mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    return model, experts


def routers(model):
    out = {}
    for name, mod in model.named_modules():
        m = re.search(r"layers\.(\d+)\.mlp\.gate$", name)
        if m:
            out[int(m.group(1))] = mod
    return out


def logits_from_hidden(model, h):
    return model.logit_softcap(model.lm_head(h).float())


# ---------------------------------------------------------------- data
def cmd_data(a):
    """Token shards: train mix and eval sets. Text sources are fetched on the VM."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 200006
    T = a.seq
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(0)

    def pack(texts, budget, name, shard_seqs=256):
        buf, n_tok, seqs, shard = [], 0, [], 0
        for t in texts:
            ids = tok(t, add_special_tokens=False)["input_ids"] if isinstance(t, str) else t
            buf.extend(ids + [eos])
            while len(buf) >= T:
                seqs.append(buf[:T]); buf = buf[T:]
                n_tok += T
                if len(seqs) == shard_seqs:
                    np.save(f"{a.out}/{name}_{shard:04d}.npy", np.array(seqs, dtype=np.uint32)); shard += 1; seqs = []
            if n_tok >= budget:
                break
        if seqs:
            np.save(f"{a.out}/{name}_{shard:04d}.npy", np.array(seqs, dtype=np.uint32))
        print(f"{name}: {n_tok/1e6:.2f}M tokens", flush=True)
        return n_tok

    # Oskar's code. Held-out = the 42 files of inkling-8gb-m1/results/heldout_code_files.txt (the earlier
    # pruning run's code eval), built the same way as its heldout_code.txt; they never enter training.
    code_files = [json.loads(l) for l in open(a.code_jsonl)]
    fmt = lambda r: f"# {r['repo']}/{r['path']}\n{r['text']}"
    held = [l.strip() for l in open(a.heldout_list) if l.strip()]
    hs = set(held)
    train_code = [r for r in code_files if r["path"] not in hs]
    print(f"code files {len(code_files)}, excluded as held-out {len(code_files) - len(train_code)}", flush=True)
    rng.shuffle(train_code)
    parts = []
    for n in held:
        for repo in ("experiments", "claude-skills", "muninn-utilities", "m5"):
            p = os.path.join("/content/repos", repo, n)
            if os.path.exists(p):
                parts.append(f"# file: {n}\n{open(p, encoding='utf-8').read()}\n\n")
                break
        else:
            print("held-out file missing:", n, flush=True)
    heldtxt = "".join(parts)
    open("/content/heldout_code.txt", "w").write(heldtxt)
    print(f"held-out code: {len(parts)} files {len(heldtxt)} chars", flush=True)
    pack([heldtxt], 10**9, "eval_code", shard_seqs=10**9)

    from datasets import load_dataset
    wiki = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    pack(["\n".join(wiki["text"])], a.eval_tokens, "eval_wiki", shard_seqs=10**9)

    chat = load_dataset("HuggingFaceTB/smoltalk", "all", split="train", streaming=True).shuffle(seed=0, buffer_size=20000)
    chat_it = iter(chat)

    def chat_texts():
        for ex in chat_it:
            try:
                yield tok.apply_chat_template(ex["messages"], tokenize=False)
            except Exception:
                continue
    pack(chat_texts(), a.eval_tokens, "eval_chat", shard_seqs=10**9)

    web = iter(load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True))
    pubcode = None
    for name, cfg in [("codeparrot/codeparrot-clean", None), ("bigcode/the-stack-smol", "data/python")]:
        try:
            pubcode = iter(load_dataset(name, cfg, split="train", streaming=True)) if cfg else iter(load_dataset(name, split="train", streaming=True))
            next(pubcode)
            print("public code:", name, flush=True)
            break
        except Exception as ex:
            print("no", name, ex, flush=True)
            pubcode = None

    def mix():
        own = iter(train_code)
        w = np.array([a.mix_code_own, a.mix_code_pub, a.mix_chat, a.mix_web], dtype=float)
        w /= w.sum()
        while True:
            k = rng.choice(4, p=w)
            try:
                if k == 0:
                    yield fmt(next(own))
                elif k == 1 and pubcode is not None:
                    yield next(pubcode)["content"]
                elif k == 2:
                    yield next(chat_texts())
                else:
                    yield next(web)["text"]
            except StopIteration:
                if k == 0:
                    own = iter(train_code)
                continue
    pack(mix(), a.train_tokens, "train", shard_seqs=a.shard_seqs)
    sys.stdout.flush()
    os._exit(0)  # streaming dataset threads otherwise keep the interpreter alive


# ---------------------------------------------------------------- teacher
@torch.no_grad()
def cmd_teacher(a):
    model, experts = load_model(None)
    STATE.student = False
    shards = sorted(glob.glob(os.path.join(a.data, a.pattern)), reverse=a.reverse)
    os.makedirs(a.out, exist_ok=True)
    STATE.collect = a.stats
    t0, ntok = time.time(), 0
    for sh in shards:
        dst = os.path.join(a.out, os.path.basename(sh).replace(".npy", ".npz"))
        if os.path.exists(dst) or os.path.exists(os.path.join(a.out, "skip_" + os.path.basename(sh))):
            continue
        x = np.load(sh).astype(np.int64)
        idx_all = np.zeros((*x.shape, TOPK), np.int32)
        lp_all = np.zeros((*x.shape, TOPK), np.float16)
        nll = np.full(x.shape, np.nan, np.float32)
        for i in range(0, len(x), a.bs):
            ids = torch.from_numpy(x[i:i + a.bs]).cuda()
            h = model.model(input_ids=ids).last_hidden_state
            T = h.shape[1]
            for j in range(h.shape[0]):
                for c0 in range(0, T, a.lchunk):
                    c1 = min(T, c0 + a.lchunk)
                    lp = torch.log_softmax(logits_from_hidden(model, h[j, c0:c1]), -1)
                    v, ix = lp.topk(TOPK, -1)
                    idx_all[i + j, c0:c1] = ix.int().cpu().numpy()
                    lp_all[i + j, c0:c1] = v.half().cpu().numpy()
                    e1 = min(c1, T - 1)
                    if e1 > c0:
                        nll[i + j, c0:e1] = (-lp[:e1 - c0].gather(-1, ids[j, c0 + 1:e1 + 1, None])[:, 0]).float().cpu().numpy()
                    del lp
            ntok += ids.numel()
        np.savez(dst + ".tmp.npz", idx=idx_all, lp=lp_all, nll=nll)
        os.replace(dst + ".tmp.npz", dst)
        el = time.time() - t0
        print(f"teacher {os.path.basename(sh)} done; {ntok/1e6:.2f}M tok {ntok/el:.0f} tok/s; nll {np.nanmean(nll):.4f}", flush=True)
        if a.stats:
            save_stats(experts, a.out)
    print("TEACHER_DONE", flush=True)


def save_stats(experts, out):
    st = {li: {"count": m.stats_count.cpu().tolist(), "reap": m.stats_reap.cpu().tolist(),
               "gate": m.stats_gate.cpu().tolist()} for li, m in experts.items()}
    json.dump(st, open(os.path.join(out, "routing_stats.json.tmp"), "w"))
    os.replace(os.path.join(out, "routing_stats.json.tmp"), os.path.join(out, "routing_stats.json"))


def cmd_select(a):
    st = json.load(open(a.stats))
    out = {}
    for crit in ("freq", "reap", "reapsum"):
        keep = {}
        for li, d in st.items():
            c = np.array(d["count"]); r = np.array(d["reap"])
            if crit == "freq":
                score = c
            elif crit == "reap":
                score = r / np.maximum(c, 1)
            else:
                score = r
            keep[int(li)] = sorted(np.argsort(-score)[:a.k].tolist())
        cov = np.mean([np.array(st[str(li)]["count"])[keep[li]].sum() / np.sum(st[str(li)]["count"]) for li in keep])
        print(f"{crit}: K={a.k} mean routing coverage {cov:.3f}", flush=True)
        json.dump(keep, open(os.path.join(a.out, f"keep_{crit}_{a.k}.json"), "w"))


def load_keep(path):
    return {int(k): v for k, v in json.load(open(path)).items()}


# ---------------------------------------------------------------- student
def prep_student(a):
    keep = load_keep(a.keep)
    model, experts = load_model(keep)
    STATE.student = True
    STATE.chunk = a.chunk
    for li, r in routers(model).items():
        km = torch.zeros(128, dtype=torch.bool, device=r.weight.device)
        km[torch.tensor(keep[li])] = True
        r.keep_mask = km
        if a.router_delta:
            r.delta = nn.Parameter(torch.zeros_like(r.weight, dtype=torch.float32))
    if a.rank > 0:
        for m in experts.values():
            m.add_lora(a.rank, a.alpha)
    return model, experts, keep


def trainable(model):
    lora = [p for n, p in model.named_parameters() if ".lora." in n and p.requires_grad]
    rd = [p for n, p in model.named_parameters() if n.endswith(".delta") and p.requires_grad]
    return lora, rd


def adapter_state(model):
    return {n: p.detach().cpu() for n, p in model.named_parameters() if (".lora." in n or n.endswith(".delta"))}


def kd_loss_chunk(model, h, idx, lp_t):
    """h [n, H] student hidden, idx/lp_t [n, K] teacher top-k -> (sum KL, n)."""
    lg = logits_from_hidden(model, h)
    ls = torch.log_softmax(lg, -1)
    s_top = ls.gather(-1, idx)
    lp_t = lp_t.float()
    p_t = lp_t.exp()
    pt_rest = (1 - p_t.sum(-1)).clamp_min(1e-6)
    ps_rest = (1 - s_top.exp().sum(-1)).clamp_min(1e-6)
    kl = (p_t * (lp_t - s_top)).sum(-1) + pt_rest * (pt_rest.log() - ps_rest.log())
    return kl.sum()


@torch.no_grad()
def evaluate(model, data_dir, tk_dir, names, max_seqs, bs):
    res = {}
    for name in names:
        x = np.load(os.path.join(data_dir, f"{name}_0000.npy"))[:max_seqs].astype(np.int64)
        t = dict(np.load(os.path.join(tk_dir, f"{name}_0000.npz")))
        kl_sum = n = top1 = 0.0
        nll_s = nll_t = 0.0
        for i in range(0, len(x), bs):
            ids = torch.from_numpy(x[i:i + bs]).cuda()
            h = model.model(input_ids=ids).last_hidden_state
            for j in range(h.shape[0]):
                k = i + j
                T1 = h.shape[1] - 1
                for c0 in range(0, T1, 256):
                    c1 = min(T1, c0 + 256)
                    idx = torch.from_numpy(t["idx"][k][c0:c1]).long().cuda()
                    lpt = torch.from_numpy(t["lp"][k][c0:c1]).cuda()
                    ls = torch.log_softmax(logits_from_hidden(model, h[j, c0:c1]), -1)
                    s_top = ls.gather(-1, idx)
                    p_t = lpt.float().exp()
                    pt_rest = (1 - p_t.sum(-1)).clamp_min(1e-6)
                    ps_rest = (1 - s_top.exp().sum(-1)).clamp_min(1e-6)
                    kl = (p_t * (lpt.float() - s_top)).sum(-1) + pt_rest * (pt_rest.log() - ps_rest.log())
                    kl_sum += kl.sum().item()
                    top1 += (ls.argmax(-1) == idx[:, 0]).sum().item()
                    nll_s += (-ls.gather(-1, ids[j, c0 + 1:c1 + 1, None])[:, 0]).sum().item()
                    n += ls.shape[0]
                nll_t += float(np.nansum(t["nll"][k][:T1]))
        res[name] = {"kl": kl_sum / n, "top1": top1 / n, "ppl_student": math.exp(nll_s / n),
                     "ppl_teacher": math.exp(nll_t / n), "tokens": int(n)}
        print(f"  eval {name}: KL {res[name]['kl']:.4f} top1 {res[name]['top1']:.4f} "
              f"PPL {res[name]['ppl_student']:.3f} (teacher {res[name]['ppl_teacher']:.3f})", flush=True)
    return res


def cmd_evalstu(a):
    model, experts, keep = prep_student(a)
    if a.ckpt:
        sd = torch.load(a.ckpt, map_location="cuda")
        missing = model.load_state_dict(sd["adapters"], strict=False)
        print("loaded", a.ckpt, "unexpected", missing.unexpected_keys[:3], flush=True)
    model.eval()
    res = evaluate(model, a.data, a.tk, a.evals.split(","), a.eval_seqs, a.bs)
    json.dump(res, open(a.out, "w"), indent=1)


def cmd_train(a):
    model, experts, keep = prep_student(a)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    lora, rd = trainable(model)
    print(f"trainable: lora {sum(p.numel() for p in lora)/1e6:.1f}M router {sum(p.numel() for p in rd)/1e6:.1f}M; "
          f"mem {torch.cuda.memory_allocated()/1e9:.1f} GB", flush=True)
    groups = [{"params": lora, "lr": a.lr}]
    if rd:
        groups.append({"params": rd, "lr": a.lr_router})
    opt = torch.optim.AdamW(groups, betas=(0.9, 0.99), weight_decay=0.0, eps=1e-8)
    base_lrs = [g["lr"] for g in opt.param_groups]
    os.makedirs(a.out, exist_ok=True)
    log = open(os.path.join(a.out, "train_log.jsonl"), "a")
    step, seen = 0, 0
    ck = os.path.join(a.out, "last.pt")
    if os.path.exists(ck):
        sd = torch.load(ck, map_location="cuda")
        model.load_state_dict(sd["adapters"], strict=False)
        opt.load_state_dict(sd["opt"])
        step, seen = sd["step"], sd["seen"]
        print(f"resumed at step {step} seen {seen}", flush=True)
    elif a.init and os.path.exists(a.init):  # off-VM adapter copy after a lost runtime; optimizer restarts
        sd = torch.load(a.init, map_location="cuda")
        model.load_state_dict({k: v.float() for k, v in sd["adapters"].items()}, strict=False)
        step, seen = sd["step"], sd["seen"]
        print(f"initialized from {a.init} at step {step} seen {seen} (fresh optimizer)", flush=True)

    def batches():
        """Yield (ids, idx, lp) micro-batches from teacher shards that exist, in order, forever-waiting."""
        done = set()
        k = 0
        while True:
            tks = sorted(glob.glob(os.path.join(a.tk, "train_*.npz")))
            new = [t for t in tks if t not in done]
            if not new:
                if os.path.exists(os.path.join(a.tk, "TEACHER_DONE")) or a.no_wait:
                    return
                time.sleep(30)
                continue
            for t in new:
                done.add(t)
                x = np.load(os.path.join(a.data, os.path.basename(t).replace(".npz", ".npy"))).astype(np.int64)
                d = dict(np.load(t))
                for i in range(0, len(x) - a.bs + 1, a.bs):
                    k += 1
                    yield x[i:i + a.bs], d["idx"][i:i + a.bs], d["lp"][i:i + a.bs]

    it = batches()
    skip = seen // (a.bs * a.seq)
    for _ in range(skip):
        next(it)
    t0 = time.time()
    tok0 = seen
    total_steps = a.steps
    while step < total_steps:
        lr_f = min(1.0, (step + 1) / a.warmup) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, step / total_steps))))
        for g, b in zip(opt.param_groups, base_lrs):
            g["lr"] = b * lr_f
        opt.zero_grad(set_to_none=True)
        loss_acc, n_acc = 0.0, 0
        stop = False
        for _ in range(a.accum):
            try:
                x, idx, lp = next(it)
            except StopIteration:
                stop = True
                break
            ids = torch.from_numpy(x).cuda()
            h = model.model(input_ids=ids).last_hidden_state[:, :-1]
            H = h.shape[-1]
            hf = h.reshape(-1, H)
            idx_t = torch.from_numpy(idx[:, :-1].reshape(-1, TOPK)).long().cuda()
            lp_t = torch.from_numpy(lp[:, :-1].reshape(-1, TOPK)).cuda()
            n = hf.shape[0]
            total = 0.0
            for c0 in range(0, n, a.loss_chunk):
                c1 = min(n, c0 + a.loss_chunk)
                total = total + torch.utils.checkpoint.checkpoint(
                    kd_loss_chunk, model, hf[c0:c1], idx_t[c0:c1], lp_t[c0:c1], use_reentrant=False)
            loss = total / (n * a.accum)
            loss.backward()
            loss_acc += total.item()
            n_acc += n
            seen += x.size
        if n_acc == 0:
            break
        gn = torch.nn.utils.clip_grad_norm_(lora + rd, a.clip)
        opt.step()
        step += 1
        el = time.time() - t0
        rec = {"step": step, "kl": loss_acc / n_acc, "gn": float(gn), "lr": opt.param_groups[0]["lr"],
               "seen": seen, "tok_s": (seen - tok0) / el, "mem": torch.cuda.max_memory_allocated() / 1e9, "t": time.time()}
        log.write(json.dumps(rec) + "\n"); log.flush()
        if step % 10 == 0 or step == 1:
            print(f"step {step} kl {rec['kl']:.4f} gn {rec['gn']:.3f} {rec['tok_s']:.0f} tok/s seen {seen/1e6:.2f}M "
                  f"maxmem {rec['mem']:.1f}GB", flush=True)
        if step % a.ckpt_every == 0 or step == total_steps or stop:
            sd = {"adapters": adapter_state(model), "opt": opt.state_dict(), "step": step, "seen": seen,
                  "keep": keep, "rank": a.rank, "alpha": a.alpha}
            torch.save(sd, ck + ".tmp"); os.replace(ck + ".tmp", ck)
            small = {"adapters": {k: v.to(torch.bfloat16) for k, v in sd["adapters"].items()}, "step": step,
                     "seen": seen, "keep": keep, "rank": a.rank, "alpha": a.alpha}
            lt = os.path.join(a.out, "adapters_latest.pt")  # pulled off the VM between check-ins
            torch.save(small, lt + ".tmp"); os.replace(lt + ".tmp", lt)
            if a.eval_every and step % a.eval_every == 0:
                torch.save(small, os.path.join(a.out, f"adapters_{step:05d}.pt"))
        if a.eval_every and (step % a.eval_every == 0 or step == total_steps or stop):
            model.eval()
            res = evaluate(model, a.data, a.tk, a.evals.split(","), a.eval_seqs, a.bs)
            model.train()
            log.write(json.dumps({"step": step, "seen": seen, "eval": res}) + "\n"); log.flush()
        if stop:
            break
    print("TRAIN_DONE", flush=True)


# ---------------------------------------------------------------- export
@torch.no_grad()
def cmd_export(a):
    """Healed experts for kept ids: gate/up/down bf16 + router rows, as safetensors per layer."""
    from safetensors.torch import save_file
    sd = torch.load(a.ckpt, map_location="cpu") if a.ckpt else None
    keep = load_keep(a.keep)
    from safetensors import safe_open
    os.makedirs(a.out, exist_ok=True)
    ad = sd["adapters"] if sd else {}
    rank = sd["rank"] if sd else 0
    scale = (sd["alpha"] / rank) if sd and rank else 0.0

    def find(li, key):
        for n, t in ad.items():
            if re.search(rf"layers\.{li}\.mlp\.{key}$", n):
                return t
        return None
    with safe_open(os.path.join(MODEL_DIR, os.path.basename(st_file())), framework="pt", device="cpu") as f:
        keys = set(f.keys())
        for li in range(24):
            ids = torch.tensor(sorted(keep[li]))
            tens = {}
            for proj in ("w13", "w2"):
                base = f"model.llm.layers.{li}.mlp.experts.{proj}_weight"
                d = {"weight": f.get_tensor(base)[ids].cuda(), "scale": f.get_tensor(base + "_scale")[ids].cuda()}
                if base + "_amax" in keys:
                    d["amax"] = f.get_tensor(base + "_amax")[ids].cuda()
                    d["den"] = f.get_tensor(base + "_outer_scale_denominator").cuda()
                W = dequant(d, 0, len(ids), dtype=torch.float32)
                A = find(li, "experts.lora.A13" if proj == "w13" else "experts.lora.A2")
                B = find(li, "experts.lora.B13" if proj == "w13" else "experts.lora.B2")
                if A is not None:
                    W = W + scale * torch.bmm(B.cuda().float(), A.cuda().float())
                if proj == "w13":
                    tens["gate"] = W[:, 0::2].to(torch.bfloat16).cpu().contiguous()
                    tens["up"] = W[:, 1::2].to(torch.bfloat16).cpu().contiguous()
                else:
                    tens["down"] = W.to(torch.bfloat16).cpu().contiguous()
            rw = f.get_tensor(f"model.llm.layers.{li}.mlp.gate.weight").float()
            dl = find(li, "gate.delta")
            if dl is not None:
                rw = rw + dl.float()
            tens["router"] = rw[ids].contiguous()
            tens["bias"] = f.get_tensor(f"model.llm.layers.{li}.mlp.gate.bias").float()[ids].contiguous()
            tens["ids"] = ids.to(torch.int32)
            save_file(tens, os.path.join(a.out, f"layer_{li:02d}.safetensors"))
            print("exported layer", li, flush=True)
    print("EXPORT_DONE", flush=True)


def main():
    p = argparse.ArgumentParser()
    sp = p.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("data")
    d.add_argument("--out", default="/content/data"); d.add_argument("--code-jsonl", default="/content/code.jsonl")
    d.add_argument("--heldout-list", default="/content/heldout_code_files.txt")
    d.add_argument("--seq", type=int, default=1024); d.add_argument("--train-tokens", type=float, default=40e6)
    d.add_argument("--eval-tokens", type=float, default=131072); d.add_argument("--shard-seqs", type=int, default=512)
    d.add_argument("--mix-code-own", type=float, default=0.2); d.add_argument("--mix-code-pub", type=float, default=0.2)
    d.add_argument("--mix-chat", type=float, default=0.3); d.add_argument("--mix-web", type=float, default=0.3)
    t = sp.add_parser("teacher")
    t.add_argument("--data", default="/content/data"); t.add_argument("--pattern", default="*.npy")
    t.add_argument("--out", default="/content/tk"); t.add_argument("--bs", type=int, default=2)
    t.add_argument("--stats", action="store_true"); t.add_argument("--lchunk", type=int, default=256)
    t.add_argument("--reverse", action="store_true")
    s = sp.add_parser("select")
    s.add_argument("--stats", default="/content/tk/routing_stats.json"); s.add_argument("--k", type=int, default=96)
    s.add_argument("--out", default="/content")
    for name in ("train", "evalstu"):
        q = sp.add_parser(name)
        q.add_argument("--keep", required=True); q.add_argument("--data", default="/content/data")
        q.add_argument("--tk", default="/content/tk"); q.add_argument("--rank", type=int, default=8)
        q.add_argument("--alpha", type=float, default=16); q.add_argument("--router-delta", type=int, default=1)
        q.add_argument("--bs", type=int, default=2); q.add_argument("--chunk", type=int, default=16)
        q.add_argument("--evals", default="eval_code,eval_wiki,eval_chat"); q.add_argument("--eval-seqs", type=int, default=64)
        if name == "train":
            q.add_argument("--out", default="/content/run"); q.add_argument("--seq", type=int, default=1024)
            q.add_argument("--accum", type=int, default=4); q.add_argument("--steps", type=int, default=2000)
            q.add_argument("--lr", type=float, default=2e-4); q.add_argument("--lr-router", type=float, default=5e-5)
            q.add_argument("--warmup", type=int, default=30); q.add_argument("--clip", type=float, default=1.0)
            q.add_argument("--loss-chunk", type=int, default=512); q.add_argument("--ckpt-every", type=int, default=50)
            q.add_argument("--eval-every", type=int, default=200); q.add_argument("--no-wait", action="store_true")
            q.add_argument("--init", default=None)
        else:
            q.add_argument("--ckpt", default=None); q.add_argument("--out", default="/content/eval.json")
    x = sp.add_parser("export")
    x.add_argument("--keep", required=True); x.add_argument("--ckpt", default=None); x.add_argument("--out", default="/content/export")
    a = p.parse_args()
    {"data": cmd_data, "teacher": cmd_teacher, "select": cmd_select, "train": cmd_train,
     "evalstu": cmd_evalstu, "export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
