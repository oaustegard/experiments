#!/usr/bin/env python3
"""The ane-context-filter student (Ettin-17m cross-encoder) as a Core ML model for the Neural Engine.

The student scores one (task, chunk) pair per sequence: ModernBERT encoder, the [CLS]
vector through a small MLP head, at most MAX_LEN = 384 tokens (ane-context-filter/train.py).
This script rewrites it in the layout Core ML keeps on the Neural Engine, checks the rewrite
against the Hugging Face model in PyTorch, converts it at fixed shapes, and on a Mac prints
where Core ML places each operation and how long a pair takes with CPU_AND_NE and CPU_ONLY,
GPU idle and GPU busy.

    python3 crossenc_coreml.py --weights ~/private/student17-fp16.pt   # the trained student
    python3 crossenc_coreml.py                                         # stock Ettin-17m, random head

The weights are private (an fp16 state dict of train.py's Scorer). Without --weights the
encoder is stock jhu-clsp/ettin-encoder-17m and the head is random (seed 0): placement and
timing are the same, the scores mean nothing.

What the rewrite changes (ScorerANE below; the layout is ettin-ane/run/ane_model.py's):
  - activations are (B, C, 1, S), every Linear a 1x1 convolution, attention per head;
  - sequences stay padded to S; padding is an additive -1e4 on the key axis, so
    ModernBERT's unpadding path is not needed;
  - the RoPE tables and the local-attention window mask are constants for that S;
  - norm eps is at least 1e-5, and the norms come in two variants, both converted and
    timed: "plain" (coremltools recognises it and emits layer_norm ops; ettin-ane ran this
    form on an M1) and "prescaled", where the centred input is divided by a constant power
    of two before it is squared. Centred inputs reach about 1,100 here, whose square is past
    fp16's 65,504: evaluated literally in fp16 the plain form is off by 0.38 in the logit,
    the prescaled one by 0.003. Whether Core ML's layer_norm op has that problem on the
    Neural Engine is what the Mac run shows (max prob diff column);
  - the head runs on the [CLS] position only.

On Linux the script stops after the conversion (Core ML prediction needs macOS); that run
is the check that the graph converts. Phases: torch parity -> convert -> (macOS) per
compute unit: load, placement (MLComputePlan), parity against PyTorch, ms per pair.

Requires: python3 >= 3.10. Makes its own venv in work/venv-crossenc.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import common
from common import log

PINS = ["torch==2.7.0", "transformers==4.57.6", "coremltools==9.0", "numpy<2.4"]
MAX_LEN = 384
TASK_TOKENS = 120
TOL = 1e-4

# Invented (task, chunk header, chunk body) texts in the student's input format. The real
# pairs come from session transcripts and stay private; pass --pairs for those.
SAMPLE_TASKS = [
    "Find where the retry backoff is configured and raise the cap to 30 seconds. Report the file and the old value.",
    "Write a benchmark script for the tokenizer and commit it. Use the same environment checks as the existing one.",
    "Summarise what the migration changed in the schema and list any columns that lost their index.",
    "Check whether the nightly job still passes with the new cache path, and say what failed if not.",
]
SAMPLE_CHUNKS = [
    ("<tool Bash | age<=3> ", "input: grep -rn backoff src/\nresult: src/net/retry.py:41:    backoff = min(cap, base * 2 ** attempt)\nsrc/net/retry.py:12:CAP_SECONDS = 10"),
    ("<assistant | age<=10> ", "The cache path moved from /var/cache/app to /srv/cache/app in the last deploy; the nightly job reads the path from settings.toml, which was not updated."),
    ("<user | age<=1> ", "Use the checks the other benchmark has: refuse to run on battery, record the chip and the OS version, and write the results next to the script."),
    ("<tool Read | age<=30> ", "input: db/migrations/0042_split_orders.sql\nresult: ALTER TABLE orders ADD COLUMN region TEXT;\nCREATE TABLE order_lines (id INTEGER PRIMARY KEY, order_id INTEGER, sku TEXT, qty INTEGER);\nDROP INDEX orders_customer_idx;"),
    ("<tool Bash | age>300> ", "input: ls\nresult: README.md\nsrc\ntests\npyproject.toml"),
    ("<assistant | age<=100> ", "I ran the test suite twice. " + "The second run passed all 412 tests in 38 seconds, and the flaky one did not recur. " * 12),
    ("<tool Bash | age<=3> ", "input: pytest -q\nresult: " + "tests/test_retry.py::test_backoff_cap PASSED\ntests/test_cache.py::test_path FAILED\n" * 40),
]


def build_models(args):
    """(reference Scorer, ScorerANE factory, tokenizer). Imports torch lazily (venv)."""
    import torch
    from torch import nn
    from transformers import AutoModel, AutoTokenizer

    class Scorer(nn.Module):
        """ane-context-filter/train.py's Scorer, unchanged: the reference."""

        def __init__(self, init):
            super().__init__()
            self.enc = AutoModel.from_pretrained(init, attn_implementation="sdpa")
            h = self.enc.config.hidden_size
            self.head = nn.Sequential(nn.Linear(h, h), nn.GELU(), nn.LayerNorm(h), nn.Linear(h, 1))

        def forward(self, ids, att):
            return self.head(self.enc(input_ids=ids, attention_mask=att).last_hidden_state[:, 0]).squeeze(-1)

    class LN(nn.Module):
        """LayerNorm over the channel axis of (B, C, 1, S). `pre` scales the centred input
        before it is squared and scales eps to match, so the result is unchanged."""

        def __init__(self, w, b=None, eps=1e-5):
            super().__init__()
            self.w = nn.Parameter(w.reshape(1, -1, 1, 1).clone())
            self.b = None if b is None else nn.Parameter(b.reshape(1, -1, 1, 1).clone())
            self.eps, self.pre, self.peak = max(float(eps), 1e-5), 1.0, 0.0

        def forward(self, x):
            x = x - x.mean(dim=1, keepdim=True)
            if not torch.jit.is_tracing():
                self.peak = max(self.peak, float(x.detach().abs().max()))
            if self.pre != 1.0:
                x = x * self.pre
            x = x * torch.rsqrt((x * x).mean(dim=1, keepdim=True) + self.eps * self.pre ** 2) * self.w
            return x if self.b is None else x + self.b

    def conv(w, b=None):
        o, i = w.shape
        c = nn.Conv2d(i, o, 1, bias=b is not None)
        c.weight.data = w.reshape(o, i, 1, 1).clone()
        if b is not None:
            c.bias.data = b.clone()
        return c

    class Layer(nn.Module):
        def __init__(self, sd, i, heads, is_global, S, window, theta, eps):
            super().__init__()
            g = lambda k: sd[f"layers.{i}.{k}"]  # noqa: E731
            self.attn_norm = LN(g("attn_norm.weight"), eps=eps) if f"layers.{i}.attn_norm.weight" in sd else None
            self.qkv, self.o = conv(g("attn.Wqkv.weight")), conv(g("attn.Wo.weight"))
            self.mlp_norm = LN(g("mlp_norm.weight"), eps=eps)
            self.wi, self.wo = conv(g("mlp.Wi.weight")), conv(g("mlp.Wo.weight"))
            H = self.o.weight.shape[0]
            self.heads, self.d = heads, H // heads
            inv = 1.0 / (theta ** (torch.arange(0, self.d, 2).float() / self.d))
            f = torch.outer(torch.arange(S).float(), inv)
            emb = torch.cat([f, f], -1).T  # (d, S)
            self.register_buffer("cos", emb.cos().reshape(1, self.d, 1, S))
            self.register_buffer("sin", emb.sin().reshape(1, self.d, 1, S))
            if is_global:
                self.local = None
            else:
                idx = torch.arange(S)
                far = (idx[:, None] - idx[None, :]).abs() > window // 2  # (key, query)
                self.register_buffer("local", (far.float() * -1e4).reshape(1, S, 1, S))

        def rope(self, x):
            a, b = x[:, : self.d // 2], x[:, self.d // 2:]
            return x * self.cos + torch.cat([-b, a], 1) * self.sin

        def forward(self, x, key_mask):  # key_mask (B, S, 1, 1): 0 or -1e4
            h = x if self.attn_norm is None else self.attn_norm(x)
            q, k, v = self.qkv(h).split(self.heads * self.d, dim=1)
            bias = key_mask if self.local is None else key_mask + self.local
            outs = []
            for qh, kh, vh in zip(q.split(self.d, 1), k.split(self.d, 1), v.split(self.d, 1)):
                qh, kh = self.rope(qh) * (self.d ** -0.5), self.rope(kh)
                w = torch.einsum("bchq,bkhc->bkhq", qh, kh.transpose(1, 3)) + bias  # (B, S_key, 1, S_query)
                outs.append(torch.einsum("bkhq,bchk->bchq", w.softmax(dim=1), vh))
            x = x + self.o(torch.cat(outs, 1))
            a, gate = self.wi(self.mlp_norm(x)).chunk(2, dim=1)
            return x + self.wo(nn.functional.gelu(a) * gate)

    class ScorerANE(nn.Module):
        def __init__(self, ref, S):
            super().__init__()
            c, sd = ref.enc.config, ref.enc.state_dict()
            rp = getattr(c, "rope_parameters", None) or {}  # transformers 5 nests the thetas

            def theta(kind):
                v = getattr(c, f"{kind}_rope_theta", None)
                if v is None:
                    v = (rp.get(f"{kind}_attention") or rp.get(kind) or rp).get("rope_theta")
                return float(v)

            self.S = S
            self.emb = nn.Embedding.from_pretrained(sd["embeddings.tok_embeddings.weight"].clone())
            self.emb_norm = LN(sd["embeddings.norm.weight"], sd.get("embeddings.norm.bias"), eps=c.norm_eps)
            every = c.global_attn_every_n_layers
            self.layers = nn.ModuleList(
                Layer(sd, i, c.num_attention_heads, i % every == 0, S, c.local_attention,
                      theta("global" if i % every == 0 else "local"), c.norm_eps)
                for i in range(c.num_hidden_layers))
            self.final_norm = LN(sd["final_norm.weight"], sd.get("final_norm.bias"), eps=c.norm_eps)
            hs = ref.head.state_dict()
            self.h0 = conv(hs["0.weight"], hs["0.bias"])
            self.hln = LN(hs["2.weight"], hs["2.bias"], eps=ref.head[2].eps)
            self.h1 = conv(hs["3.weight"], hs["3.bias"])

        def norms(self):
            return [m for m in self.modules() if isinstance(m, LN)]

        def forward(self, input_ids, pad_mask):
            """input_ids (B, S) int; pad_mask (B, S) float, 1 for real tokens. Logits (B,)."""
            x = self.emb_norm(self.emb(input_ids).transpose(1, 2).unsqueeze(2))  # (B, H, 1, S)
            key_mask = ((1.0 - pad_mask) * -1e4).unsqueeze(-1).unsqueeze(-1)  # (B, S, 1, 1)
            for layer in self.layers:
                x = layer(x, key_mask)
            x = self.final_norm(x[:, :, :, 0:1])  # the [CLS] column
            return self.h1(self.hln(nn.functional.gelu(self.h0(x)))).reshape(-1)

    torch.manual_seed(0)  # the random head, when there are no weights
    ref = Scorer(args.init)
    if args.weights:
        sd = torch.load(args.weights, map_location="cpu", weights_only=True)
        sd = sd.get("model", sd)
        ref.load_state_dict({k: v.float() for k, v in sd.items()}, strict=True)
    ref = ref.float().eval()
    tok = AutoTokenizer.from_pretrained(args.init)
    return ref, ScorerANE, tok


def encode_pairs(tok, pairs, S):
    """train.py's Encoder: [CLS] task [SEP] header+body [SEP], padded to S. (ids, mask) int64."""
    import torch

    ids = torch.full((len(pairs), S), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(pairs), S), dtype=torch.long)
    for i, (task, chunk) in enumerate(pairs):
        t = tok(task, add_special_tokens=False)["input_ids"]
        if len(t) > TASK_TOKENS:
            t = t[: TASK_TOKENS * 2 // 3] + t[-(TASK_TOKENS - TASK_TOKENS * 2 // 3):]
        b = tok(chunk, add_special_tokens=False)["input_ids"][: S - len(t) - 3]
        row = [tok.cls_token_id] + t + [tok.sep_token_id] + b + [tok.sep_token_id]
        ids[i, : len(row)] = torch.tensor(row)
        att[i, : len(row)] = 1
    return ids, att


def sample_pairs(args):
    if args.pairs:
        rows = [json.loads(l) for l in open(args.pairs)]
        return [(r["task"], r["chunk"]) for r in rows]
    return [(t, h + b) for t in SAMPLE_TASKS for h, b in SAMPLE_CHUNKS]


def torch_parity(ref, ane, ids, att):
    """Reference vs rewrite in fp32 on padded batches, then the rewrite in fp16."""
    import torch

    out = {}
    with torch.no_grad():
        want = ref(ids, att)
        got = ane(ids, att.float())
        # the reference again on each pair unpadded: padding must not move the score
        solo = torch.stack([ref(ids[i:i + 1, : int(att[i].sum())], att[i:i + 1, : int(att[i].sum())])[0]
                            for i in range(len(ids))])
    out["pairs"] = len(ids)
    out["tokens_min_max"] = [int(att.sum(1).min()), int(att.sum(1).max())]
    out["max_abs_logit_diff"] = float((want - got).abs().max())
    out["max_abs_prob_diff"] = float((want.sigmoid() - got.sigmoid()).abs().max())
    out["max_abs_logit_diff_vs_unpadded"] = float((solo - got).abs().max())
    out["logit_range"] = [float(want.min()), float(want.max())]
    out["norm_input_peak"] = round(max(m.peak for m in ane.norms()), 2)
    try:  # fp16 arithmetic in PyTorch: an estimate of what the fp16 Core ML model will do
        import copy

        half = copy.deepcopy(ane).half()
        with torch.no_grad():
            g16 = half(ids, att.half()).float()
        out["fp16_max_abs_logit_diff"] = float((want - g16).abs().max())
        out["fp16_max_abs_prob_diff"] = float((want.sigmoid() - g16.sigmoid()).abs().max())
        out["fp16_finite"] = bool(torch.isfinite(g16).all())
    except Exception as e:  # noqa: BLE001 - CPU fp16 kernels vary by torch build
        out["fp16_error"] = repr(e)
    return out


def calibrate_norms(ane, ids, att):
    """Pick each norm's power-of-two pre-scale so the largest centred input seen, with 4x
    headroom, squares to under 1e4 (fp16 tops out at 65504). Returns {norm: (peak, pre)}."""
    import torch

    with torch.no_grad():
        ane(ids, att.float())
    rec = {}
    for name, m in ane.named_modules():
        if hasattr(m, "peak"):
            limit = 100.0 / 4
            m.pre = 1.0 if m.peak <= limit else 2.0 ** -math.ceil(math.log2(m.peak / limit))
            rec[name] = [round(m.peak, 2), m.pre]
            m.peak = 0.0
    return rec


def convert(ane, B, S, path, args):
    import numpy as np
    import torch
    import coremltools as ct

    ex = (torch.zeros((B, S), dtype=torch.int32), torch.ones((B, S), dtype=torch.float32))
    traced = torch.jit.trace(ane, ex, check_trace=False)
    t = time.time()
    m = ct.convert(
        traced, convert_to="mlprogram",
        inputs=[ct.TensorType(name="input_ids", shape=(B, S), dtype=np.int32),
                ct.TensorType(name="pad_mask", shape=(B, S), dtype=np.float32)],
        outputs=[ct.TensorType(name="logits")],
        compute_precision=ct.precision.FLOAT16 if args.precision == "fp16" else ct.precision.FLOAT32,
        minimum_deployment_target=ct.target.macOS15, skip_model_load=True)
    m.save(str(path))
    ops = {}
    prog = m._mil_program
    if prog is not None:
        for fn in prog.functions.values():
            for op in fn.operations:
                ops[op.op_type] = ops.get(op.op_type, 0) + 1
    size = sum(f.stat().st_size for f in Path(path).rglob("*") if f.is_file())
    return {"convert_s": round(time.time() - t, 1), "package_mb": round(size / 2**20, 1),
            "mil_ops": sum(v for k, v in ops.items() if k != "const"), "mil_op_types": ops}


def placement(path, cu):
    """Operations per preferred compute device, and the estimated cost share on each."""
    import coremltools as ct
    from coremltools.models.compute_plan import MLComputePlan

    compiled = ct.models.MLModel(str(path), compute_units=cu).get_compiled_model_path()
    plan = MLComputePlan.load_from_path(path=compiled, compute_units=cu)
    counts, cost, off = {}, {}, {}
    for fn in plan.model_structure.program.functions.values():
        for op in fn.block.operations:
            if op.operator_name == "const":
                continue
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            d = type(u.preferred_compute_device).__name__ if u else "none"
            counts[d] = counts.get(d, 0) + 1
            c = plan.get_estimated_cost_for_mlprogram_operation(op)
            if c is not None:
                cost[d] = cost.get(d, 0.0) + c.weight
            if "NeuralEngine" not in d:
                off.setdefault(d, {})
                off[d][op.operator_name] = off[d].get(op.operator_name, 0) + 1
    return {"ops": counts, "cost_share": {k: round(v, 4) for k, v in cost.items()},
            "not_on_neural_engine": off}


def time_model(path, cu, feed, want, seconds, load):
    """Load twice (first load compiles for the device), check against PyTorch, time predict."""
    import numpy as np
    import coremltools as ct

    r = {}
    loads = []
    for _ in range(2):
        t = time.perf_counter()
        m = ct.models.MLModel(str(path), compute_units=cu)
        loads.append(round(time.perf_counter() - t, 3))
    r["load_s_first"], r["load_s_second"] = loads
    got = np.asarray(m.predict(feed)["logits"], dtype=np.float32).reshape(-1)
    r["max_abs_logit_diff_vs_torch"] = float(np.abs(got - want).max())
    sig = lambda z: 1 / (1 + np.exp(-z))  # noqa: E731
    r["max_abs_prob_diff_vs_torch"] = float(np.abs(sig(got) - sig(want)).max())
    r["finite"] = bool(np.isfinite(got).all())
    for _ in range(5):
        m.predict(feed)
    if load is not None:
        load.wait_busy()
    ts, t0 = [], time.time()
    while time.time() - t0 < seconds or len(ts) < 20:
        t = time.perf_counter()
        m.predict(feed)
        ts.append(time.perf_counter() - t)
    t1 = time.time()
    B = feed["input_ids"].shape[0]
    ts.sort()
    r.update({"calls": len(ts), "ms_per_call_median": round(1e3 * common.median(ts), 3),
              "ms_per_call_p90": round(1e3 * ts[int(0.9 * (len(ts) - 1))], 3),
              "ms_per_pair": round(1e3 * common.median(ts) / B, 3),
              "pairs_per_s": round(B / common.median(ts), 1)})
    r.update(common.gpu_record(load, t0, t1))
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights", type=Path, help="state dict of train.py's Scorer (private); "
                    "default: stock encoder, random head")
    ap.add_argument("--init", default="jhu-clsp/ettin-encoder-17m")
    ap.add_argument("--seq", type=int, default=MAX_LEN)
    ap.add_argument("--batches", default="1,16", help="fixed batch sizes, one package each")
    ap.add_argument("--precision", choices=["fp16", "fp32"], default="fp16")
    ap.add_argument("--norms", default="plain,prescaled",
                    help="plain: the textbook norm, which coremltools turns into layer_norm ops "
                    "(what ettin-ane ran); prescaled: divided by a constant before squaring")
    ap.add_argument("--units", default="CPU_AND_NE,CPU_ONLY",
                    help="also available: CPU_AND_GPU, ALL")
    ap.add_argument("--seconds", type=float, default=15, help="timed seconds per configuration")
    ap.add_argument("--pairs", type=Path, help="jsonl of {task, chunk} for the parity check "
                    "(default: invented pairs)")
    ap.add_argument("--convert-only", action="store_true", help="stop after the conversion")
    common.add_gpu_args(ap)
    args = ap.parse_args()
    common.reexec_in_venv("crossenc", PINS)

    import numpy as np
    import torch
    import transformers
    import coremltools as ct

    mac = common.is_mac()
    env, out = common.start("crossenc", args, mac_only=False)
    S = args.seq
    summ = {"env": env, "args": {k: str(v) for k, v in vars(args).items()},
            "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                         "coremltools": ct.__version__, "numpy": np.__version__},
            "weights": "student (private)" if args.weights else "stock encoder, random head"}

    log(f"building the reference and the rewrite ({summ['weights']})")
    ref, ScorerANE, tok = build_models(args)
    ids, att = encode_pairs(tok, sample_pairs(args), S)
    variants, summ["torch_parity"], summ["norm_prescale"] = {}, {}, None
    for v in args.norms.split(","):
        ane = ScorerANE(ref, S).eval()
        scales = calibrate_norms(ane, ids, att)
        if v == "plain":
            for m in ane.norms():
                m.pre = 1.0
        else:
            summ["norm_prescale"] = scales
        par = torch_parity(ref, ane, ids, att)
        summ["torch_parity"][v] = par
        variants[v] = ane
        log(f"{v} norms, torch parity on {par['pairs']} pairs ({par['tokens_min_max']} tokens): max |logit diff| "
            f"{par['max_abs_logit_diff']:.2e} (vs unpadded reference {par['max_abs_logit_diff_vs_unpadded']:.2e}), "
            f"fp16 in torch {par.get('fp16_max_abs_logit_diff', par.get('fp16_error'))}")
        if par["max_abs_logit_diff"] > TOL or par["max_abs_logit_diff_vs_unpadded"] > TOL:
            sys.exit(f"the rewrite differs from the Hugging Face model by more than {TOL}; not converting")
    (out / "summary.json").write_text(json.dumps(summ, indent=1))

    pk = common.WORK / "crossenc"
    pk.mkdir(parents=True, exist_ok=True)
    batches = [int(b) for b in args.batches.split(",")]
    summ["models"] = {}
    for v, ane in variants.items():
        for B in batches:
            path = pk / f"scorer-{'student' if args.weights else 'stock'}-{v}-b{B}-s{S}-{args.precision}.mlpackage"
            log(f"converting {v} norms, batch {B}, {S} tokens -> {path.name}")
            try:
                M = {"package": path.name, "norms": v, "batch": B, **convert(ane, B, S, path, args)}
            except Exception as e:  # noqa: BLE001
                M = {"norms": v, "batch": B, "error": repr(e)}
                log(f"conversion failed: {e!r}")
            summ["models"][f"{v}-b{B}"] = M
            (out / "summary.json").write_text(json.dumps(summ, indent=1))

    if not mac or args.convert_only:
        summ["not_run"] = ("prediction, placement and timing need macOS"
                           if not mac else "--convert-only")
        common.finish(out, summ, summarize(summ))
        return

    units = [u for u in args.units.split(",") if u]
    with torch.no_grad():
        want_all = ref(ids, att).numpy()

    def feed_for(B):
        rep = (ids.repeat((B // len(ids)) + 1, 1)[:B], att.repeat((B // len(att)) + 1, 1)[:B])
        return ({"input_ids": rep[0].numpy().astype(np.int32),
                 "pad_mask": rep[1].numpy().astype(np.float32)}, np.resize(want_all, B))

    ok = {k: M for k, M in summ["models"].items() if "error" not in M}
    for k, M in ok.items():
        M["placement"] = {}
        for u in units:
            try:
                M["placement"][u] = placement(pk / M["package"], getattr(ct.ComputeUnit, u))
                log(f"{k} {u} placement: {M['placement'][u]['ops']}")
            except Exception as e:  # noqa: BLE001
                M["placement"][u] = {"error": repr(e)}
    for cond, load in common.gpu_conditions(args, out / "raw"):
        if isinstance(load, Exception):
            summ["busy_error"] = repr(load)
            log(f"GPU-busy condition skipped: {load!r}")
            continue
        for k, M in ok.items():
            feed, want = feed_for(M["batch"])
            for u in units:
                log(f"timing {k} {u}, GPU {cond}")
                try:
                    r = time_model(pk / M["package"], getattr(ct.ComputeUnit, u), feed, want, args.seconds, load)
                except Exception as e:  # noqa: BLE001
                    r = {"error": repr(e)}
                M.setdefault("timing", {}).setdefault(cond, {})[u] = r
                (out / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    common.finish(out, summ, summarize(summ))


def summarize(S):
    e = S["env"]
    L = [f"# Cross-encoder on Core ML: {e.get('chip') or e.get('platform')}, {e['timestamp']}", "",
         f"Weights: {S['weights']}. macOS {e.get('macos')}, AC {e.get('on_ac_power')}, "
         f"low power {e.get('low_power_mode')}. " + ", ".join(f"{k} {v}" for k, v in S["versions"].items()) + ".", ""]
    for v, p in S["torch_parity"].items():
        L.append(f"- PyTorch, {v} norms against the Hugging Face model, {p['pairs']} pairs of "
                 f"{p['tokens_min_max'][0]} to {p['tokens_min_max'][1]} tokens: max logit difference "
                 f"{p['max_abs_logit_diff']:.2e} ({p['max_abs_logit_diff_vs_unpadded']:.2e} against the model run "
                 f"unpadded); the same module in PyTorch fp16: {p.get('fp16_max_abs_logit_diff', p.get('fp16_error'))}. "
                 f"Largest centred norm input {p['norm_input_peak']}.")
    L.append("")
    for B, M in S.get("models", {}).items():
        if "error" in M:
            L += [f"{B}: conversion failed: {M['error'][:300]}", ""]
            continue
        L += [f"## {B}: {M['package']}", "",
              f"Converted in {M['convert_s']} s, {M['package_mb']} MB, {M['mil_ops']} operations.", ""]
        for u, pl in (M.get("placement") or {}).items():
            L.append(f"- Placement with {u}: " + (pl.get("error") or
                     f"{pl['ops']}; cost share {pl['cost_share']}; off the Neural Engine: {pl['not_on_neural_engine']}"))
        if M.get("timing"):
            L += ["", "| GPU | compute units | ms/pair | pairs/s | ms/call p50 (p90) | load s first / second | "
                  "max prob diff vs PyTorch | GPU tok/s alone -> during |", "|---|---|---|---|---|---|---|---|"]
            for cond, per in M["timing"].items():
                for u, r in per.items():
                    if "error" in r:
                        L.append(f"| {cond} | {u} | failed: {r['error'][:120]} | | | | | |")
                        continue
                    g = (f"{r.get('gpu_alone_tok_s')} -> {r.get('gpu_during_tok_s')} "
                         f"(busy {r.get('gpu_busy_fraction')})" if cond == "busy" else "")
                    L.append(f"| {cond} | {u} | {r['ms_per_pair']} | {r['pairs_per_s']} | "
                             f"{r['ms_per_call_median']} ({r['ms_per_call_p90']}) | "
                             f"{r['load_s_first']} / {r['load_s_second']} | "
                             f"{r['max_abs_prob_diff_vs_torch']:.2e} | {g} |")
        L.append("")
    if S.get("not_run"):
        L.append(f"Not run: {S['not_run']}.")
    if S.get("busy_error"):
        L.append(f"GPU-busy condition skipped: {S['busy_error']}")
    return "\n".join(L)


if __name__ == "__main__":
    main()
