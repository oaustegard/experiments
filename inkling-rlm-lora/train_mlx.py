#!/usr/bin/env python3
"""Interruptible LoRA SFT of Inkling 2 Mini on RLM traces, on Apple silicon with MLX.

Three subcommands:

  prep      tokenize traces from a1zhang/synthetic-rlm-sft-data into ids + loss mask
  train     LoRA training that checkpoints adapters, optimizer state and data position;
            Ctrl-C (or SIGTERM, or a closed lid with caffeinate off) loses at most
            --save-every optimizer steps, and rerunning the same command resumes
  selftest  check on a tiny random MoE model that an interrupted-and-resumed run ends
            with the same adapter as an uninterrupted one (runs anywhere MLX runs)

Needs the mlx-lm branch that adds inkling2_loop (ml-explore/mlx-lm#2086) for the real
model; selftest needs only released mlx-lm. See README.md for the recipe.
"""
import argparse
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

DATASET = "a1zhang/synthetic-rlm-sft-data"
# Module paths that match this are never given an adapter: the float32 router and
# Inkling's unquantized loop message gates.
DEFAULT_EXCLUDE = r"(^|\.)(router|gate|loop_gate\w*|message_gate\w*)$"


# ----------------------------------------------------------------------------- prep

def render(tok, messages, gen_prompt):
    ids = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen_prompt)
    if isinstance(ids, dict):
        ids = ids["input_ids"]
    return list(ids)


def tokenize_trace(tok, messages, train_from_turn):
    """ids, mask with mask=1 on assistant tokens. train_from_turn >= 0 is read as the
    index of the first assistant turn to train on (-1: all of them)."""
    ids, mask, a_idx = [], [], 0
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        head = render(tok, messages[:i], True)
        full = render(tok, messages[: i + 1], False)
        if head[: len(ids)] != ids or full[: len(head)] != head:
            return None  # template is not prefix-stable here; drop the trace
        on = 1 if train_from_turn < 0 or a_idx >= train_from_turn else 0
        mask += [0] * (len(head) - len(ids)) + [on] * (len(full) - len(head))
        ids = full
        a_idx += 1
    return (ids, mask) if sum(mask) else None


def cmd_prep(a):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)
    rng = np.random.default_rng(a.seed)
    train, val, stats = [], [], {}
    for cfg in a.configs.split(","):
        path = hf_hub_download(DATASET, f"data/{cfg}.jsonl", repo_type="dataset")
        rows = [json.loads(l) for l in open(path)]
        rows = [r for r in rows if r.get("reward", 0) >= 1.0 and not r.get("broken_code")]
        rng.shuffle(rows)
        kept = dropped = 0
        for r in rows:
            if kept >= a.n_per_config + a.n_val:
                break
            t = tokenize_trace(tok, r["messages"], r.get("train_from_turn", -1))
            if t is None or len(t[0]) > a.max_tokens:
                dropped += 1
                continue
            rec = {"cfg": cfg, "id": r.get("example_id"), "ids": t[0], "mask": t[1]}
            (val if kept < a.n_val else train).append(rec)
            kept += 1
        stats[cfg] = {"kept": kept, "dropped": dropped}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in (("train", train), ("val", val)):
        with open(out / f"{name}.jsonl", "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
    ntok = sum(len(r["ids"]) for r in train)
    print(json.dumps({"configs": stats, "train": len(train), "val": len(val),
                      "train_tokens": ntok, "mean_len": ntok // max(len(train), 1)}))


# ----------------------------------------------------------------------------- train

def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if p and Path(p).exists() else []


def add_lora(model, rank, alpha, dropout, exclude):
    import re
    from mlx_lm.tuner.utils import linear_to_lora_layers
    from mlx_lm.models.switch_layers import QuantizedSwitchLinear, SwitchLinear

    model.freeze()
    types = (nn.Linear, nn.QuantizedLinear, SwitchLinear, QuantizedSwitchLinear)
    pat = re.compile(exclude)
    keys = set()
    for layer in model.layers:
        for k, m in layer.named_modules():
            if isinstance(m, types) and not pat.search(k):
                keys.add(k)
    cfg = {"rank": rank, "scale": alpha / rank, "dropout": dropout, "keys": sorted(keys)}
    linear_to_lora_layers(model, len(model.layers), cfg)
    return cfg


def masked_loss(model, ids, mask):
    logits = model(ids[None, :-1]).astype(mx.float32)
    tgt, m = ids[None, 1:], mask[None, 1:].astype(mx.float32)
    ce = nn.losses.cross_entropy(logits, tgt, reduction="none")
    n = m.sum()
    return (ce * m).sum() / mx.maximum(n, 1), n


def save_ckpt(out, step, model, opt, state, keep=2):
    """Write ckpt-<step>/ then repoint LATEST atomically; a crash mid-save leaves the
    previous checkpoint in force."""
    d = out / f"ckpt-{step:06d}"
    tmp = out / f".tmp-{step:06d}"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    mx.save_safetensors(str(tmp / "adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    mx.save_safetensors(str(tmp / "optimizer.safetensors"), dict(tree_flatten(opt.state)))
    (tmp / "state.json").write_text(json.dumps(state))
    shutil.rmtree(d, ignore_errors=True)
    os.replace(tmp, d)
    (out / ".LATEST.tmp").write_text(d.name)
    os.replace(out / ".LATEST.tmp", out / "LATEST")
    for old in sorted(out.glob("ckpt-*"))[:-keep]:
        shutil.rmtree(old, ignore_errors=True)


def write_adapter(out, model, lora_cfg):
    mx.save_safetensors(str(out / "adapters.safetensors"),
                        dict(tree_flatten(model.trainable_parameters())))
    (out / "adapter_config.json").write_text(json.dumps({
        "fine_tune_type": "lora", "num_layers": len(model.layers),
        "lora_parameters": lora_cfg}, indent=1))


def train(a, model=None):
    if model is None:
        from mlx_lm.utils import load_model
        from huggingface_hub import snapshot_download
        p = Path(a.model) if Path(a.model).exists() else Path(snapshot_download(a.model))
        model, _ = load_model(p)
    if mx.metal.is_available():
        mx.set_wired_limit(mx.device_info()["max_recommended_working_set_size"])

    data, val = load_jsonl(a.data), load_jsonl(a.val)
    lora_cfg = add_lora(model, a.rank, a.alpha, a.dropout, a.exclude)
    n_train = sum(v.size for _, v in tree_flatten(model.trainable_parameters()))
    print(f"adapter: {len(lora_cfg['keys'])} module kinds, {n_train/1e6:.2f}M trainable params")
    if a.dry_run:
        print("\n".join(lora_cfg["keys"]))
        return model

    if a.grad_checkpoint:
        from mlx_lm.tuner.trainer import grad_checkpoint
        grad_checkpoint(model.layers[0])

    steps_per_epoch = len(data) // a.accum
    total = a.max_steps or steps_per_epoch * a.epochs
    warm = max(1, int(a.warmup * total))
    sched = optim.join_schedules(
        [optim.linear_schedule(a.lr * 0.01, a.lr, warm),
         optim.cosine_decay(a.lr, max(1, total - warm), a.lr * 0.01)], [warm])
    opt = optim.AdamW(learning_rate=sched, weight_decay=0.0)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    state = {"step": 0, "micro": 0, "seed": a.seed, "rank": a.rank, "total": total}
    if (out / "LATEST").exists():
        d = out / (out / "LATEST").read_text().strip()
        model.load_weights(str(d / "adapters.safetensors"), strict=False)
        opt.init(model.trainable_parameters())
        opt.state = tree_unflatten(list(mx.load(str(d / "optimizer.safetensors")).items()))
        state = json.loads((d / "state.json").read_text())
        if state["rank"] != a.rank:
            sys.exit(f"{d} was trained at rank {state['rank']}, not {a.rank}")
        print(f"resumed from {d.name}: step {state['step']}/{total}")
    mx.eval(model.parameters(), opt.state)

    stop = {"n": 0}

    def on_signal(sig, frm):
        stop["n"] += 1
        if stop["n"] > 1:
            print("\nsecond interrupt: exiting without saving", flush=True)
            os._exit(130)
        print("\ninterrupt: finishing this optimizer step, then saving", flush=True)

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)

    loss_grad = nn.value_and_grad(model, masked_loss)
    log = open(out / "log.jsonl", "a")
    t0, toks = time.time(), 0

    def order(epoch):
        return np.random.default_rng(state["seed"] + epoch).permutation(len(data))

    start_step = state["step"]
    while state["step"] < total:
        epoch, pos = divmod(state["micro"], steps_per_epoch * a.accum)
        perm = order(epoch)
        acc, losses = None, []
        for k in range(a.accum):
            rec = data[perm[pos + k]]
            ids = mx.array(rec["ids"][: a.max_tokens])
            mask = mx.array(rec["mask"][: a.max_tokens])
            (loss, n), g = loss_grad(model, ids, mask)
            acc = g if acc is None else tree_map2(mx.add, acc, g)
            mx.eval(acc, loss)
            losses.append(loss.item())
            toks += ids.size
        acc = tree_map1(lambda x: x / a.accum, acc)
        opt.update(model, acc)
        mx.eval(model.trainable_parameters(), opt.state)
        state["step"] += 1
        state["micro"] += a.accum

        dt = time.time() - t0
        rec = {"step": state["step"], "loss": float(np.mean(losses)),
               "lr": float(sched(opt.state["step"] - 1)), "tok_s": round(toks / dt, 1)}
        if val and a.val_every and state["step"] % a.val_every == 0:
            rec["val_loss"] = evaluate(model, val, a.max_tokens)
        eta = (total - state["step"]) * dt / max(1, state["step"] - start_step)
        print(f"step {state['step']}/{total} loss {rec['loss']:.4f} "
              f"{rec['tok_s']} tok/s eta {eta/3600:.1f} h"
              + (f" val {rec['val_loss']:.4f}" if "val_loss" in rec else ""), flush=True)
        log.write(json.dumps(rec) + "\n")
        log.flush()

        if state["step"] % a.save_every == 0 or stop["n"] or state["step"] == total:
            save_ckpt(out, state["step"], model, opt, state)
        if stop["n"]:
            print(f"saved at step {state['step']}; rerun the same command to resume")
            return model

    write_adapter(out, model, lora_cfg)
    print(f"done: {out}/adapters.safetensors")
    return model


def evaluate(model, val, max_tokens):
    tot = n = 0.0
    for rec in val:
        loss, k = masked_loss(model, mx.array(rec["ids"][:max_tokens]),
                              mx.array(rec["mask"][:max_tokens]))
        mx.eval(loss, k)
        tot += loss.item() * k.item()
        n += k.item()
    return tot / max(n, 1)


def tree_map1(f, t):
    from mlx.utils import tree_map
    return tree_map(f, t)


def tree_map2(f, a, b):
    from mlx.utils import tree_map
    return tree_map(f, a, b)


# ----------------------------------------------------------------------------- selftest

def tiny_moe(seed):
    """A 4-layer Qwen3-MoE with experts quantized the way the MLX Inkling conversion
    stores them (nvfp4), so LoRASwitchLinear runs over a quantized expert bank."""
    from mlx_lm.models import qwen3_moe
    mx.random.seed(seed)
    args = qwen3_moe.ModelArgs(
        model_type="qwen3_moe", hidden_size=64, num_hidden_layers=4, intermediate_size=128,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16, rms_norm_eps=1e-6,
        vocab_size=256, num_experts=8, num_experts_per_tok=2, decoder_sparse_step=1,
        mlp_only_layers=[], moe_intermediate_size=64, rope_theta=1e6, tie_word_embeddings=False,
        max_position_embeddings=4096, norm_topk_prob=True)
    m = qwen3_moe.Model(args)
    mode = "nvfp4"
    try:
        nn.quantize(m, group_size=16, bits=4, mode=mode,
                    class_predicate=lambda p, l: "switch_mlp" in p and hasattr(l, "to_quantized"))
    except Exception as e:  # older MLX without nvfp4: fall back to affine
        mode = f"affine (nvfp4 unavailable: {e})"
        nn.quantize(m, group_size=32, bits=4,
                    class_predicate=lambda p, l: "switch_mlp" in p and hasattr(l, "to_quantized"))
    mx.eval(m.parameters())
    return m, mode


def cmd_selftest(a):
    import subprocess
    import tempfile
    work = Path(tempfile.mkdtemp(prefix="inkling-lora-selftest-"))
    rng = np.random.default_rng(0)
    with open(work / "train.jsonl", "w") as f:
        for _ in range(48):
            n = int(rng.integers(40, 120))
            ids = rng.integers(0, 256, n).tolist()
            mask = [0] * (n // 2) + [1] * (n - n // 2)
            f.write(json.dumps({"ids": ids, "mask": mask}) + "\n")
    base = [sys.executable, __file__, "_selftest_train", "--data", str(work / "train.jsonl"),
            "--rank", "2", "--accum", "2", "--max-steps", "12", "--save-every", "3", "--lr", "1e-3"]

    subprocess.run(base + ["--out", str(work / "straight")], check=True)
    # Interrupted run: SIGINT lands mid-training, then the same command resumes.
    p = subprocess.Popen(base + ["--out", str(work / "broken"), "--_sleep", "0.4"])
    while not (work / "broken" / "LATEST").exists():
        time.sleep(0.05)
    time.sleep(0.5)
    p.send_signal(signal.SIGINT)
    p.wait()
    first = json.loads((work / "broken" / (work / "broken" / "LATEST").read_text().strip()
                        / "state.json").read_text())["step"]
    subprocess.run(base + ["--out", str(work / "broken")], check=True)

    A = mx.load(str(work / "straight" / "adapters.safetensors"))
    B = mx.load(str(work / "broken" / "adapters.safetensors"))
    diff = max(mx.abs(A[k] - B[k]).max().item() for k in A)
    moved = max(mx.abs(A[k]).max().item() for k in A if "lora_b" in k)
    ok = set(A) == set(B) and diff == 0.0 and moved > 0 and 0 < first < 12
    print(json.dumps({"interrupted_at_step": first, "max_abs_diff": diff,
                      "lora_b_moved": moved, "tensors": len(A), "pass": ok, "dir": str(work)}))
    sys.exit(0 if ok else 1)


def cmd_selftest_train(a):
    model, mode = tiny_moe(1234)
    print(f"tiny MoE, experts quantized as {mode}")
    if a._sleep:  # slow the run so the interrupt lands mid-training
        orig = optim.AdamW.update
        def slow(self, *x, **k):
            time.sleep(a._sleep)
            return orig(self, *x, **k)
        optim.AdamW.update = slow
    train(a, model)


# ----------------------------------------------------------------------------- cli

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prep")
    p.add_argument("--model", default="mlx-community/Inkling-2-Mini-4bit")
    p.add_argument("--configs", default="loogle,longreason,oolong_real")
    p.add_argument("--n-per-config", type=int, default=170)
    p.add_argument("--n-val", type=int, default=10)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="data")

    def train_args(p):
        p.add_argument("--data", default="data/train.jsonl")
        p.add_argument("--val", default="data/val.jsonl")
        p.add_argument("--out", default="runs/r1")
        p.add_argument("--rank", type=int, default=1)
        p.add_argument("--alpha", type=float, default=32.0, help="scale = alpha/rank")
        p.add_argument("--dropout", type=float, default=0.0)
        p.add_argument("--lr", type=float, default=5e-5)
        p.add_argument("--warmup", type=float, default=0.05)
        p.add_argument("--accum", type=int, default=8)
        p.add_argument("--epochs", type=int, default=1)
        p.add_argument("--max-steps", type=int, default=0)
        p.add_argument("--max-tokens", type=int, default=8192)
        p.add_argument("--save-every", type=int, default=5)
        p.add_argument("--val-every", type=int, default=10)
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--exclude", default=DEFAULT_EXCLUDE)
        p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction, default=True)
        p.add_argument("--dry-run", action="store_true", help="list adapted modules and exit")

    t = sub.add_parser("train")
    t.add_argument("--model", default="mlx-community/Inkling-2-Mini-4bit")
    train_args(t)
    sub.add_parser("selftest")
    st = sub.add_parser("_selftest_train")
    train_args(st)
    st.add_argument("--_sleep", type=float, default=0.0)

    a = ap.parse_args()
    if a.cmd == "_selftest_train":
        a.val = None
        a.grad_checkpoint = False
    {"prep": cmd_prep, "train": lambda a: train(a), "selftest": cmd_selftest,
     "_selftest_train": cmd_selftest_train}[a.cmd](a)


if __name__ == "__main__":
    main()
