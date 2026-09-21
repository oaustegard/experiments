"""Adapt ettin-decoder-32m to bidirectional MLM under one of five attention
arms, then eval on the probe corpus. See CLAUDE task spec for the arm
definitions.

Run: python3 adapt.py --arm causal
Smoke: python3 adapt.py --arm causal --smoke
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import masks
import probe

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

MODEL_NAME = "jhu-clsp/ettin-decoder-32m"
SEQ_LEN = 256
MASK_RATE = 0.15
TRAIN_SEED = 20260921

ARMS = ("causal", "global", "look8", "both", "bidir")


# ---------------------------------------------------------------------------
# arm masks

def build_arm_mask(arm: str, layer_types: list[str], sliding_window: int, dtype: torch.dtype) -> torch.Tensor:
    """(1, num_layers, seq_len, seq_len) additive float mask for `arm`, AND-ed
    with the model's own base mask (masks.base_allowed_tensor)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm: {arm!r}")

    num_layers = len(layer_types)
    seq_len = SEQ_LEN
    base = masks.base_allowed_tensor(layer_types, sliding_window=sliding_window, seq_len=seq_len)
    causal = masks.causal_tensor(seq_len=seq_len)
    full = torch.ones((seq_len, seq_len), dtype=torch.bool)

    i_idx = torch.arange(seq_len).unsqueeze(1)
    j_idx = torch.arange(seq_len).unsqueeze(0)
    look8 = j_idx <= (i_idx + 8)

    out = torch.empty((num_layers, seq_len, seq_len), dtype=torch.bool)
    for l, t in enumerate(layer_types):
        is_full_layer = t == "full_attention"
        if arm == "causal":
            layer_pos = causal
        elif arm == "global":
            layer_pos = full if is_full_layer else causal
        elif arm == "look8":
            layer_pos = look8
        elif arm == "both":
            layer_pos = full if is_full_layer else look8
        elif arm == "bidir":
            layer_pos = full
        out[l] = base[l] & layer_pos

    return masks.to_additive(out, dtype).unsqueeze(0)


# ---------------------------------------------------------------------------
# masking

def mask_batch(ids: np.ndarray, step: int):
    """15% masking of positions 1..254, seeded by (TRAIN_SEED + step). Same
    draw for every arm at a given step. Returns (masked_input_ids, labels,
    masked_bool) — labels is -100 where not masked."""
    n, seq_len = ids.shape
    candidates = np.arange(1, seq_len - 1)
    k = int(round(len(candidates) * MASK_RATE))
    rng = np.random.RandomState(TRAIN_SEED + step)
    mask_id = probe.EXPECTED_SPECIAL_IDS["mask"]

    masked_input_ids = ids.copy()
    labels = np.full((n, seq_len), -100, dtype=np.int64)
    masked_bool = np.zeros((n, seq_len), dtype=bool)

    for i in range(n):
        chosen = rng.choice(candidates, size=k, replace=False)
        chosen.sort()
        labels[i, chosen] = ids[i, chosen]
        masked_input_ids[i, chosen] = mask_id
        masked_bool[i, chosen] = True

    return masked_input_ids, labels, masked_bool


# ---------------------------------------------------------------------------
# forward helper

def masked_logits(model, x: torch.Tensor, sel: torch.Tensor) -> torch.Tensor:
    """Logits at the positions where sel is True. x: (B, L) input ids.
    sel: (B, L) bool. Returns (n_selected, vocab)."""
    h = model.model(input_ids=x).last_hidden_state
    return model.decoder(model.lm_head(h[sel]))


# ---------------------------------------------------------------------------
# eval

def evaluate(model, holder, arm_mask: torch.Tensor, ids: np.ndarray, batch_size: int = 16):
    """Runs the probe's masked-LM eval over `ids` (n, seq_len) under arm_mask.
    Returns {"ce": float, "acc": float} or None on no masked positions."""
    masked_input_ids, labels, _ = probe.build_masked_inputs(ids)
    n = masked_input_ids.shape[0]

    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            x = torch.from_numpy(masked_input_ids[start:end].astype(np.int64))
            y = torch.from_numpy(labels[start:end])
            sel = y != -100

            holder.mask = arm_mask
            logits = masked_logits(model, x, sel)
            gold = y[sel]
            if gold.numel() == 0:
                continue
            loss_sum = F.cross_entropy(logits.float(), gold, reduction="sum")
            total_loss += loss_sum.item()
            total_correct += (logits.argmax(dim=-1) == gold).sum().item()
            total_count += gold.numel()

    if total_count == 0:
        return None
    return {"ce": total_loss / total_count, "acc": total_correct / total_count}


# ---------------------------------------------------------------------------
# lr schedule

def make_lr_lambda(total_steps: int, warmup_frac: float = 0.05, final_frac: float = 0.10):
    warmup_steps = max(1, int(round(total_steps * warmup_frac)))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        if total_steps <= warmup_steps:
            return 1.0
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 1.0 - progress * (1.0 - final_frac)

    return lr_lambda


# ---------------------------------------------------------------------------
# checkpoint / results IO

def save_checkpoint(path: Path, step: int, model, optimizer, scheduler) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "step": step,
        "model_state": model.state_dict(),
        "opt_state": optimizer.state_dict(),
        "sched_state": scheduler.state_dict(),
        "rng": {
            "np": np.random.get_state(),
            "torch": torch.random.get_rng_state(),
        },
    }
    torch.save(payload, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: Path):
    if not path.exists():
        return None
    return torch.load(path, map_location="cpu", weights_only=False)


def save_json_atomic(data: dict, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def load_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", required=True, choices=ARMS)
    p.add_argument("--steps", type=int, default=1465)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--out", default=None)
    p.add_argument("--smoke", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)
    torch.set_default_dtype(torch.float32)

    arm = args.arm
    steps = args.steps
    batch_size = args.batch
    lr = args.lr
    eval_every = args.eval_every
    eval_n = None

    if args.smoke:
        steps = 20
        eval_every = 10
        batch_size = 4
        eval_n = 32
        out_rel = args.out or f"results/adapt/smoke_{arm}.json"
    else:
        out_rel = args.out or f"results/adapt/{arm}.json"

    out_path = HERE / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_path.parent / f"{out_path.stem}_ckpt.pt"  # keyed to the output name, so a smoke never seeds a full run

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    got = {
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
        "pad": tokenizer.pad_token_id,
    }
    if got != probe.EXPECTED_SPECIAL_IDS:
        raise AssertionError(f"tokenizer special ids mismatch: {got} != {probe.EXPECTED_SPECIAL_IDS}")

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, attn_implementation="sdpa")
    layer_types = list(model.config.layer_types)
    sliding_window = model.config.sliding_window
    dtype = next(model.parameters()).dtype

    arm_mask = build_arm_mask(arm, layer_types, sliding_window, dtype)
    holder, handles = masks.install_mask_hooks(model.model.layers)

    train_ids = np.load(DATA_DIR / "train_ids.npy")
    eval_corpus = probe.load_corpus_ids(256)
    if eval_n is not None:
        eval_corpus = eval_corpus[:eval_n]

    n_train_rows = train_ids.shape[0]
    max_steps_by_data = n_train_rows // batch_size
    if steps > max_steps_by_data:
        raise ValueError(
            f"--steps {steps} needs {steps * batch_size} rows but train_ids has {n_train_rows}"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.98)
    )
    lr_lambda = make_lr_lambda(steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    results = load_json(out_path)
    if results is None:
        results = {
            "arm": arm,
            "config": {
                "steps": steps,
                "batch": batch_size,
                "lr": lr,
                "eval_every": eval_every,
                "seed": TRAIN_SEED,
                "model": MODEL_NAME,
            },
            "evals": [],
        }

    start_step = 0
    skip_step0_eval = False
    ckpt = load_checkpoint(ckpt_path)
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["opt_state"])
        scheduler.load_state_dict(ckpt["sched_state"])
        np.random.set_state(ckpt["rng"]["np"])
        torch.random.set_rng_state(ckpt["rng"]["torch"])
        start_step = ckpt["step"]
        skip_step0_eval = True

    t_start = time.time()
    tokens_seen = start_step * batch_size * SEQ_LEN
    losses_since_eval = []

    def run_eval(step: int) -> None:
        holder.mask = arm_mask
        ev = evaluate(model, holder, arm_mask, eval_corpus, batch_size=batch_size)
        eval_ce = ev["ce"] if ev is not None else None
        eval_acc = ev["acc"] if ev is not None else None
        train_loss_avg = (
            float(np.mean(losses_since_eval)) if losses_since_eval else None
        )
        elapsed = time.time() - t_start
        record = {
            "step": step,
            "tokens_seen": tokens_seen,
            "train_loss_avg": train_loss_avg,
            "eval_ce": eval_ce,
            "eval_acc": eval_acc,
            "seconds_elapsed": elapsed,
        }
        results["evals"].append(record)
        save_json_atomic(results, out_path)
        save_checkpoint(ckpt_path, step, model, optimizer, scheduler)
        losses_since_eval.clear()

        eval_ce_str = f"{eval_ce:.4f}" if eval_ce is not None else "None"
        eval_acc_str = f"{eval_acc:.3f}" if eval_acc is not None else "None"
        train_str = f"{train_loss_avg:.4f}" if train_loss_avg is not None else "None"
        print(
            f"arm={arm} step={step} tokens={tokens_seen} train={train_str} "
            f"eval_ce={eval_ce_str} eval_acc={eval_acc_str} s={elapsed:.1f}",
            flush=True,
        )
        model.train()

    if not skip_step0_eval:
        model.eval()
        run_eval(0)

    model.train()
    for step in range(start_step, steps):
        row_start = step * batch_size
        row_end = row_start + batch_size
        batch_ids = train_ids[row_start:row_end]

        masked_ids, labels, _ = mask_batch(batch_ids, step)
        x = torch.from_numpy(masked_ids.astype(np.int64))
        y = torch.from_numpy(labels)
        sel = y != -100
        gold = y[sel]

        holder.mask = arm_mask
        optimizer.zero_grad()
        logits = masked_logits(model, x, sel)
        loss = F.cross_entropy(logits.float(), gold)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        tokens_seen += batch_size * SEQ_LEN
        losses_since_eval.append(loss.item())

        next_step = step + 1
        if (next_step) % 50 == 0:
            print(f"arm={arm} step={next_step} loss={loss.item():.4f}", flush=True)

        is_last = next_step == steps
        if next_step % eval_every == 0 or is_last:
            model.eval()
            run_eval(next_step)
            model.train()

    for h in handles:
        h.remove()

    final = results["evals"][-1] if results["evals"] else None
    results["final_eval_ce"] = final["eval_ce"] if final else None
    results["final_eval_acc"] = final["eval_acc"] if final else None
    results["total_seconds"] = time.time() - t_start
    results["steps"] = steps
    results["tokens"] = tokens_seen
    save_json_atomic(results, out_path)

    print(f"done -> {out_rel}")


if __name__ == "__main__":
    main()
