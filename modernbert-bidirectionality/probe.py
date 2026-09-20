"""modernbert-bidirectionality probe. See PLAN.md for the design.

Run: python3 probe.py --model answerdotai/ModernBERT-base --tag mb
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import unicodedata
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

import masks

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"

EXPECTED_SPECIAL_IDS = {"cls": 50281, "sep": 50282, "mask": 50284, "pad": 50283}
SEQ_LEN = 256
MASK_RATE = 0.15
MASK_SEED = 20260920
ANCHOR_PS = (5, 10, 25)

PUNCT_LITERALS = {
    ".", ",", ";", ":", "(", ")", '"', "'", "!", "?",
    "@-@", "@,@", "@.@", "–", "—",
}


# ---------------------------------------------------------------------------
# arm list

def build_canonical_arm_order(num_layers: int) -> list[str]:
    """Arm names are relative to the model's layer count (ModernBERT-base 22,
    ettin-32m 10); the *8 keep-arms need at least 16 layers."""
    eight = num_layers >= 16
    arms = ["bidir", "causal", "truncate"]
    arms += [f"single:{l}" for l in range(num_layers)]
    arms += ["keep:top4"] + (["keep:top8"] if eight else [])
    arms += [f"prefix:{k}" for k in range(1, num_layers)]
    arms += [f"suffix:{k}" for k in range(1, num_layers)]
    arms += ["keep:global", "keep:local", "keep:every4", "keep:first4", "keep:last4"]
    arms += ["keep:first8", "keep:last8"] if eight else []
    arms += [f"look:{m}" for m in (1, 2, 4, 8, 16, 32, 64, 128)]
    arms += ["anchor:special", "anchor:first4", "anchor:punct"]
    arms += [f"anchor:attn:{p}" for p in ANCHOR_PS]
    arms += [f"anchor:rand:{p}" for p in ANCHOR_PS]
    arms += ["anchor:masked", "look:8+punct", "look:8+attn:10"]
    # sink-reachable variants of the layer axis: causal layers may still read [CLS]/[SEP]
    arms += ["causal+special", "keep:global+special"]
    arms += [f"single+special:{l}" for l in range(num_layers)]
    return arms


SMOKE_ARMS = [
    "bidir", "causal", "truncate", "single:0", "single+special:0", "causal+special",
    "prefix:3", "keep:global", "look:8", "anchor:punct", "anchor:first4", "anchor:attn:10",
]


def resolve_arm_list(spec: str, num_layers: int) -> list[str]:
    canonical = build_canonical_arm_order(num_layers)
    if spec == "all":
        return canonical
    if spec == "smoke":
        requested = list(SMOKE_ARMS)
    else:
        requested = [a.strip() for a in spec.split(",") if a.strip()]

    canonical_index = {a: i for i, a in enumerate(canonical)}
    for a in requested:
        if a not in canonical_index:
            raise ValueError(f"unknown arm: {a!r}")

    needs_top = any(a in ("keep:top4", "keep:top8") for a in requested)
    if needs_top:
        for l in range(num_layers):
            name = f"single:{l}"
            if name not in requested:
                requested.append(name)

    seen: set[str] = set()
    ordered = []
    for a in requested:
        if a not in seen:
            seen.add(a)
            ordered.append(a)
    ordered.sort(key=lambda a: canonical_index[a])
    return ordered


# ---------------------------------------------------------------------------
# corpus / masking / anchor sets

def load_corpus_ids(n: int) -> np.ndarray:
    path = DATA_DIR / "corpus_ids.json"
    with open(path) as f:
        data = json.load(f)
    return np.array(data[:n], dtype=np.int64)


def build_masked_inputs(corpus_ids: np.ndarray):
    """Deterministic 15% masking draw, seeded, identical for every arm/model
    and stable under truncation to a smaller --n (sequential RNG draws in
    corpus order)."""
    n, seq_len = corpus_ids.shape
    candidates = np.arange(1, seq_len - 1)
    k = int(round(len(candidates) * MASK_RATE))
    rng = np.random.RandomState(MASK_SEED)

    masked_input_ids = corpus_ids.copy()
    labels = np.full((n, seq_len), -100, dtype=np.int64)
    masked_bool = np.zeros((n, seq_len), dtype=bool)
    mask_id = EXPECTED_SPECIAL_IDS["mask"]

    for i in range(n):
        chosen = rng.choice(candidates, size=k, replace=False)
        chosen.sort()
        labels[i, chosen] = corpus_ids[i, chosen]
        masked_input_ids[i, chosen] = mask_id
        masked_bool[i, chosen] = True

    return masked_input_ids, labels, masked_bool


def is_punct_text(text: str) -> bool:
    t = text.strip()
    if not t:
        return False
    if t in PUNCT_LITERALS:
        return True
    return all(unicodedata.category(ch).startswith("P") for ch in t)


def build_punct_sets(corpus_ids: np.ndarray, tokenizer) -> list[torch.Tensor]:
    n, seq_len = corpus_ids.shape
    unique_ids = sorted(set(int(x) for x in corpus_ids.flatten()))
    punct_ids = {tid for tid in unique_ids if is_punct_text(tokenizer.decode([tid]))}

    sets = []
    for i in range(n):
        bool_arr = torch.zeros(seq_len, dtype=torch.bool)
        bool_arr[0] = True
        bool_arr[-1] = True
        row = corpus_ids[i]
        for pos in range(1, seq_len - 1):
            if int(row[pos]) in punct_ids:
                bool_arr[pos] = True
        sets.append(bool_arr)
    return sets


def build_masked_sets(masked_bool: np.ndarray) -> list[torch.Tensor]:
    return [torch.from_numpy(masked_bool[i].copy()) for i in range(masked_bool.shape[0])]


def k_for_p(p: int) -> int:
    return math.ceil(p / 100 * 254)


def build_rand_sets(n: int, seq_len: int, ps=ANCHOR_PS) -> dict[int, list[torch.Tensor]]:
    out: dict[int, list[torch.Tensor]] = {}
    for p in ps:
        k = k_for_p(p)
        rng = np.random.RandomState(MASK_SEED + p)
        sets_p = []
        for _ in range(n):
            chosen = rng.choice(seq_len, size=k, replace=False)
            bool_arr = torch.zeros(seq_len, dtype=torch.bool)
            bool_arr[chosen] = True
            sets_p.append(bool_arr)
        out[p] = sets_p
    return out


def build_attn_top_sets(received_future_mass: torch.Tensor, ps=ANCHOR_PS) -> dict[int, list[torch.Tensor]]:
    n, seq_len = received_future_mass.shape
    out: dict[int, list[torch.Tensor]] = {}
    for p in ps:
        k = k_for_p(p)
        sets_p = []
        for i in range(n):
            idx = torch.topk(received_future_mass[i], k).indices
            bool_arr = torch.zeros(seq_len, dtype=torch.bool)
            bool_arr[idx] = True
            sets_p.append(bool_arr)
        out[p] = sets_p
    return out


# ---------------------------------------------------------------------------
# descriptive pass

def run_descriptive(model, holder, corpus_ids: np.ndarray, ctx: dict, batch_size: int):
    num_layers = len(ctx["layer_types"])
    n, seq_len = corpus_ids.shape
    dtype = next(model.parameters()).dtype

    i_idx = torch.arange(seq_len).unsqueeze(1)
    j_idx = torch.arange(seq_len).unsqueeze(0)
    future_bool = (j_idx > i_idx)
    dist = (j_idx - i_idx).clamp(min=0).float()
    special_col = torch.zeros(seq_len, dtype=torch.bool)
    special_col[0] = True
    special_col[-1] = True
    punct_cols = torch.stack(ctx["punct"])  # (n, seq_len)
    k5 = max(1, round(0.05 * seq_len))

    state = {
        "future_mass_sum": torch.zeros(num_layers, dtype=torch.float64),
        "total_mass_sum": torch.zeros(num_layers, dtype=torch.float64),
        "special_future_sum": torch.zeros(num_layers, dtype=torch.float64),
        "punct_future_sum": torch.zeros(num_layers, dtype=torch.float64),
        "dist_weighted_sum": torch.zeros(num_layers, dtype=torch.float64),
        "top5_share_sum": torch.zeros(num_layers, dtype=torch.float64),
        "received_future_mass": torch.zeros(n, seq_len, dtype=torch.float64),
        "current_slice": slice(0, 0),
        "future_bool": future_bool,
        "dist": dist,
        "special_col": special_col,
        "punct_cols": punct_cols,
        "k5": k5,
    }

    def make_attn_hook(layer_idx):
        def hook(module, inp, output):
            w = output[1]  # (B, H, L, L)
            w_sum_heads = w.sum(dim=1)  # (B, L, L) -- reduce over heads immediately
            fb = state["future_bool"]
            total_per_row = w_sum_heads.sum(dim=-1)
            future_per_row = (w_sum_heads * fb).sum(dim=-1)
            state["future_mass_sum"][layer_idx] += future_per_row.sum().double()
            state["total_mass_sum"][layer_idx] += total_per_row.sum().double()

            spec_future = (w_sum_heads * fb * state["special_col"]).sum(dim=-1)
            state["special_future_sum"][layer_idx] += spec_future.sum().double()

            b_slice = state["current_slice"]
            punct_cols_batch = state["punct_cols"][b_slice].unsqueeze(1)  # (B,1,L)
            punct_future = (w_sum_heads * fb * punct_cols_batch).sum(dim=-1)
            state["punct_future_sum"][layer_idx] += punct_future.sum().double()

            dist_weighted = (w_sum_heads * fb * state["dist"]).sum(dim=-1)
            state["dist_weighted_sum"][layer_idx] += dist_weighted.sum().double()

            received_contrib = (w_sum_heads * fb).sum(dim=1)  # (B, L) received per key position
            state["received_future_mass"][b_slice] += received_contrib.double()

            total_future_per_seq = future_per_row.sum(dim=-1)
            topvals, _ = received_contrib.topk(state["k5"], dim=-1)
            share = topvals.sum(dim=-1) / total_future_per_seq.clamp(min=1e-12)
            state["top5_share_sum"][layer_idx] += share.sum().double()
        return hook

    attn_handles = [
        layer.attn.register_forward_hook(make_attn_hook(l))
        for l, layer in enumerate(model.model.layers)
    ]

    bidir_allowed = masks.allowed("bidir", None, ctx)
    additive = masks.to_additive(bidir_allowed, dtype).unsqueeze(0)

    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                state["current_slice"] = slice(start, end)
                holder.mask = additive
                batch_ids = torch.from_numpy(corpus_ids[start:end])
                model(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids))
    finally:
        for h in attn_handles:
            h.remove()

    future_share = (state["future_mass_sum"] / state["total_mass_sum"]).tolist()
    special_share = (state["special_future_sum"] / state["future_mass_sum"]).tolist()
    punct_share = (state["punct_future_sum"] / state["future_mass_sum"]).tolist()
    mean_future_distance = (state["dist_weighted_sum"] / state["future_mass_sum"]).tolist()
    top5_share = (state["top5_share_sum"] / n).tolist()

    return {
        "future_share": future_share,
        "top5_share": top5_share,
        "special_share": special_share,
        "punct_share": punct_share,
        "mean_future_distance": mean_future_distance,
        "received_future_mass": state["received_future_mass"].tolist(),
    }


# ---------------------------------------------------------------------------
# arm evaluation

def run_truncate(model, holder, masked_input_ids: np.ndarray, labels: np.ndarray,
                 sep_id: int, pad_id: int, batch_size: int = 16) -> dict:
    """In-distribution left-context-only reference: for every masked position i the
    model sees tokens 0..i (other masks to the left kept as masked) followed by [SEP],
    under its own bidirectional attention. Unlike `causal`, nothing here is off the
    training distribution; it is what a bidirectional model can do with no right
    context at all."""
    holder.mask = None  # hooks pass through; the stock 2D padding mask applies
    items = []  # (seq, i, length)
    n, seq_len = masked_input_ids.shape
    for s_ in range(n):
        for i in np.nonzero(labels[s_] != -100)[0]:
            items.append((s_, int(i), int(i) + 2))
    items.sort(key=lambda t: t[2])
    total_loss = 0.0; total_correct = 0; total_count = 0; n_forward = 0
    t0 = time.time()
    with torch.no_grad():
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            L = max(t[2] for t in chunk)
            ids = torch.full((len(chunk), L), pad_id, dtype=torch.long)
            am = torch.zeros((len(chunk), L), dtype=torch.long)
            for b, (s_, i, ln) in enumerate(chunk):
                ids[b, : i + 1] = torch.from_numpy(masked_input_ids[s_, : i + 1].astype(np.int64))
                ids[b, i + 1] = sep_id
                am[b, :ln] = 1
            logits = model(input_ids=ids, attention_mask=am).logits
            n_forward += 1
            rows = torch.arange(len(chunk))
            pos = torch.tensor([t[1] for t in chunk])
            gold = torch.tensor([int(labels[s_, i]) for s_, i, _ in chunk])
            fl = logits[rows, pos].float()
            total_loss += torch.nn.functional.cross_entropy(fl, gold, reduction="sum").item()
            total_correct += (fl.argmax(-1) == gold).sum().item()
            total_count += len(chunk)
    ce = total_loss / total_count if total_count else None
    acc = total_correct / total_count if total_count else None
    return {"ce": ce, "acc": acc, "n_masked": total_count,
            "seconds": time.time() - t0, "n_forward": n_forward}


def run_arm(model, holder, arm: str, ctx: dict, masked_input_ids: np.ndarray,
            labels: np.ndarray, batch_size: int) -> dict:
    if arm == "truncate":
        return run_truncate(model, holder, masked_input_ids, labels,
                            EXPECTED_SPECIAL_IDS["sep"], EXPECTED_SPECIAL_IDS["pad"])
    dtype = next(model.parameters()).dtype
    n, seq_len = masked_input_ids.shape
    seq_dependent = masks.is_seq_dependent_arm(arm)

    if not seq_dependent:
        allowed_bool = masks.allowed(arm, None, ctx)
        additive = masks.to_additive(allowed_bool, dtype).unsqueeze(0)

    total_loss = 0.0
    total_correct = 0
    total_count = 0
    n_forward = 0
    t0 = time.time()

    with torch.no_grad():
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            batch_ids = torch.from_numpy(masked_input_ids[start:end])
            batch_labels = torch.from_numpy(labels[start:end])

            if seq_dependent:
                stacked = torch.stack(
                    [masks.allowed(arm, i, ctx) for i in range(start, end)], dim=0
                )
                holder.mask = masks.to_additive(stacked, dtype)
            else:
                holder.mask = additive

            outputs = model(input_ids=batch_ids, attention_mask=torch.ones_like(batch_ids))
            n_forward += 1
            logits = outputs.logits  # (B, L, vocab)

            mask_bool = batch_labels != -100
            flat_logits = logits[mask_bool].float()
            flat_labels = batch_labels[mask_bool]
            if flat_labels.numel() > 0:
                loss_sum = torch.nn.functional.cross_entropy(
                    flat_logits, flat_labels, reduction="sum"
                )
                total_loss += loss_sum.item()
                preds = flat_logits.argmax(dim=-1)
                total_correct += (preds == flat_labels).sum().item()
                total_count += flat_labels.numel()

    elapsed = time.time() - t0
    ce = total_loss / total_count if total_count else None
    acc = total_correct / total_count if total_count else None
    return {
        "ce": ce, "acc": acc, "n_masked": total_count,
        "seconds": elapsed, "n_forward": n_forward,
    }


def compute_retained(ce_arm, ce_bidir, ce_causal):
    denom = ce_causal - ce_bidir
    if denom == 0:
        return None
    return (ce_causal - ce_arm) / denom


def backfill_retained(results: dict) -> None:
    """retained: benefit kept relative to the all-causal model (denominator includes
    the off-distribution collapse). retained_trunc: relative to the in-distribution
    left-only reference `truncate` (negative = worse than having no right context)."""
    arms = results["arms"]
    if "bidir" not in arms:
        return
    ce_bidir = arms["bidir"]["ce"]
    ce_causal = arms["causal"]["ce"] if "causal" in arms else None
    ce_trunc = arms["truncate"]["ce"] if "truncate" in arms else None
    for r in arms.values():
        if r.get("ce") is None:
            continue
        if r.get("delta_ce") is None:
            r["delta_ce"] = r["ce"] - ce_bidir
        if r.get("retained") is None and ce_causal is not None:
            r["retained"] = compute_retained(r["ce"], ce_bidir, ce_causal)
        if r.get("retained_trunc") is None and ce_trunc is not None:
            r["retained_trunc"] = compute_retained(r["ce"], ce_bidir, ce_trunc)


def ensure_top_layers(ctx: dict, results: dict, num_layers: int) -> None:
    if ctx.get("top_layers"):
        return
    deltas = []
    for l in range(num_layers):
        name = f"single:{l}"
        if name not in results["arms"]:
            raise RuntimeError(f"keep:top* requires {name} to be computed first")
        deltas.append((results["arms"][name]["ce"] - results["arms"]["bidir"]["ce"], l))
    deltas.sort(key=lambda x: -x[0])
    ctx["top_layers"] = {
        "top4": [l for _, l in deltas[:4]],
        "top8": [l for _, l in deltas[:8]],
    }
    results.setdefault("meta", {})["top_layers"] = ctx["top_layers"]


# ---------------------------------------------------------------------------
# results IO

def load_results(out_path: Path) -> dict:
    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)
    return {"meta": None, "descriptive": None, "arms": {}}


def save_results(results: dict, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(results, f)
    os.replace(tmp, out_path)


def log_arm(name: str, r: dict) -> None:
    retained = r.get("retained")
    retained_str = f"{retained:.3f}" if retained is not None else "None"
    ce = r.get("ce")
    acc = r.get("acc")
    ce_str = f"{ce:.4f}" if ce is not None else "None"
    acc_str = f"{acc:.3f}" if acc is not None else "None"
    rt = r.get("retained_trunc")
    rt_str = f"{rt:.3f}" if rt is not None else "None"
    print(f"arm={name} ce={ce_str} acc={acc_str} retained={retained_str} retained_trunc={rt_str} s={r['seconds']:.1f}", flush=True)


# ---------------------------------------------------------------------------

def assert_special_ids(tokenizer) -> None:
    got = {
        "cls": tokenizer.cls_token_id,
        "sep": tokenizer.sep_token_id,
        "mask": tokenizer.mask_token_id,
        "pad": tokenizer.pad_token_id,
    }
    if got != EXPECTED_SPECIAL_IDS:
        raise AssertionError(f"tokenizer special ids mismatch: {got} != {EXPECTED_SPECIAL_IDS}")


def build_ctx(layer_types, sliding_window, corpus_ids, tokenizer, masked_bool) -> dict:
    n, seq_len = corpus_ids.shape
    ctx = {
        "layer_types": layer_types,
        "seq_len": seq_len,
        "base": masks.base_allowed_tensor(layer_types, sliding_window=sliding_window, seq_len=seq_len),
        "causal": masks.causal_tensor(seq_len=seq_len),
        "punct": build_punct_sets(corpus_ids, tokenizer),
        "masked": build_masked_sets(masked_bool),
        "rand": build_rand_sets(n, seq_len),
        "attn_top": {},
        "top_layers": {},
    }
    return ctx


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--n", type=int, default=None)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--arms", default="all")
    p.add_argument("--out", default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.set_num_threads(4)

    n = args.n if args.n is not None else (16 if args.arms == "smoke" else 256)
    out_rel = args.out or f"results/{args.tag}.json"
    out_path = HERE / out_rel
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    assert_special_ids(tokenizer)

    model = AutoModelForMaskedLM.from_pretrained(args.model, attn_implementation="eager")
    model.eval()

    num_layers = model.config.num_hidden_layers
    layer_types = list(model.config.layer_types)
    sliding_window = model.config.sliding_window

    corpus_ids = load_corpus_ids(n)
    n = corpus_ids.shape[0]
    masked_input_ids, labels, masked_bool = build_masked_inputs(corpus_ids)

    results = load_results(out_path)
    if results["meta"] is not None:
        if results["meta"]["n"] != n:
            raise RuntimeError(
                f"resume n mismatch: results has n={results['meta']['n']}, this run has n={n}"
            )
        if results["meta"]["model"] != args.model:
            raise RuntimeError(
                f"resume model mismatch: results has {results['meta']['model']}, this run has {args.model}"
            )
    else:
        results["meta"] = {
            "model": args.model,
            "tag": args.tag,
            "n": n,
            "batch": args.batch,
            "seed": MASK_SEED,
            "mask_rate": MASK_RATE,
            "num_layers": num_layers,
            "layer_types": layer_types,
            "sliding_window": sliding_window,
            "special_ids": EXPECTED_SPECIAL_IDS,
        }
        save_results(results, out_path)

    ctx = build_ctx(layer_types, sliding_window, corpus_ids, tokenizer, masked_bool)

    if results.get("descriptive") is not None:
        received = torch.tensor(results["descriptive"]["received_future_mass"], dtype=torch.float64)
        ctx["attn_top"] = build_attn_top_sets(received, ANCHOR_PS)
    else:
        holder, handles = masks.install_mask_hooks(model.model.layers)
        try:
            descriptive = run_descriptive(model, holder, corpus_ids, ctx, args.batch)
        finally:
            for h in handles:
                h.remove()
        results["descriptive"] = descriptive
        save_results(results, out_path)
        received = torch.tensor(descriptive["received_future_mass"], dtype=torch.float64)
        ctx["attn_top"] = build_attn_top_sets(received, ANCHOR_PS)
        print("descriptive done", flush=True)

    if results["meta"].get("top_layers"):
        ctx["top_layers"] = results["meta"]["top_layers"]

    holder, handles = masks.install_mask_hooks(model.model.layers)
    try:
        arm_list = resolve_arm_list(args.arms, num_layers)
        for arm in arm_list:
            if arm not in results["arms"]:
                if arm in ("keep:top4", "keep:top8"):
                    ensure_top_layers(ctx, results, num_layers)
                r = run_arm(model, holder, arm, ctx, masked_input_ids, labels, args.batch)
                r["delta_ce"] = None
                r["retained"] = None
                r["retained_trunc"] = None
                results["arms"][arm] = r
                backfill_retained(results)
                save_results(results, out_path)
                log_arm(arm, results["arms"][arm])
            else:
                backfill_retained(results)
                save_results(results, out_path)
    finally:
        for h in handles:
            h.remove()

    print(f"done -> {out_rel}")


if __name__ == "__main__":
    main()
