"""Zero-shot NER benchmark harness for GLiNER models (7 sets, entity-level exact-match micro F1).

    python3 run/bench.py --model <hf id or local dir> --out results/<name>.json
                         [--limit N] [--device cuda|cpu] [--batch-size 12] [--splits dev,test]
                         [--sets CrossNER_AI,...] [--check-library]

Data: data/bench/<set>/{test,dev,labels}.json written by prep_data.py (upstream's own conversion).

Method, per set and split: the batches are built exactly as GLiNER.evaluate builds them (same
collator, same order, batch size 12 by default, all of the set's lowercased labels in every
prompt). For each batch the library's forward runs ONCE (model.run_batch at threshold 0.1; the
uni- and bi-encoder span/token forwards do not read the threshold). The library's own decoder
(model.decode_batch, flat_ner=True, multi_label=False, i.e. upstream's greedy rule) is then run on
those same logits at 0.1 and at every reported threshold. So each threshold's predictions are what
the library would return at that threshold, decoded from the raw scores, not filtered from another
threshold's output. As a self-check the harness also filters the 0.1 decode by score > t and counts
sentences where that differs from the true decode at t (`filter_mismatch_sentences`; greedy flat
decoding makes them equal, apart from exact score ties at t in the token decoder).

Metric: micro P/R/F1 over (sample, label, start, end) sets per label, as the library's
BaseNEREvaluator computes it (duplicates collapse; gold entities that do not align with the split
words stay as unmatchable (None, None) entries). --check-library also runs model.evaluate at 0.5
on the same data and records its F1 next to ours.

Output JSON: per set, per split, per threshold {p, r, f1, tp, n_pred, n_true}; test_mean7 (the
unweighted mean over the 7 test sets) and dev_mean5 (over the 5 CrossNER dev sets) per threshold;
the dev-chosen threshold and the test mean at it.
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import torch

D = Path(__file__).resolve().parents[1]
CROSSNER = ["CrossNER_AI", "CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science"]
MIT = ["mit-movie", "mit-restaurant"]
SETS = CROSSNER + MIT
THRESHOLDS = (0.3, 0.4, 0.5, 0.6)
LOW = 0.1


def tkey(t):
    return f"{t:.1f}"


def set_prefix_space(tokenizer, on):
    """Make the HF fast tokenizer the library calls treat every pre-split word as if a space
    preceded it (ByteLevel add_prefix_space), so text words and the first word of each label get
    the G-dot form. Special tokens (<<ENT>>, <<SEP>>, [CLS], [SEP]) are matched before
    pre-tokenization and are unaffected. The flag is also stored in the tokenizer's init kwargs so
    save_pretrained writes it to tokenizer_config.json and a reload keeps it."""
    from tokenizers import pre_tokenizers
    pt = tokenizer.backend_tokenizer.pre_tokenizer
    st = json.loads(pt.__getstate__())
    if st.get("type") != "ByteLevel":
        raise ValueError(f"prefix-space needs a ByteLevel pre-tokenizer, got {st.get('type')}")
    st["add_prefix_space"] = bool(on)
    cls = getattr(pre_tokenizers, st.pop("type"))
    tokenizer.backend_tokenizer.pre_tokenizer = cls(**st)
    tokenizer.add_prefix_space = bool(on)
    tokenizer.init_kwargs["add_prefix_space"] = bool(on)


def prefix_space_state(tokenizer):
    try:
        return bool(json.loads(tokenizer.backend_tokenizer.pre_tokenizer.__getstate__()).get("add_prefix_space"))
    except Exception:
        return None


def load_model(model_id, device):
    from gliner import GLiNER
    model = GLiNER.from_pretrained(str(model_id))
    if getattr(model.config, "prefix_space", False):
        set_prefix_space(model.data_processor.transformer_tokenizer, True)
    model.to(device)
    model.eval()
    return model


def load_split(name, split, limit=None):
    base = D / "data/bench" / name
    data = json.loads((base / f"{split}.json").read_text())
    labels = json.loads((base / "labels.json").read_text())
    # JSON turns upstream's tuples into lists; the evaluator unpacks (s, e, label) either way
    data = [{"tokenized_text": r["tokenized_text"], "ner": [tuple(e) for e in r["ner"]]} for r in data]
    return (data[:limit] if limit else data), labels


def prf(trues, preds):
    """Micro P/R/F1 with BaseNEREvaluator semantics (per-label sets of (span, sample))."""
    tr, pr = set(), set()
    for i, ents in enumerate(trues):
        for s, e, lab in ents:
            tr.add((lab, s, e, i))
    for i, ents in enumerate(preds):
        for s, e, lab in ents:
            pr.add((lab, s, e, i))
    tp, n_pred, n_true = len(tr & pr), len(pr), len(tr)
    precision = tp / n_pred if n_pred else 0.0
    recall = tp / n_true if n_true else 0.0
    denom = precision + recall
    f1 = 2 * (precision * recall) / (denom if denom else 1)
    return {"p": precision, "r": recall, "f1": f1, "tp": tp, "n_pred": n_pred, "n_true": n_true}


def predict(model, data, labels, thresholds=THRESHOLDS, low=LOW, batch_size=12):
    """One forward per batch; library decode at `low` and at each threshold from the same logits."""
    collator = model.create_collator()
    loader = torch.utils.data.DataLoader(
        data, batch_size=batch_size, shuffle=False, collate_fn=lambda b: model.collate_batch(b, labels, collator))
    trues, low_preds, preds = [], [], {t: [] for t in thresholds}
    for batch in loader:
        out = model.run_batch(batch, threshold=low)
        dec = model.decode_batch(out, batch, threshold=low, flat_ner=True, multi_label=False)
        low_preds.extend([[(s.start, s.end, s.entity_type, float(s.score)) for s in ents] for ents in dec])
        for t in thresholds:
            dec = model.decode_batch(out, batch, threshold=t, flat_ner=True, multi_label=False)
            preds[t].extend([[(s.start, s.end, s.entity_type) for s in ents] for ents in dec])
        trues.extend(batch["entities"])
    return trues, low_preds, preds


def bench_split(model, name, split, limit=None, batch_size=12, thresholds=THRESHOLDS, low=LOW,
                check_library=False):
    data, labels = load_split(name, split, limit)
    t0 = time.time()
    trues, low_preds, preds = predict(model, data, labels, thresholds, low, batch_size)
    res = {"n_sent": len(data), "n_labels": len(labels), "seconds": round(time.time() - t0, 2), "by_threshold": {}}
    mism = {}
    for t in thresholds:
        res["by_threshold"][tkey(t)] = prf(trues, preds[t])
        filt = [[(s, e, lab) for s, e, lab, sc in ents if sc > t] for ents in low_preds]
        mism[tkey(t)] = sum(1 for a, b in zip(filt, preds[t]) if sorted(a) != sorted(b))
    res["filter_mismatch_sentences"] = mism
    res["low_threshold"] = {"t": low, **prf(trues, [[x[:3] for x in ents] for ents in low_preds])}
    if check_library:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            out, f1 = model.evaluate(data, flat_ner=True, threshold=0.5, batch_size=batch_size, entity_types=labels)
        res["library_evaluate_f1_at_0.5"] = float(f1)
        res["library_evaluate_str"] = out.strip()
    return res


def run_bench(model, splits=("dev", "test"), sets=None, limit=None, batch_size=12, thresholds=THRESHOLDS,
              low=LOW, check_library=False, log=print):
    """Bench a loaded GLiNER model. Returns the result dict (see module docstring)."""
    sets = list(sets or SETS)
    was_training = model.training
    model.eval()
    out = {"thresholds": [tkey(t) for t in thresholds], "low_threshold": low, "limit": limit,
           "batch_size": batch_size, "device": str(model.device),
           "decode": "one library forward per batch (run_batch at the low threshold); library decode_batch "
                     "(flat_ner=True, multi_label=False) on the same logits at each threshold",
           "sets": {}}
    t0 = time.time()
    for s in sets:
        for split in splits:
            if split == "dev" and s not in CROSSNER:
                continue    # MIT dev == test byte for byte; never used
            r = bench_split(model, s, split, limit, batch_size, thresholds, low, check_library)
            out["sets"].setdefault(s, {})[split] = r
            if log:
                f = " ".join(f"{k}:{v['f1'] * 100:5.1f}" for k, v in r["by_threshold"].items())
                lib = f" lib@0.5:{r['library_evaluate_f1_at_0.5'] * 100:.4f}" if check_library else ""
                log(f"  {s:20} {split:4} n={r['n_sent']:4} {f}{lib} ({r['seconds']}s)")
    for split, group, key in (("test", SETS, "test_mean7"), ("dev", CROSSNER, "dev_mean5")):
        if split in splits and all(s in out["sets"] and split in out["sets"][s] for s in group):
            out[key] = {tkey(t): sum(out["sets"][s][split]["by_threshold"][tkey(t)]["f1"] for s in group) / len(group)
                        for t in thresholds}
    if "dev_mean5" in out:
        best = max(out["dev_mean5"], key=lambda k: (out["dev_mean5"][k], -abs(float(k) - 0.5)))
        out["dev_best_threshold"] = best
        out["dev_best_mean5"] = out["dev_mean5"][best]
        if "test_mean7" in out:
            out["test_mean7_at_dev_best"] = out["test_mean7"][best]
    out["seconds"] = round(time.time() - t0, 2)
    if was_training:
        model.train()
    return out


def versions():
    import importlib.metadata as md
    v = {"python": sys.version.split()[0]}
    for p in ("gliner", "torch", "transformers", "tokenizers", "accelerate", "huggingface_hub", "safetensors"):
        try:
            v[p] = md.version(p)
        except md.PackageNotFoundError:
            v[p] = None
    v["cuda"] = torch.version.cuda
    v["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--limit", type=int, default=None, help="first N sentences of each split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--splits", default="dev,test")
    ap.add_argument("--sets", default=None)
    ap.add_argument("--check-library", action="store_true",
                    help="also run model.evaluate (threshold 0.5) on the same data and record its F1")
    a = ap.parse_args()
    warnings.filterwarnings("ignore", message="Sentence of length")
    torch.set_grad_enabled(False)
    t0 = time.time()
    model = load_model(a.model, a.device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded {a.model} ({type(model).__name__}, {n_params:,} params) in {time.time() - t0:.1f}s", flush=True)
    res = run_bench(model, splits=tuple(a.splits.split(",")), sets=a.sets.split(",") if a.sets else None,
                    limit=a.limit, batch_size=a.batch_size, check_library=a.check_library)
    res = {"name": a.out.stem, "model": a.model, "class": type(model).__name__, "params": n_params,
           "prefix_space": prefix_space_state(model.data_processor.transformer_tokenizer),
           "versions": versions(), "wall_seconds": round(time.time() - t0, 1), "bench": res}
    out = a.out if a.out.is_absolute() else Path.cwd() / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    b = res["bench"]
    for key in ("test_mean7", "dev_mean5"):
        if key in b:
            print(key, " ".join(f"{k}:{v * 100:.2f}" for k, v in b[key].items()))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
