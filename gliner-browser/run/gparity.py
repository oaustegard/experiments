"""Parity of the packed model (GRef) against the gliner library, per checkpoint.

    python3 run/gparity.py CKPT [--n 40] [--out results/pack_parity.json] [--name NAME] [--shift auto|none]

CKPT is a gliner directory or a cached hub id (bi-encoders included: the label encoder is packed with the
same variant as the text encoder, and (a) also compares label-tokenizer ids and pooled label vectors).

Packs CKPT_DIR as fp32, t4-w8L63, t4-w8L31 and t4-w8L63 with int8 heads (web/models/<name>-<variant>),
then on N+ inputs (bench sentences of different sets and lengths, a text longer than max_len words,
non-ASCII and punctuation, multi-word / duplicate / empty / cased labels, more labels than max_types):
--shift none skips the shifted-bias decode (a trained checkpoint needs none) and measures (c) at its own bias.

 (a) fp32 pack vs the library's own logits (single-text batches): token ids and words_mask equal, max abs
     logit difference over valid entries; naive vs factorised span head on the same input.
 (b) GRef.predict_entities vs model.predict_entities at 0.5 and 0.3 (start, end, label equal, score to
     1e-4), at the checkpoint's own logit bias and with an extra bias chosen so the fp32 decode at 0.5 yields about
     1.5 entities per input (the smoke checkpoints are barely trained and predict nothing otherwise;
     the bias is set on the library's config and on GRef identically).
 (c) each quantized pack vs the fp32 pack: max / mean abs logit difference, and entity changes at the
     shifted bias.

Results merge into the output JSON under the checkpoint's name.
"""
import argparse
import json
import os
import random
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gpack  # noqa: E402
import gref  # noqa: E402

D = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(D / "data" / "hf"))
SETS = ["CrossNER_AI", "CrossNER_literature", "CrossNER_music", "CrossNER_politics", "CrossNER_science",
        "mit-movie", "mit-restaurant"]
VARIANTS = [("fp32", "fp32"), ("t4-w8L63", "fp32"), ("t4-w8L31", "fp32"), ("t4-w8L63", "int8")]


def build_inputs(n, max_len, max_types):
    rng = random.Random(0)
    data = {s: json.loads((D / "data/bench" / s / "test.json").read_text()) for s in SETS}
    labs = {s: json.loads((D / "data/bench" / s / "labels.json").read_text()) for s in SETS}
    union = sorted({lab for s in SETS for lab in labs[s]})
    out = []
    for text, labels in gpack.FIXTURE_INPUTS:
        out.append({"kind": "handmade", "text": text, "labels": labels})
    # long text: one set's sentences joined past max_len words
    words, i = [], 0
    while len(words) <= max_len + 60:
        words += data["CrossNER_politics"][i]["tokenized_text"]
        i += 1
    out.append({"kind": f"long ({len(words)} words > max_len {max_len})", "text": " ".join(words),
                "labels": labs["CrossNER_politics"]})
    # more labels than max_types, with multi-word, duplicate, cased and empty labels
    many = rng.sample(union, min(len(union), max_types + 6)) + ["Person", "person", ""]
    r = data["CrossNER_science"][14]
    out.append({"kind": f"{len(many)} labels > max_types {max_types}", "text": " ".join(r["tokenized_text"]),
                "labels": many})
    k = 0
    while len(out) < n:
        s = SETS[k % len(SETS)]
        rows = data[s]
        if k % 3 == 0 and s.startswith("CrossNER"):    # prefer a non-ASCII sentence every third pick
            pool = [r for r in rows if any(ord(c) > 127 for c in " ".join(r["tokenized_text"]))]
        else:
            pool = rows
        r = rng.choice(pool)
        if k % 4 == 1:
            labels = rng.sample(union, rng.randint(1, 6))
        else:
            labels = labs[s]
        out.append({"kind": f"bench {s}", "text": " ".join(r["tokenized_text"]), "labels": labels})
        k += 1
    return out


def lib_forward(model, text, labels):
    pb = model.prepare_batch(text, labels)
    b = model.collate_batch(pb["input_x"], pb["entity_types"], label_names=pb["label_names"])
    out = model.run_batch(b)
    extra = {}
    if "labels_input_ids" in b:      # bi-encoder: label-tokenizer ids (unpadded) and pooled label vectors
        li, lm = b["labels_input_ids"], b["labels_attention_mask"]
        extra["label_ids"] = [r[: int(k.sum())].tolist() for r, k in zip(li, lm)]
        extra["label_vecs"] = model.model.token_rep_layer.encode_labels(li, lm).numpy()
    return out[0][0].numpy(), b["input_ids"][0].tolist(), b["words_mask"][0].tolist(), b["tokens"][0], extra


def valid_mask(z, span_mode):
    return np.isfinite(z) if span_mode == "markerV0" else np.ones(z.shape, bool)


def ent_key(e):
    return (e["start"], e["end"], e["label"])


def compare_ents(a, b, tol=1e-4):
    """a: library, b: GRef. Identical lists in order, scores within tol."""
    if [ent_key(e) for e in a] != [ent_key(e) for e in b]:
        return False, None
    d = max([abs(x["score"] - y["score"]) for x, y in zip(a, b)], default=0.0)
    return d <= tol, d


def change_stats(ref, other):
    A, B = {ent_key(e) for e in ref}, {ent_key(e) for e in other}
    return len(A), len(B), len(A & B)


def pick_shift(g, zs, per_input=1.5):
    """Extra logit bias at which the fp32 decode at 0.5 yields about `per_input` entities per input
    (bisection on the cached logits; the count is close to monotone in the bias)."""
    b0 = g.b

    def count(x):
        g.b = b0 + x
        return sum(len(g.decode(z, labels, 0.5)) for _, _, labels, z, _ in zs)
    lo, hi, target = -30.0, 60.0, per_input * len(zs)
    for _ in range(40):
        mid = (lo + hi) / 2
        if count(mid) < target:
            lo = mid
        else:
            hi = mid
    g.b = b0
    return hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--out", type=Path, default=D / "results" / "pack_parity.json")
    ap.add_argument("--name", default=None)
    ap.add_argument("--shift", default="auto", choices=["auto", "none"])
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    import torch
    import bench
    torch.set_grad_enabled(False)
    ck = gpack.resolve_ckpt(a.ckpt)
    cfg, sd = gpack.load_checkpoint(ck)
    name = a.name or gpack.ckpt_name(ck, cfg)
    dirs = {}
    for var, head in VARIANTS:
        key = var + ("+h8" if head == "int8" else "")
        out = D / "web" / "models" / f"{name}-{var}{'-h8' if head == 'int8' else ''}"
        m, data = gpack.pack(cfg, sd, var, head)
        out.mkdir(parents=True, exist_ok=True)
        gpack.write_tokenizer(ck, out, m["gliner"])
        m["gliner"]["source"] = gpack.source_info(a.ckpt, ck, name)
        (out / "weights.bin").write_bytes(data)
        (out / "manifest.json").write_text(json.dumps(m))
        dirs[key] = out
    model = bench.load_model(ck, "cpu")
    assert model.decoder.config is model.config, "decoder reads another config object"
    refs = {k: gref.load(v) for k, v in dirs.items()}
    g = refs["fp32"]
    mode = g.g["span_mode"]
    inputs = build_inputs(a.n, g.g["max_len"], g.g["max_types_train"] or 25)

    # ---- (a) logits
    rows, zs = [], []
    t_lib = t_ref = 0.0
    for inp in inputs:
        t0 = time.perf_counter()
        zt, ids_l, wm_l, toks, extra = lib_forward(model, inp["text"], inp["labels"])
        t1 = time.perf_counter()
        labels = g.norm_labels(inp["labels"])
        words = [w for w, _, _ in g.split(inp["text"])][: g.g["max_len"]]
        z, b = g.logits(words, labels)
        t2 = time.perf_counter()
        t_lib += t1 - t0
        t_ref += t2 - t1
        vm = valid_mask(z, mode)
        row = {"kind": inp["kind"], "n_words": len(words), "n_labels": len(labels), "n_tokens": len(b["ids"]),
               "ids_equal": b["ids"] == ids_l, "words_mask_equal": b["words_mask"] == wm_l,
               "words_equal": list(toks) == words, "shape_equal": list(zt.shape) == list(z.shape)}
        row["max_abs"] = float(np.abs(z[vm] - zt[vm]).max()) if row["shape_equal"] else None
        if g.bi:
            row["label_ids_equal"] = extra["label_ids"] == b["label_ids"]
            row["label_vec_max_abs"] = float(np.abs(extra["label_vecs"] - g.label_vecs(labels)).max())
        row["max_abs_logit"] = float(np.abs(zt[vm]).max())
        if mode == "markerV0":
            zn, _ = g.logits(words, labels, factorised=False)
            row["naive_vs_factorised"] = float(np.abs(zn[vm] - z[vm]).max())
            row["naive_vs_library"] = float(np.abs(zn[vm] - zt[vm]).max())
        rows.append(row)
        zs.append((inp, words, labels, z, vm))
    shift = pick_shift(g, zs) if a.shift == "auto" else 0.0
    a_sum = {"n_inputs": len(rows), "all_ids_equal": all(r["ids_equal"] for r in rows),
             "all_words_mask_equal": all(r["words_mask_equal"] for r in rows),
             "all_shapes_equal": all(r["shape_equal"] for r in rows),
             "max_abs_logit_diff": max(r["max_abs"] for r in rows),
             "median_of_per_input_max": float(np.median([r["max_abs"] for r in rows])),
             "largest_abs_logit": max(r["max_abs_logit"] for r in rows),
             "library_seconds": round(t_lib, 2), "gref_seconds": round(t_ref, 2)}
    if g.bi:
        a_sum["all_label_ids_equal"] = all(r["label_ids_equal"] for r in rows)
        a_sum["label_vec_max_abs_diff"] = max(r["label_vec_max_abs"] for r in rows)
    if mode == "markerV0":
        a_sum["naive_vs_factorised_max"] = max(r["naive_vs_factorised"] for r in rows)
        a_sum["naive_vs_library_max"] = max(r["naive_vs_library"] for r in rows)
    print(f"[{name}] (a) {json.dumps(a_sum)}")

    # ---- (b) decoded entities vs model.predict_entities
    b_sum = {"extra_logit_bias": shift}
    lib_ents = {}
    base_bias = model.config.logit_bias
    for bias_name, extra in (("own_bias", 0.0), ("shifted_bias", shift))[: 2 if a.shift == "auto" else 1]:
        model.config.logit_bias = (base_bias or 0.0) + extra
        g.b = (base_bias or 0.0) + extra
        for t in (0.5, 0.3):
            ok = n_ent = 0
            maxd = 0.0
            fails = []
            for i, inp in enumerate(inputs):
                le = model.predict_entities(inp["text"], inp["labels"], threshold=t)
                ge = g.predict_entities(inp["text"], inp["labels"], threshold=t)
                lib_ents[(bias_name, t, i)] = le
                same, d = compare_ents(le, ge)
                n_ent += len(le)
                if same:
                    ok += 1
                    maxd = max(maxd, d)
                else:
                    fails.append({"i": i, "kind": inp["kind"], "lib": le[:5], "gref": ge[:5]})
            b_sum[f"{bias_name}@{t}"] = {"inputs_identical": ok, "inputs": len(inputs), "entities": n_ent,
                                         "max_score_diff": maxd, "failures": fails[:3]}
            print(f"[{name}] (b) {bias_name} t={t}: {ok}/{len(inputs)} identical, {n_ent} entities, "
                  f"max score diff {maxd:.2e}")
    model.config.logit_bias = base_bias
    g.b = float(base_bias or 0.0)

    # ---- (c) quantized vs fp32
    c_sum = {}
    for key, r in refs.items():
        if key == "fp32":
            continue
        mx, sabs, cnt = 0.0, 0.0, 0
        st = {0.5: [0, 0, 0], 0.3: [0, 0, 0]}
        for inp, words, labels, z, vm in zs:
            zq, _ = r.logits(words, labels)
            d = np.abs(zq[vm] - z[vm])
            mx, sabs, cnt = max(mx, float(d.max())), sabs + float(d.sum()), cnt + d.size
            r.b = g.b + shift
            g.b = g.b + shift
            for t in (0.5, 0.3):
                ef = g.predict_entities(inp["text"], labels, t)
                eq = r.predict_entities(inp["text"], labels, t)
                for j, v in enumerate(change_stats(ef, eq)):
                    st[t][j] += v
            g.b -= shift
            r.b = g.b
        c_sum[key] = {"max_abs_logit_diff": mx, "mean_abs_logit_diff": sabs / cnt}
        for t, (nf, nq, nc) in st.items():
            c_sum[key][f"{'shifted' if a.shift == 'auto' else 'own'}_bias@{t}"] = {"fp32_entities": nf, "variant_entities": nq, "common": nc,
                                               "changed_fraction": (nf + nq - 2 * nc) / max(nf + nq - nc, 1)}
        print(f"[{name}] (c) {key}: {json.dumps(c_sum[key])}")
    # where the t4-w8L63 error comes from: 4-bit table alone (fp32 layers) and int8 layers alone (fp32 table)
    tq, tf = refs["t4-w8L63"], refs["fp32"]
    split = {}
    swaps = [("table4_only", gref.load(dirs["fp32"]), "table", tq.table),
             ("layers8_only", gref.load(dirs["t4-w8L63"]), "table", tf.table)]
    if g.bi:   # the label encoder alone (t4-w8L63) and the int8 head alone, everything else fp32
        swaps += [("label_encoder_t4-w8L63_only", gref.load(dirs["fp32"]), "lenc", tq.lenc),
                  ("head_int8_only", gref.load(dirs["fp32"]), "hsrc", refs["t4-w8L63+h8"])]
    for nm, eng, attr, part in swaps:
        setattr(eng, attr, part)
        ds = [np.abs(eng.logits(w, lb)[0][vm] - z[vm]) for _, w, lb, z, vm in zs]
        split[nm] = {"max_abs_logit_diff": float(max(d.max() for d in ds)),
                     "mean_abs_logit_diff": float(sum(d.sum() for d in ds) / sum(d.size for d in ds))}
    allz = np.concatenate([z[vm].ravel() for _, _, _, z, vm in zs])
    split["fp32_logit_std"] = float(allz.std())
    c_sum["t4-w8L63_error_split"] = split
    print(f"[{name}] (c) split {json.dumps(split)}")

    res = json.loads(a.out.read_text()) if a.out.exists() else {}
    res[name] = {"checkpoint": str(a.ckpt), "encoder_kind": g.g.get("encoder_kind", "uni"),
                 "shift_mode": a.shift,
                 "span_mode": mode, "num_rnn_layers": g.g["num_rnn_layers"], "prefix_space": g.g["prefix_space"],
                 "max_len": g.g["max_len"], "max_width": g.g["max_width"],
                 "a_logits_fp32_vs_library": a_sum, "b_entities_vs_predict_entities": b_sum,
                 "c_quantized_vs_fp32": c_sum, "per_input": rows,
                 "sizes": {k: gpack.sizes(json.loads((v / "manifest.json").read_text()),
                                          (v / "weights.bin").read_bytes(), v) for k, v in dirs.items()}}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
