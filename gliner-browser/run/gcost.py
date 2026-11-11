"""What packing costs a bi-encoder GLiNER in F1, part by part (the zero-shot benchmark through GRefBi).

    python3 run/gcost.py CKPT [--name NAME] [--workers 4] [--limit N] [--arms a,b] [--timing N] [--sizes]
                         [--out results/pack_cost_<name>.json]

CKPT: a bi-encoder gliner directory or cached hub id (e.g. knowledgator/gliner-bi-edge-v2.0).

Three packs are written with gpack into out/gcost/<name>/:
  F  fp32 everywhere
  Q  text t4-w8L63, label t4-w8L63, heads int8 (+-127)       = the main browser build
  A  text t8-w8L31, label t8-w8L127, heads fp32              = the alternative formats (int8 tables, L31 text
                                                               layers, L127 label layers)
and every arm takes each of five components from one of them: text token table (tt), text layers (tl),
label token table (lt), label layers (ll), span head incl. prompt_rep_layer (hd). The BiLSTM is not a
component: the library never applies it to this model (gpack drops it). Swapping components is exact: an
arm computes the same numbers as a single pack built with those formats (gref's components only read
their own records).

Per sentence the text encoder runs once per distinct (tt, tl) among the arms, label vectors once per set
and (lt, ll), and the head and decoder once per arm; decoding is the library's greedy flat decode at each
of bench.py's thresholds, scored with bench.prf (entity-level exact match, micro over the set). Sets and
splits are bench.py's: 7 test sets and the 5 CrossNER dev sets. Each (set, split) is cut into shards of
--shard sentences; each shard's counts (tp, n_pred, n_true per arm and threshold) are checkpointed to
out/gcost/<name>/parts/ and a rerun only computes the missing (shard, arm) pairs. --limit N takes the first
N sentences of each split (bench.py's --limit), recorded in the output.

--timing N: per-stage wall time of a plain GRefBi on N sentences sampled with seed 0 from the 7 test sets
(single thread), for F and Q: tokenize, text encoder, label encoder (uncached: the whole label set per
sentence, and per label), span head, decode at 0.5. --sizes: brotli sizes of every part of F (quality 9),
Q and A (quality 11) and of the composed builds.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
D = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(D / "data" / "hf"))
import bench  # noqa: E402
import gpack  # noqa: E402
import gref  # noqa: E402

PACKS = {"F": ("fp32", "fp32", "fp32"), "Q": ("t4-w8L63", "t4-w8L63", "int8"), "A": ("t8-w8L31", "t8-w8L127", "fp32")}
COMP = ("tt", "tl", "lt", "ll", "hd")
FMT = {"tt": {"F": "fp32", "Q": "4-bit", "A": "8-bit"}, "tl": {"F": "fp32", "Q": "int8 L63", "A": "int8 L31"},
       "lt": {"F": "fp32", "Q": "4-bit", "A": "8-bit"}, "ll": {"F": "fp32", "Q": "int8 L63", "A": "int8 L127"},
       "hd": {"F": "fp32", "Q": "int8 L127", "A": "fp32"}}


def arm(**kw):
    a = dict.fromkeys(COMP, "F")
    a.update(kw)
    return a


ARMS = {
    "fp32": arm(),
    "text_table_t4": arm(tt="Q"),
    "text_table_t8": arm(tt="A"),
    "text_layers_L63": arm(tl="Q"),
    "text_layers_L31": arm(tl="A"),
    "label_table_t4": arm(lt="Q"),
    "label_table_t8": arm(lt="A"),
    "label_layers_L63": arm(ll="Q"),
    "label_layers_L127": arm(ll="A"),
    "heads_int8": arm(hd="Q"),
    "all_t4-w8L63": arm(tt="Q", tl="Q", lt="Q", ll="Q", hd="Q"),
    "all_but_label_encoder_fp32": arm(tt="Q", tl="Q", hd="Q"),
    "all_text_table_t8": arm(tt="A", tl="Q", lt="Q", ll="Q", hd="Q"),
}


def pack_dirs(ck, name):
    """Write (once) the three packs; returns {key: dir}."""
    root = D / "out" / "gcost" / name
    cfg = sd = None
    dirs = {}
    for k, (tv, lv, hv) in PACKS.items():
        out = root / k
        dirs[k] = out
        spec = {"variant": tv, "label_variant": lv, "head": hv}
        if (out / "manifest.json").exists() and json.loads((out / "spec.json").read_text()) == spec:
            continue
        if cfg is None:
            cfg, sd = gpack.load_checkpoint(ck)
            if not gpack.is_bi(cfg):
                raise SystemExit("gcost.py runs bi-encoder checkpoints only")
        m, data = gpack.pack(cfg, sd, tv, hv, lv)
        out.mkdir(parents=True, exist_ok=True)
        gpack.write_tokenizer(ck, out, m["gliner"])
        (out / "weights.bin").write_bytes(data)
        (out / "manifest.json").write_text(json.dumps(m))
        (out / "spec.json").write_text(json.dumps(spec))
    return dirs


class Bank:
    """The three packs and the composed components, built lazily and shared by every arm."""

    def __init__(self, dirs):
        self.e = {k: gref.load(v) for k, v in dirs.items()}
        self.base = self.e["F"]
        self._t, self._l = {}, {}

    def tenc(self, tt, tl):
        k = (tt, tl)
        if k not in self._t:
            src = self.e[tl]
            r = gref.Ref(src.m, src.d)
            r.table = gref.Ref(self.e[tt].m, self.e[tt].d).table
            self._t[k] = r
        return self._t[k]

    def lenc(self, lt, ll):
        k = (lt, ll)
        if k not in self._l:
            r = gref.BertRef(self.e[ll].m["label_encoder"], self.e[ll].d)
            r.table = gref.BertRef(self.e[lt].m["label_encoder"], self.e[lt].d).table
            self._l[k] = r
        return self._l[k]

    def label_vecs(self, lt, ll, labels):
        enc = self.lenc(lt, ll)
        return np.stack([enc.pooled(self.base.label_ids(x)) for x in labels]).astype(np.float32)


_BANK = None


def _init(dirs):
    global _BANK
    _BANK = Bank(dirs)


def run_shard(job):
    """job: (set, split, shard index, start, end, limit, arm names, part path). Writes the part file."""
    s, split, k, a, b, limit, names, path = job
    bank = _BANK
    g = bank.base
    data, labels = bench.load_split(s, split, limit)
    data = data[a:b]
    labels = g.norm_labels(labels)
    arms = {n: ARMS[n] for n in names}
    tkeys = sorted({(x["tt"], x["tl"]) for x in arms.values()})
    lab = {lk: bank.label_vecs(*lk, labels) for lk in sorted({(x["lt"], x["ll"]) for x in arms.values()})}
    th = bench.THRESHOLDS
    trues, preds = [], {n: {t: [] for t in th} for n in arms}
    t0 = time.perf_counter()
    for r in data:
        words = list(r["tokenized_text"])[: g.g["max_len"]] or ["[PAD]"]
        bb = g.build(words, labels)
        ids = np.array(bb["ids"])
        W = {tk: g.words_from_hidden(bank.tenc(*tk).hidden(ids), bb, len(words)) for tk in tkeys}
        for n, x in arms.items():
            z = bank.e[x["hd"]].span_logits(W[(x["tt"], x["tl"])], lab[(x["lt"], x["ll"])])
            for t in th:
                preds[n][t].append([(sp.start, sp.end, sp.label) for sp in g.decode(z, labels, t)])
        trues.append(list(r["ner"]))
    res = {"set": s, "split": split, "shard": k, "range": [a, a + len(data)], "seconds": round(time.perf_counter() - t0, 2),
           "arms": {n: {bench.tkey(t): {q: v for q, v in bench.prf(trues, preds[n][t]).items()
                                         if q in ("tp", "n_pred", "n_true")} for t in th} for n in arms}}
    old = json.loads(path.read_text()) if path.exists() else None
    if old:
        res["arms"] = old["arms"] | res["arms"]
        res["seconds"] += old.get("seconds", 0)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(res))
    tmp.replace(path)
    return s, split, k, len(data), res["seconds"]


def prf_counts(tp, n_pred, n_true):
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_true if n_true else 0.0
    return {"p": p, "r": r, "f1": 2 * p * r / (p + r if p + r else 1), "tp": tp, "n_pred": n_pred, "n_true": n_true}


def assemble(parts_dir, names, limit):
    """Sum shard counts per (set, split, arm, threshold) and aggregate as bench.run_bench does."""
    acc = {}
    for f in sorted(parts_dir.glob("*.json")):
        r = json.loads(f.read_text())
        for n, by in r["arms"].items():
            for t, c in by.items():
                cell = acc.setdefault(n, {}).setdefault(r["set"], {}).setdefault(r["split"], {}).setdefault(
                    t, [0, 0, 0])
                cell[0] += c["tp"]
                cell[1] += c["n_pred"]
                cell[2] += c["n_true"]
    out = {}
    for n in names:
        sets = {s: {sp: {t: prf_counts(*c) for t, c in by.items()} for sp, by in v.items()}
                for s, v in acc.get(n, {}).items()}
        o = {"sets": sets}
        for split, group, key in (("test", bench.SETS, "test_mean7"), ("dev", bench.CROSSNER, "dev_mean5")):
            if all(s in sets and split in sets[s] for s in group):
                o[key] = {bench.tkey(t): sum(sets[s][split][bench.tkey(t)]["f1"] for s in group) / len(group)
                          for t in bench.THRESHOLDS}
        out[n] = o
    return out


def stage_timing(dirs, n):
    """Per-stage seconds per sentence of a plain GRefBi (F and Q packs), single thread."""
    rng = random.Random(0)
    pool = []
    for s in bench.SETS:
        data, labels = bench.load_split(s, "test")
        pool += [(r["tokenized_text"], labels) for r in data]
    sample = rng.sample(pool, n)
    res = {"n_sentences": n, "sample": "random.Random(0).sample of the 7 test sets' sentences pooled",
           "threads": os.environ.get("OPENBLAS_NUM_THREADS"),
           "mean_words": float(np.mean([min(len(w), 2048) for w, _ in sample])),
           "mean_labels": float(np.mean([len(lb) for _, lb in sample]))}
    for key in ("F", "Q"):
        g = gref.load(dirs[key])
        st = {k: [] for k in ("tokenize", "text_encoder", "label_encoder_per_set", "lstm", "span_head", "decode")}
        per_label, ntok = [], []
        for i, (words, labels) in enumerate(sample + sample[:5]):     # 5 warm-up repeats, dropped below
            labels = g.norm_labels(labels)
            words = list(words)[: g.g["max_len"]] or ["[PAD]"]
            t0 = time.perf_counter()
            b = g.build(words, labels)
            t1 = time.perf_counter()
            W = g.words_from_hidden(g.tenc.hidden(np.array(b["ids"])), b, len(words))
            t2 = time.perf_counter()
            lab = np.stack([g.lenc.pooled(g.label_ids(x)) for x in labels]).astype(np.float32)
            t3 = time.perf_counter()
            z = g.span_logits(W, lab)
            t4 = time.perf_counter()
            g.decode(z, labels, 0.5)
            t5 = time.perf_counter()
            if i < 5:
                continue
            for k, v in zip(("tokenize", "text_encoder", "label_encoder_per_set", "span_head", "decode"),
                            (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4)):
                st[k].append(v)
            st["lstm"].append(0.0)
            per_label.append((t3 - t2) / len(labels))
            ntok.append(len(b["ids"]))
        ms = {k: {"mean_ms": 1e3 * float(np.mean(v)), "median_ms": 1e3 * float(np.median(v))} for k, v in st.items()}
        ms["label_encoder_per_label"] = {"mean_ms": 1e3 * float(np.mean(per_label)),
                                         "median_ms": 1e3 * float(np.median(per_label))}
        cached = sum(ms[k]["mean_ms"] for k in ("tokenize", "text_encoder", "span_head", "decode"))
        ms["total_labels_cached_mean_ms"] = cached
        ms["total_labels_uncached_mean_ms"] = cached + ms["label_encoder_per_set"]["mean_ms"]
        ms["mean_text_tokens"] = float(np.mean(ntok))
        ms["lstm_note"] = "0: the bi-encoder forward applies no BiLSTM (library and GRefBi)"
        res[key + ("_fp32" if key == "F" else "_t4-w8L63+label_t4-w8L63+h8")] = ms
    return res


def sizes(dirs):
    out = {}
    for k, d in dirs.items():
        m = json.loads((d / "manifest.json").read_text())
        out[k] = gpack.sizes(m, (d / "weights.bin").read_bytes(), d, quality=9 if k == "F" else 11)
        out[k]["spec"] = json.loads((d / "spec.json").read_text())
    parts = ("text_token_table", "text_layers", "label_token_table", "label_layers", "bilstm", "span_head")
    files = ("tokenizer.json", "labels_tokenizer.json", "manifest.json")

    def build(sel):
        tot = {"raw": 0, "brotli": 0}
        for p, k in zip(parts, sel):
            for q in tot:
                tot[q] += out[k].get(p, {"raw": 0, "brotli": 0})[q]
        for f in files:
            for q in tot:
                tot[q] += out["Q"][f][q]
        return tot
    out["composed"] = {
        "note": "sum of per-part sizes (each part compressed alone) + tokenizers + the Q manifest; "
                "the BiLSTM is dropped (0 bytes)",
        "fp32": build("FFFFFF"), "t4-w8L63 + label t4-w8L63 + h8": build("QQQQQQ"),
        "t8-w8L63 + label t4-w8L63 + h8": build("AQQQQQ")}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--shard", type=int, default=250)
    ap.add_argument("--arms", default=None, help="comma list (default: all)")
    ap.add_argument("--timing", type=int, default=0)
    ap.add_argument("--sizes", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--check-against", type=Path, default=None,
                    help="a gbench_packed.py result for the fp32 pack: record whether the fp32 arm's counts equal it")
    a = ap.parse_args()
    ck = gpack.resolve_ckpt(a.ckpt)
    cfg = json.loads((ck / "gliner_config.json").read_text())
    name = a.name or gpack.ckpt_name(ck, cfg)
    out_path = a.out or D / "results" / f"pack_cost_{name}.json"
    dirs = pack_dirs(ck, name)
    names = a.arms.split(",") if a.arms else list(ARMS)
    tag = f"limit{a.limit}" if a.limit else "full"
    parts_dir = D / "out" / "gcost" / name / f"parts_{tag}"
    parts_dir.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    t_start = time.time()
    if not a.no_bench:
        jobs = []
        for s in bench.SETS:
            for split in ("dev", "test"):
                if split == "dev" and s not in bench.CROSSNER:
                    continue
                n = len(bench.load_split(s, split, a.limit)[0])
                for k, st in enumerate(range(0, n, a.shard)):
                    p = parts_dir / f"{s}__{split}__{k:02d}.json"
                    have = set(json.loads(p.read_text())["arms"]) if p.exists() else set()
                    miss = [x for x in names if x not in have]
                    if miss:
                        jobs.append((s, split, k, st, min(n, st + a.shard), a.limit, miss, p))
        jobs.sort(key=lambda j: -(j[4] - j[3]) * len({(ARMS[x]["tt"], ARMS[x]["tl"]) for x in j[6]}))
        print(f"{len(jobs)} shards to run on {a.workers} workers ({tag})", flush=True)
        if jobs:
            if a.workers > 1:
                import multiprocessing as mp
                with mp.get_context("fork").Pool(a.workers, initializer=_init, initargs=(dirs,)) as pool:
                    for r in pool.imap_unordered(run_shard, jobs):
                        print(f"  done {r[0]} {r[1]} shard {r[2]}: {r[3]} sentences in {r[4]:.0f}s "
                              f"({time.time() - t_start:.0f}s)", flush=True)
            else:
                _init(dirs)
                for j in jobs:
                    r = run_shard(j)
                    print(f"  done {r[0]} {r[1]} shard {r[2]}: {r[3]} sentences in {r[4]:.0f}s", flush=True)
        arms = assemble(parts_dir, names, a.limit)
        res["bench"] = {
            "checkpoint": str(a.ckpt), "limit": a.limit, "subset": "full sets" if not a.limit else
            f"first {a.limit} sentences of each split (bench.py --limit)", "thresholds": list(map(bench.tkey, bench.THRESHOLDS)),
            "metric": "bench.prf: entity-level exact match micro P/R/F1 per set; test_mean7 = mean of 7 test sets, "
                      "dev_mean5 = mean of the 5 CrossNER dev sets; greedy flat decode (flat_ner=True)",
            "packs": {k: {"variant": v[0], "label_variant": v[1], "head": v[2]} for k, v in PACKS.items()},
            "components": FMT, "arm_components": {n: ARMS[n] for n in names}, "arms": arms}
        ref = arms.get("fp32")
        if ref and "test_mean7" in ref:
            res["bench"]["summary"] = {
                n: {"test_mean7@0.5": v.get("test_mean7", {}).get("0.5"), "dev_mean5@0.5": v.get("dev_mean5", {}).get("0.5"),
                    "delta_test_mean7@0.5_vs_fp32": (v["test_mean7"]["0.5"] - ref["test_mean7"]["0.5"])
                    if "test_mean7" in v else None,
                    "delta_dev_mean5@0.5_vs_fp32": (v["dev_mean5"]["0.5"] - ref["dev_mean5"]["0.5"])
                    if "dev_mean5" in v else None,
                    "test_mean7": v.get("test_mean7"), "dev_mean5": v.get("dev_mean5")} for n, v in arms.items()}
            for n, v in res["bench"]["summary"].items():
                if v["test_mean7@0.5"] is not None:
                    print(f"  {n:28} test7@0.5 {100 * v['test_mean7@0.5']:6.2f} ({100 * v['delta_test_mean7@0.5_vs_fp32']:+.2f})"
                          f"  dev5@0.5 {100 * v['dev_mean5@0.5']:6.2f} ({100 * v['delta_dev_mean5@0.5_vs_fp32']:+.2f})")
    if a.check_against and "bench" in res:
        gb = json.loads(a.check_against.read_text())
        fa = res["bench"]["arms"]["fp32"]["sets"]
        cells = diff = 0
        for s_, v in gb["bench"]["sets"].items():
            for sp, r in v.items():
                for t, c in r["by_threshold"].items():
                    cells += 1
                    d = fa.get(s_, {}).get(sp, {}).get(t)
                    diff += not d or (c["tp"], c["n_pred"], c["n_true"]) != (d["tp"], d["n_pred"], d["n_true"])
        res["bench"]["fp32_arm_check"] = {"against": str(a.check_against.relative_to(D)) if a.check_against.is_absolute()
                                          else str(a.check_against), "cells": cells, "cells_differing": diff,
                                          "against_verify": gb.get("verify", {}).get("identical")}
        print("fp32 arm check:", res["bench"]["fp32_arm_check"])
    if a.timing:
        res["stage_timing"] = stage_timing(dirs, a.timing)
        print(json.dumps(res["stage_timing"], indent=1))
    if a.sizes:
        res["sizes"] = sizes(dirs)
        for k, v in res["sizes"]["composed"].items():
            if isinstance(v, dict):
                print(f"  {k:34} {v['raw'] / 1e6:8.2f} MB raw {v['brotli'] / 1e6:8.2f} MB brotli")
    out_path.write_text(json.dumps(res, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
