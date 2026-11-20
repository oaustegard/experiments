"""Measure cheap line-level arms for tesseract-gpu on the line benchmark.

    python3 run/improve_lines.py            # run every arm, write results/improve_lines.json
    python3 run/improve_lines.py --limit 40 # quick smoke run: first 40 lines per set and language
    python3 run/improve_lines.py --summary  # print the ranked table from the json

Sample: every 4th synthetic sentence (all eight conditions, 1,200 lines) and every 2nd real line
(400). Each language is read by its own tessdata_best model; the fast model is used only by the
model-ensemble arms. Every (model, preprocessing) pair is one batched forward pass over the
sample and is computed once: arms that share a pair reuse its readings, and an arm's cost is
the number of pairs it needs per line.
"""

import argparse
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "improve_lines.json"
LANGS = ("eng", "fra", "grc")
FAST_COMPUTE = 0.32        # eng fast vs eng best, RESULTS.md Finding 7; applied to every language
ENSEMBLE_VARIANTS = [(p, s) for p in (2, 4, 8) for s in (False, True)]
GREEDY_MIN_GAIN = 0.0005   # absolute change in the mean of real and synthetic CER to accept a step
THREADS = int(os.environ.get("IMPROVE_THREADS", "1"))  # 4 threads crawl on the shared 4-CPU box (load 20); 1 is 25x faster here

_cache = {}

# Photometric views: the page-level transforms of the sibling Indra OCR study, used exactly as
# defined there (run/perturb.py PAGE). Applied to the uint8 line crop before improve.prepare.
PERTURB_PATH = Path("/home/user/experiments/indra-browser/run/perturb.py")
VIEWS = ["gamma0.5", "gamma2", "contrast0.6", "contrast1.5", "otsu", "thicken", "thin"]
_views = {}


def view_fn(name):
    if not _views:
        import importlib.util
        spec = importlib.util.spec_from_file_location("indra_perturb", PERTURB_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _views.update(mod.PAGE)
    return _views[name]


def norm(s):
    return " ".join(s.split())


def cer(pred, truth, casefold=False):
    """Levenshtein / len(truth) after whitespace collapse; casefold also lowercases both
    (SROIE ground truth is uppercase where the receipt prints lowercase)."""
    t, p = norm(truth), norm(pred)
    if casefold:
        t, p = t.casefold(), p.casefold()
    if not t:
        return None
    return Levenshtein.distance(p, t) / len(t)


# ----------------------------------------------------------------------------- data

def load_sample(limit=None):
    from tessgpu import bench

    synth_rows = bench.rows("synth")
    ids = []
    for r in synth_rows:
        if not ids or ids[-1] != r["id"]:
            if r["id"] not in ids:
                ids.append(r["id"])
    keep_ids = set(ids[::4])                                 # every 4th sentence, all conditions
    synth = [r for r in synth_rows if r["id"] in keep_ids]
    real = bench.rows("real")[::2]
    lines = []
    for r in synth:
        lines.append({**r, "set": "synth"})
    for r in real:
        lines.append({**r, "set": "real", "lang": "eng", "cond": "real"})
    if limit:
        keep = []
        counts = {}
        for r in lines:
            key = (r["set"], r["lang"])
            if counts.get(key, 0) < limit:
                counts[key] = counts.get(key, 0) + 1
                keep.append(r)
        lines = keep
    for i, r in enumerate(lines):
        r["idx"] = i
        r["gray"] = np.asarray(_open_gray(r["path"]))
    return lines


def _open_gray(path):
    from PIL import Image
    return Image.open(path).convert("L")


# ----------------------------------------------------------------------------- engine

_engines = {}


def engine(lang, kind):
    from server.torch_engine import TorchTess
    import torch
    torch.set_num_threads(THREADS)
    key = (lang, kind)
    if key not in _engines:
        path = ROOT / "models" / kind / f"{lang}.traineddata"
        _engines[key] = TorchTess(str(path), device="cpu", mode="exact")
    return _engines[key]


def read_variant(lines, lang, kind, flags):
    """One batched forward pass per (model, flags) over all sample lines of a language.
    Returns {idx: (confidence or None, text)}."""
    from tessgpu import beam, improve

    key = (lang, kind, tuple(sorted(flags.items())))
    if key in _cache:
        return _cache[key]
    cpath = _cache_path(key)
    if cpath is not None and cpath.exists():
        import pickle
        _cache[key] = pickle.loads(cpath.read_bytes())
        return _cache[key]
    eng = engine(lang, kind)
    subset = [r for r in lines if r["lang"] == lang]
    view = flags.get("view")
    fn = view_fn(view) if view else None
    imgs = [improve.prepare(fn(r["gray"]) if fn else r["gray"], flags, eng.H, _inverted(r))
            for r in subset]
    out = {}
    CH = 200
    for c0 in range(0, len(subset), CH):
        probs = eng.recognize(imgs[c0:c0 + CH])
        for j, p in enumerate(probs):
            r = subset[c0 + j]
            conf = improve.confidence(p)
            ids = beam.decode(p, eng.model) if len(p) else []
            text = norm("".join(eng.model.unicharset[u] for u in ids))
            out[r["idx"]] = (conf, text)
    _cache[key] = out
    if cpath is not None:
        import pickle
        cpath.parent.mkdir(parents=True, exist_ok=True)
        cpath.write_bytes(pickle.dumps(out))
    return out


def _cache_path(key):
    """Optional on-disk cache of readings (IMPROVE_CACHE=dir), so a killed run resumes."""
    d = os.environ.get("IMPROVE_CACHE")
    if not d:
        return None
    import hashlib
    h = hashlib.sha1(repr((key, THREADS)).encode()).hexdigest()[:16]
    return Path(d) / f"{key[0]}_{key[1]}_{h}.pkl"


def _inverted(r):
    from tessgpu import ref
    if "inv" not in r:
        r["inv"] = bool(ref.needs_invert(np.asarray(r["gray"], np.uint8)))
    return r["inv"]


# ----------------------------------------------------------------------------- arms

def single(flags):
    def fn(lines, lang, lidx):
        return [("best", flags)]
    return fn


def model_ensemble(flags):
    def keys(lines, lang, lidx):
        return [("best", flags), ("fast", flags)]
    return keys


def variant_ensemble(base):
    def keys(lines, lang, lidx):
        out = []
        for p, s in ENSEMBLE_VARIANTS:
            f = dict(base)
            f["pad"] = p
            f["stretch"] = s
            out.append(("best", f))
        return out
    return keys


def passes_of(keys):
    """Forward passes per line and compute units (fast model counted at FAST_COMPUTE)."""
    n = len(keys)
    units = sum(FAST_COMPUTE if k == "fast" else 1.0 for k, _ in keys)
    return n, units


def run_arm(lines, arm_keys, select):
    """select(list_of_readings_per_line, per_line_conf...) -> text per idx.
    arm_keys: function (lines, lang, idx) -> [(kind, flags)]. select: 'first', 'conf', 'rover'."""
    preds = {}
    by_lang = {}
    for r in lines:
        by_lang.setdefault(r["lang"], []).append(r)
    keylist = None
    for lang, ls in by_lang.items():
        keys = arm_keys(lines, lang, None)
        keylist = keys if keylist is None else keylist
        readings = [read_variant(lines, lang, k, f) for k, f in keys]
        for r in ls:
            cands = [rd[r["idx"]] for rd in readings]
            preds[r["idx"]] = pick(cands, select, r["text"])
    return preds, keylist


def pick(cands, select, truth=None):
    from tessgpu import improve
    if select == "first":
        return cands[0][1]
    if select == "oracle":
        # CEILING, not a score: uses the truth to pick the closest reading (baseline wins ties)
        best = min(cands, key=lambda ct: (cer(ct[1], truth) if cer(ct[1], truth) is not None
                                          else 0.0))
        return best[1]
    if select == "rover_equal":
        return improve.vote([(t, c) for c, t in cands], weighted=False)
    if select == "conf":
        i = improve.pick_by_confidence([c for c, _ in cands])
        return cands[0][1] if i is None else cands[i][1]
    if select == "rover":
        return improve.vote([(t, c) for c, t in cands])
    raise ValueError(select)


def score(lines, preds):
    res = {"real": {}, "synth": {}}
    tot = {"real": [], "synth": []}
    by = {}
    for r in lines:
        c = cer(preds[r["idx"]], r["text"])
        if c is None:
            continue
        s = r["set"]
        tot[s].append(c)
        by.setdefault((s, r["lang"], r["cond"]), []).append(c)
    out = {}
    real_all = float(np.mean(tot["real"])) if tot["real"] else None
    syn_all = float(np.mean(tot["synth"])) if tot["synth"] else None
    out["real"] = {"all": real_all, "by_cond": {"real": real_all}} if tot["real"] else None
    if tot["real"]:
        ci = [cer(preds[r["idx"]], r["text"], casefold=True) for r in lines
              if r["set"] == "real" and cer(r["text"], r["text"]) is not None]
        out["real"]["all_casefold"] = float(np.mean(ci))
        out["real"]["by_source"] = {}
        for src in sorted({r["id"].split("-")[0] for r in lines if r["set"] == "real"}):
            v = [cer(preds[r["idx"]], r["text"]) for r in lines
                 if r["set"] == "real" and r["id"].split("-")[0] == src]
            vc = [cer(preds[r["idx"]], r["text"], casefold=True) for r in lines
                  if r["set"] == "real" and r["id"].split("-")[0] == src]
            out["real"]["by_source"][src] = {"all": float(np.mean(v)), "all_casefold": float(np.mean(vc))}
    if tot["synth"]:
        bl, bc, blc = {}, {}, {}
        for (s, lang, cond), v in by.items():
            if s != "synth":
                continue
            bl.setdefault(lang, []).extend(v)
            bc.setdefault(cond, []).extend(v)
            blc.setdefault(lang, {})[cond] = float(np.mean(v))
        out["synth"] = {
            "all": syn_all,
            "by_lang": {k: float(np.mean(v)) for k, v in bl.items()},
            "by_cond": {k: float(np.mean(v)) for k, v in bc.items()},
            "by_lang_cond": blc,
        }
    else:
        out["synth"] = None
    return out


# ----------------------------------------------------------------------------- driver

def objective(lines, flags):
    preds, _ = run_arm(lines, single(flags), "first")
    s = score(lines, preds)
    vals = [s["real"]["all"], s["synth"]["all"]]
    return float(np.mean(vals)), s


PRE_GROUPS = {
    "invert": [{"invert": True}],
    "pad": [{"pad": p} for p in (2, 4, 8)],
    "stretch": [{"stretch": True}],
    "upscale": [{"upscale": True}],
    "deskew": [{"deskew": True}],
}


def greedy_combination(lines, log, start):
    chosen = dict(start)
    used = set()
    best_obj, _ = objective(lines, chosen)
    steps = []
    while True:
        cands = []
        for g, opts in PRE_GROUPS.items():
            if g in used:
                continue
            for o in opts:
                f = {**chosen, **o}
                obj, _ = objective(lines, f)
                cands.append((obj, g, o))
                log(f"  greedy try {f}: {obj:.5f}")
        if not cands:
            break
        obj, g, o = min(cands, key=lambda c: c[0])
        if best_obj - obj < GREEDY_MIN_GAIN:
            break
        chosen.update(o)
        used.add(g)
        steps.append({"added": o, "objective_before": best_obj, "objective_after": obj})
        best_obj = obj
        log(f"greedy accept {o} -> {chosen} objective {obj:.5f}")
    return chosen, steps


def per_line(lines, preds, sset):
    out = {}
    for r in lines:
        if r["set"] != sset:
            continue
        c = cer(preds[r["idx"]], r["text"])
        if c is not None:
            out[r["idx"]] = c
    return out


def vs_baseline(lines, preds, base_preds):
    """Lines fixed (baseline CER > 0 and arm CER == 0) and broken (baseline CER == 0 and arm
    CER > 0), plus lines that get closer or further from the truth. Real set: case-sensitive."""
    res, ids = {}, {}
    for sset in ("synth", "real"):
        b = per_line(lines, base_preds, sset)
        a = per_line(lines, preds, sset)
        common = b.keys() & a.keys()
        fixed = {i for i in common if b[i] > 0 and a[i] == 0}
        broken = {i for i in common if b[i] == 0 and a[i] > 0}
        res[sset] = {
            "lines": len(common),
            "baseline_failed": sum(1 for i in common if b[i] > 0),
            "fixed": len(fixed),
            "broken": len(broken),
            "net_fixed": len(fixed) - len(broken),
            "closer": sum(1 for i in common if a[i] < b[i]),
            "further": sum(1 for i in common if a[i] > b[i]),
        }
        ids[sset] = fixed
    return res, ids


def describe(keys):
    return [{"model": k, "flags": f} for k, f in keys]


NEW_BASE = {"tight": True, "flatfield": True}   # the reference every arm is compared against


def main_run(limit):
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:7.1f}s] {msg}", flush=True)

    lines = load_sample(limit)
    log(f"sample: {sum(r['set'] == 'synth' for r in lines)} synth, "
        f"{sum(r['set'] == 'real' for r in lines)} real lines")

    arms = {}
    all_preds = {}

    def add(name, description, keyfn, select):
        preds, keys = run_arm(lines, keyfn, select)
        s = score(lines, preds)
        n, units = passes_of(keys)
        arms[name] = {
            "description": description,
            "passes_per_line": n,
            "compute_units_per_line": round(units, 3),
            "keys": describe(keys),
            "cer": s,
        }
        all_preds[name] = preds
        log(f"{name:28s} real {s['real']['all']:.4f} (ci {s['real']['all_casefold']:.4f})  "
            f"synth {s['synth']['all']:.4f}  passes {n}")

    def views_keys(views, base=None):
        base = {} if base is None else base

        def fn(lines, lang, idx):
            ks = [("best", dict(base))]
            return ks + [("best", {**base, "view": v}) for v in views]
        return fn

    base = {}
    add("baseline", "Line as is, scaled to model height, beam-decoded (tessdata_best).",
        single(base), "first")
    add("invert", "Flip the line when ref.needs_invert is true.",
        single({"invert": True}), "first")
    for p in (2, 4, 8):
        add(f"pad{p}", f"Add {p} px background border before scaling.",
            single({"pad": p}), "first")
    add("stretch", "Linear stretch of the 2nd..98th percentile to 0..255.",
        single({"stretch": True}), "first")
    add("upscale", "Lines under 20 px: Lanczos straight to model height.",
        single({"upscale": True}), "first")
    add("deskew", "Per-line rotation in +-3 deg maximising row-profile variance.",
        single({"deskew": True}), "first")
    add("ens_conf", "Six variants (pad 2/4/8 x stretch on/off), pick the highest confidence.",
        variant_ensemble({}), "conf")
    add("ens_rover", "Same six variants, ROVER-style character voting weighted by confidence.",
        variant_ensemble({}), "rover")
    add("model_ens", "tessdata_best and tessdata_fast, pick the highest confidence.",
        model_ensemble({}), "conf")

    add("tight", "Crop to the ink bounding box plus 4 px (ink found after flatfield).",
        single({"tight": True}), "first")
    add("tight_ff", "NEW BASELINE: tight crop, then flatfield.",
        single(NEW_BASE), "first")

    # Views and votes sit on the new baseline (tight + flatfield).
    for v in VIEWS:
        add(f"view_{v}", f"Photometric view {v} on the tight+flatfield line.",
            single({**NEW_BASE, "view": v}), "first")
    add("views_conf", "New baseline plus all seven views, highest confidence wins.",
        views_keys(VIEWS, NEW_BASE), "conf")
    add("views_oracle", "CEILING (uses the truth): per line, the closest of the new baseline and all views.",
        views_keys(VIEWS, NEW_BASE), "oracle")
    three = ["gamma0.5", "contrast0.6"]
    add("rover_equal_3", "ROVER, equal weights, new baseline + gamma0.5 + contrast0.6 (majority of three).",
        views_keys(three, NEW_BASE), "rover_equal")
    add("rover_equal_all", "ROVER, equal weights, new baseline + all seven views.",
        views_keys(VIEWS, NEW_BASE), "rover_equal")
    add("rover_conf_3", "ROVER, confidence weights, new baseline + gamma0.5 + contrast0.6.",
        views_keys(three, NEW_BASE), "rover")
    add("rover_conf_all", "ROVER, confidence weights, new baseline + all seven views.",
        views_keys(VIEWS, NEW_BASE), "rover")

    chosen, steps = greedy_combination(lines, log, NEW_BASE)
    log(f"best combination: {chosen}")
    add("combo", "Best preprocessing combination, greedy forward selection from the new baseline.",
        single(chosen), "first")
    add("combo+ens_conf", "Best combination with the six-variant confidence ensemble.",
        variant_ensemble(chosen), "conf")
    add("combo+ens_rover", "Best combination with the six-variant ROVER vote.",
        variant_ensemble(chosen), "rover")
    add("combo+model_ens", "Best combination with the best/fast model ensemble.",
        model_ensemble(chosen), "conf")

    # Comparisons: every arm against the raw baseline and against the new baseline.
    refs = {"raw": all_preds["baseline"], "tight_ff": all_preds["tight_ff"]}
    fixed_ids = {}
    for name in arms:
        vs = {}
        for rname, rp in refs.items():
            res, ids = vs_baseline(lines, all_preds[name], rp)
            vs[rname] = res
            if rname == "tight_ff":
                fixed_ids[name] = ids
        arms[name]["vs_raw_baseline"] = vs["raw"]
        arms[name]["vs_tight_ff"] = vs["tight_ff"]

    overlap = {}
    for sset in ("synth", "real"):
        ranked = sorted((n for n in fixed_ids if n.startswith("view_")),
                        key=lambda n: -len(fixed_ids[n][sset]))[:3]
        sets = {n: fixed_ids[n][sset] for n in ranked}
        pairs = {}
        for i, a in enumerate(ranked):
            for b in ranked[i + 1:]:
                pairs[f"{a} & {b}"] = len(sets[a] & sets[b])
        triple = len(set.intersection(*sets.values())) if len(ranked) == 3 else None
        overlap[sset] = {
            "top_views_by_fixes_vs_tight_ff": {n: len(sets[n]) for n in ranked},
            "pairwise_shared_fixes": pairs,
            "shared_by_all_three": triple,
            "union_of_top_three": len(set.union(*sets.values())) if ranked else 0,
        }

    ref_arm = arms["tight_ff"]["cer"]
    ref_raw = arms["baseline"]["cer"]
    improvements = []
    for name, a in arms.items():
        if name == "tight_ff":
            continue
        rr = a["cer"]["real"]["all"]
        ss = a["cer"]["synth"]["all"]
        improvements.append({
            "arm": name,
            "real_cer": rr,
            "real_gain": ref_arm["real"]["all"] - rr,
            "real_cer_casefold": a["cer"]["real"]["all_casefold"],
            "real_gain_casefold": ref_arm["real"]["all_casefold"] - a["cer"]["real"]["all_casefold"],
            "synth_cer": ss,
            "synth_gain": ref_arm["synth"]["all"] - ss,
            "real_gain_vs_raw_baseline": ref_raw["real"]["all"] - rr,
            "synth_gain_vs_raw_baseline": ref_raw["synth"]["all"] - ss,
            "passes_per_line": a["passes_per_line"],
            "compute_units_per_line": a["compute_units_per_line"],
        })
    improvements.sort(key=lambda x: (-x["real_gain"], -x["synth_gain"]))

    doc = {
        "meta": {
            "sample": {
                "synth_lines": sum(r["set"] == "synth" for r in lines),
                "synth_sentences": len({r["id"] for r in lines if r["set"] == "synth"}),
                "real_lines": sum(r["set"] == "real" for r in lines),
                "languages": list(LANGS),
                "rule": "every 4th synthetic sentence (all 8 conditions); every 2nd real line",
            },
            "reading": "tessdata_best per language, exact mode, beam decode (tessgpu.beam)",
            "new_baseline": "tight + flatfield (tessgpu.improve.prepare); views and votes sit on it",
            "improvements_ranked_by": "gain vs the new baseline (tight_ff): real, then synthetic",
            "fast_compute_ratio": FAST_COMPUTE,
            "cer": "Levenshtein / len(truth) after whitespace collapse (rapidfuzz); real set reported "
                   "case-sensitive and case-folded",
            "fixed_broken": "fixed: reference exact-match is wrong (CER > 0) and arm is exact; "
                            "broken: reference exact and arm wrong",
            "greedy_objective": "mean of real CER and synthetic CER, starting from the new baseline",
            "greedy_steps": steps,
            "oracle": "views_oracle is a CEILING: it picks by the truth",
            "selection_bias": "the combination and the rankings are chosen on the same sample "
                              "they are reported on; no held-out split",
            "runtime_s": round(time.time() - t0, 1),
        },
        "best_combination": chosen,
        "overlap_of_fixes": overlap,
        "arms": arms,
        "improvements": improvements,
    }
    if not limit:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(doc, indent=2))
        log(f"wrote {OUT}")
    else:
        log("limit run: json not written")
    return doc


def summary():
    doc = json.loads(OUT.read_text())
    rows = doc["improvements"]
    print(f"{'arm':20s} {'real CER':>9s} {'gain':>8s} {'real ci':>8s} {'gain':>8s} "
          f"{'synth CER':>10s} {'gain':>8s} {'passes':>7s} {'units':>6s}")
    for label, key in (("raw baseline", "baseline"), ("NEW baseline tight_ff", "tight_ff")):
        b = doc["arms"][key]["cer"]
        print(f"{label:20s} {b['real']['all']:9.4f} {'':>8s} {b['real']['all_casefold']:8.4f} {'':>8s} "
              f"{b['synth']['all']:10.4f} {'':>8s} {doc['arms'][key]['passes_per_line']:7d} "
              f"{doc['arms'][key]['compute_units_per_line']:6.2f}")
    for x in rows:
        print(f"{x['arm']:20s} {x['real_cer']:9.4f} {x['real_gain']:+8.4f} "
              f"{x['real_cer_casefold']:8.4f} {x['real_gain_casefold']:+8.4f} "
              f"{x['synth_cer']:10.4f} {x['synth_gain']:+8.4f} {x['passes_per_line']:7d} "
              f"{x['compute_units_per_line']:6.2f}")
    print("best combination:", doc["best_combination"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.summary:
        summary()
    else:
        main_run(args.limit)
