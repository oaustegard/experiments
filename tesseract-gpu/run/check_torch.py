"""Check server/torch_engine.py against the NumPy reference and write results/torch_check.json.

    python3 run/check_torch.py

1. Per-layer parity of exact mode with every fixture case in data/fixtures/ (float cases:
   max|diff| <= 1e-4; int8 case: no entry more than 1 code off, at most 0.1% off by 1).
   Also pad_table_fast against ref.pad_table for several (H, W).
2. Greedy text on the synthetic benchmark (tessgpu.bench rows "synth", every 16th row):
   share of lines whose text equals ref.forward's greedy text, and mean CER against the
   ground truth, for exact and cudnn modes. eng lines run through models/best and
   models/fast, grc lines through models/best; fra has no model here and is skipped.
3. CPU throughput (lines/s) of both modes at batch 64 and 256, torch.set_num_threads(4).
"""

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "run"))

from tessgpu import bench, decode, ref  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402
from server.torch_engine import TorchTess, pad_table_fast  # noqa: E402
import make_fixtures as mf  # noqa: E402

RESULTS = HERE / "results" / "torch_check.json"
FLOAT_TOL = 1e-4
MODELS = {"eng_best": "models/best/eng.traineddata", "eng_fast": "models/fast/eng.traineddata",
          "grc_best": "models/best/grc.traineddata"}
LAYERS = ["in", "pads", "conv", "pool", "ylstm", "x1", "x2", "x3", "probs"]
QUANT_LAYERS = {"conv", "pool", "ylstm", "x1", "x2", "x3"}


# ----------------------------------------------------------------------------- helpers

def collapse(s):
    return " ".join(s.split())


def levenshtein(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(pred, truth):
    t = collapse(truth)
    return levenshtein(collapse(pred), t) / max(1, len(t))


_REF_CACHE = {}


def _ref_text(job):
    """Worker: greedy text from ref.forward for one (model, grey line)."""
    key, gray = job
    if key not in _REF_CACHE:
        _REF_CACHE[key] = load(str(HERE / MODELS[key]))
    m = _REF_CACHE[key]
    codec = decode.Codec(m)
    g = ref.scale_to_height(gray, m.input_height)
    return decode.greedy(ref.forward(m, g), codec)


def gray_of(path):
    return np.asarray(Image.open(path).convert("L"))


# ----------------------------------------------------------------------------- 1. parity

def check_pad_tables():
    cases = [(36, 3), (36, 4), (36, 17), (36, 100), (36, 522), (36, 839), (48, 257), (48, 1000)]
    out = {"checked": [], "all_equal": True}
    for key in ("eng_best", "eng_fast"):
        m = load(str(HERE / MODELS[key]))
        for H, W in cases:
            a = pad_table_fast(m, H, W)
            b = ref.pad_table(H, W, ref.TRand.for_line(m.sample_iteration), m.int_mode)
            eq = a.shape == b.shape and bool(np.array_equal(a, b))
            out["checked"].append({"model": key, "H": H, "W": W, "equal": eq})
            out["all_equal"] = out["all_equal"] and eq
    return out


def layer_parity(case, model_path, imgs):
    eng = TorchTess(str(HERE / model_path), device="cpu", mode="exact")
    H = eng.H
    man = json.loads((HERE / "data/fixtures" / case / "manifest.json").read_text())
    order = sorted(range(len(imgs)), key=lambda i: -imgs[i].shape[1])   # as make_fixtures
    xs = [ref.normalise(imgs[i], eng.int_mode) for i in order]
    st = eng._stages(xs, keep=True)
    report = {}
    for name in LAYERS:
        p = HERE / "data/fixtures" / case / f"{name}.f32"
        fx = np.fromfile(p, np.float32)
        if name in ("conv", "pool"):
            fx = fx.reshape(-1, 16)
        elif name == "pads":
            fx = fx.reshape(-1, 9)
        elif name == "in":
            fx = fx.reshape(-1)
        elif name == "probs":
            fx = fx.reshape(-1, man["C"])
        else:
            fx = fx.reshape(-1, man["ns"][{"ylstm": "y"}.get(name, name)])
        mx, n1, ng, tot = 0.0, 0, 0, 0
        for k, line in enumerate(man["lines"]):
            W, T, nb = line["W"], line["T"], line["nb"]
            if name == "in":
                seg = fx[line["in_off"]:line["in_off"] + H * W].reshape(H, W)
                ours = st["in"][k]
            elif name == "pads":
                seg = fx[line["pad_off"]:line["pad_off"] + nb]
                ours = st["pads"][k]
            elif name == "conv":
                seg = fx[line["in_off"]:line["in_off"] + H * W].reshape(H, W, 16)
                ours = st["conv"][k]
            elif name == "pool":
                seg = fx[line["pool_off"]:line["pool_off"] + (H // 3) * T].reshape(H // 3, T, 16)
                ours = st["pool"][k]
            else:
                seg = fx[line["t_off"]:line["t_off"] + T]
                ours = st[name][k]
            d = np.abs(seg.astype(np.float64) - np.asarray(ours, np.float64))
            mx = max(mx, float(d.max()))
            n1 += int(((d > 0.5) & (d <= 1.0 + 1e-3)).sum())
            ng += int((d > 1.0 + 1e-3).sum())
            tot += d.size
        if name in QUANT_LAYERS and eng.int_mode:
            ok = ng == 0 and n1 / tot <= 0.001
        else:
            ok = mx <= FLOAT_TOL
        report[name] = {"max_abs_diff": mx, "entries": tot, "off_by_1": n1, "off_by_more": ng,
                        "pass": bool(ok), "criterion": "int codes" if (
                            name in QUANT_LAYERS and eng.int_mode) else "float 1e-4"}
    return {"int_mode": eng.int_mode, "lines": len(imgs), "layers": report,
            "pass": all(v["pass"] for v in report.values())}


def parity():
    parity_imgs = [np.asarray(Image.open(HERE / f"data/parity/l{i}.png")) for i in range(3)]
    eng_imgs = parity_imgs + mf.lines_for("eng", 3, 36)
    grc_imgs = mf.lines_for("grc", 3, 48)
    cases = {}
    cases["fast_eng"] = layer_parity("fast_eng", MODELS["eng_fast"], eng_imgs)
    cases["best_eng"] = layer_parity("best_eng", MODELS["eng_best"], eng_imgs)
    cases["best_grc"] = layer_parity("best_grc", MODELS["grc_best"], grc_imgs)
    return {"float_tolerance": FLOAT_TOL, "int_off_by_one_fraction_max": 0.001,
            "cases": cases, "pass": all(c["pass"] for c in cases.values())}


# ----------------------------------------------------------------------------- 2. text

def text_agreement(pool):
    rows = bench.rows("synth")[::16]
    by_model = {"eng_best": [], "eng_fast": [], "grc_best": []}
    skipped = 0
    for r in rows:
        if r["lang"] == "eng":
            by_model["eng_best"].append(r)
            by_model["eng_fast"].append(r)
        elif r["lang"] == "grc":
            by_model["grc_best"].append(r)
        else:
            skipped += 1
    grays = {}
    report = {"sampled_rows": len(rows), "skipped_no_model": skipped, "models": {}}
    for key, rs in by_model.items():
        if not rs:
            continue
        grays_k = [gray_of(bench.resolve(r, HERE / "data/lines/labels.jsonl")) for r in rs]
        refs = list(pool.map(_ref_text, [(key, g) for g in grays_k], chunksize=4))
        m_path = str(HERE / MODELS[key])
        entry = {"lines": len(rs), "ref_cer": float(np.mean(
            [cer(t, r["text"]) for t, r in zip(refs, rs)]))}
        for mode in ("exact", "cudnn"):
            eng = TorchTess(m_path, device="cpu", mode=mode)
            preds = eng.texts(grays_k)
            agree = np.mean([p == t for p, t in zip(preds, refs)])
            entry[mode] = {"text_agreement": float(agree),
                           "mean_cer_vs_ground_truth": float(np.mean(
                               [cer(p, r["text"]) for p, r in zip(preds, rs)]))}
        report["models"][key] = entry
    tot = sum(v["lines"] for v in report["models"].values())
    for mode in ("exact", "cudnn"):
        ag = sum(v[mode]["text_agreement"] * v["lines"] for v in report["models"].values()) / tot
        ce = sum(v[mode]["mean_cer_vs_ground_truth"] * v["lines"]
                 for v in report["models"].values()) / tot
        report[mode] = {"text_agreement": ag, "mean_cer_vs_ground_truth": ce}
    report["scored_model_lines"] = tot
    report["ref_mean_cer_vs_ground_truth"] = sum(
        v["ref_cer"] * v["lines"] for v in report["models"].values()) / tot
    return report


# ----------------------------------------------------------------------------- 3. speed

def throughput():
    rows = [r for r in bench.rows("synth") if r["lang"] == "eng"][:256]
    imgs = [gray_of(bench.resolve(r, HERE / "data/lines/labels.jsonl")) for r in rows]
    torch.set_num_threads(4)
    out = {"threads": 4, "lines": len(imgs), "model": MODELS["eng_best"], "lines_per_s": {}}
    for mode in ("exact", "cudnn"):
        eng = TorchTess(str(HERE / MODELS["eng_best"]), device="cpu", mode=mode)
        eng.recognize(imgs[:16])                      # warm-up
        for batch in (64, 256):
            t0 = time.perf_counter()
            for i in range(0, len(imgs), batch):
                eng.recognize(imgs[i:i + batch])
            dt = time.perf_counter() - t0
            out["lines_per_s"][f"{mode}_b{batch}"] = len(imgs) / dt
    return out


# ----------------------------------------------------------------------------- main

def main():
    torch.set_num_threads(4)
    t0 = time.time()
    pads = check_pad_tables()
    par = parity()
    with ProcessPoolExecutor(max_workers=4) as pool:
        agree = text_agreement(pool)
    speed = throughput()
    result = {
        "engine": "server/torch_engine.py TorchTess", "torch": torch.__version__,
        "reference": "tessgpu/ref.py", "pad_table": pads, "parity": par,
        "text": agree, "throughput": speed,
        "summary": {
            "parity_pass": par["pass"] and pads["all_equal"],
            "exact_text_agreement": agree["exact"]["text_agreement"],
            "cudnn_text_agreement": agree["cudnn"]["text_agreement"],
            "exact_cer": agree["exact"]["mean_cer_vs_ground_truth"],
            "cudnn_cer": agree["cudnn"]["mean_cer_vs_ground_truth"],
            "ref_cer": agree["ref_mean_cer_vs_ground_truth"],
            "lines_per_s": speed["lines_per_s"],
        },
        "seconds": round(time.time() - t0, 1),
    }
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=1))
    s = result["summary"]
    print("parity: layers", "PASS" if par["pass"] else "FAIL",
          "| pad tables", "PASS" if pads["all_equal"] else "FAIL")
    for name, c in par["cases"].items():
        worst = {k: round(v["max_abs_diff"], 7) for k, v in c["layers"].items()}
        print(f"  {name}: {'PASS' if c['pass'] else 'FAIL'} max|diff| {worst}")
    print(f"text agreement vs ref ({agree['scored_model_lines']} model-lines): "
          f"exact {s['exact_text_agreement']:.4f}, cudnn {s['cudnn_text_agreement']:.4f}")
    print(f"CER vs ground truth: exact {s['exact_cer']:.4f}, cudnn {s['cudnn_cer']:.4f}, "
          f"ref {s['ref_cer']:.4f}")
    print("lines/s:", {k: round(v, 2) for k, v in s["lines_per_s"].items()})
    print("wrote", RESULTS)


if __name__ == "__main__":
    main()
