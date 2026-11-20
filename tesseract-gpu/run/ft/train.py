"""Fine-tune Tesseract's eng tessdata_best LSTM with CTC on damaged document lines; export a .traineddata.

    python3 run/ft/train.py prep                      # data/ft/prep.pkl: line images scaled to model height
    python3 run/ft/train.py train --steps 600 --run pilot1 [--sroie]
    python3 run/ft/train.py export --run pilot1       # data/ft/<run>/eng.traineddata

Inputs to the network are prepared as the engines prepare them: 60% of training lines go through
tessgpu.lineprep.clean_line (the pipeline default), 40% stay as rendered (inverted ones flipped, as the engines
do), then ref.scale_to_height and ref.normalise. Batches of 24 lines of similar width; Adam, lr 1e-4 after a
100-step warm-up, gradients clipped at 1; out-of-image convolution slots draw fresh uniform [-1, 1] values.

Validation during training (every --eval-every steps, greedy decoding): held out by a fixed rule, never trained
on: generated lines with index % 50 == 0 (val_synth) and SROIE lines from receipts with number % 10 == 0
(val_sroie, CER after casefolding). With --eval local the run instead scores the experiment's evaluation sets
(data/real FUNSD and SROIE, 100 synthetic eng lines per condition); that is for checks, not for choosing a
checkpoint. Export takes the last checkpoint. Log: data/ft/<run>/log.jsonl. Learning rate: --lr after a 100-step
warm-up, cosine decay to a tenth by the last step. --device cuda:N trains on a GPU (run/ft/kaggle_job.py).
"""
import argparse
import json
import math
import pickle
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from rapidfuzz.distance import Levenshtein

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from tessgpu import decode, lineprep, ref  # noqa: E402
from tessgpu.tessnet import TessNet, encode_text, export  # noqa: E402
from tessgpu.traineddata import load  # noqa: E402

FT = ROOT / "data/ft"
MODEL = ROOT / "models/best/eng.traineddata"
MAXW = 1800
BATCH = 24


def norm(s):
    return " ".join(s.split())


def cer(gt, pred, fold=False):
    g, p = norm(gt), norm(pred)
    if fold:
        g, p = g.casefold(), p.casefold()
    return Levenshtein.distance(g, p) / max(1, len(g))


def _prep_one(args):
    path, clean, H = args
    g = np.asarray(Image.open(path).convert("L"))
    if clean:
        g = lineprep.clean_line(g)
    elif ref.needs_invert(g):
        g = (255 - g).astype(np.uint8)
    s = ref.scale_to_height(g, H)
    return s if s.shape[1] <= MAXW and s.shape[1] >= 9 else None


def prep(extra_label_files=()):
    m = load(MODEL)
    rng = random.Random(7)
    items = []
    for lf, base in [(FT / "labels.jsonl", FT)] + [(Path(p), Path(p).parent) for p in extra_label_files]:
        for l in open(lf, encoding="utf-8"):
            r = json.loads(l)
            codes = encode_text(r["text"], m)
            if not codes:
                continue
            src = r.get("src", "synth")
            val = (r.get("receipt", 1) % 10 == 0) if src != "synth" else (r.get("i", 1) % 50 == 0)
            items.append((str(base / r["file"]), rng.random() < 0.6, r["text"], codes, src, val))
    with Pool(2) as pool:
        imgs = pool.map(_prep_one, [(p, c, m.input_height) for p, c, *_ in items], chunksize=64)
    data = [(im, t, c, src, val) for im, (_, _, t, c, src, val) in zip(imgs, items) if im is not None]
    return data


def val_sets(data, m):
    out = {"val_synth": []}
    for im, t, c, src, val in data:
        if val:
            out.setdefault(f"val_{src}", []).append((ref.normalise(im, False), t))
    return {k: v for k, v in out.items() if v}


def eval_sets(m):
    sets = {}
    real = [json.loads(l) for l in open(ROOT / "data/real/labels.jsonl", encoding="utf-8")]
    for name in ("funsd", "sroie"):
        sets[f"real_{name}"] = [(ROOT / r["file"], r["text"]) for r in real if r["set"] == name]
    syn = [json.loads(l) for l in open(ROOT / "data/lines/labels.jsonl", encoding="utf-8")]
    for cond in ("clean", "blur", "noise", "jpeg", "lowres", "rotate", "shade", "invert"):
        rows = [r for r in syn if r["lang"] == "eng" and r["cond"] == cond][:100]
        sets[f"synth_{cond}"] = [(ROOT / "data/lines" / r["file"], r["text"]) for r in rows]
    out = {}
    for k, rows in sets.items():
        xs = []
        for p, t in rows:
            g = lineprep.clean_line(np.asarray(Image.open(p).convert("L")))
            xs.append((ref.normalise(ref.scale_to_height(g, m.input_height), False), t))
        out[k] = xs
    return out


@torch.no_grad()
def evaluate(net, codec, sets, gen):
    net.eval()
    res = {}
    for k, rows in sets.items():
        errs = []
        for c0 in range(0, len(rows), 64):
            chunk = rows[c0:c0 + 64]
            logp, Ls, order = net([x for x, _ in chunk], generator=gen)
            P = logp.exp().cpu().numpy()
            for b, i in enumerate(order):
                pred = decode.greedy(P[:int(Ls[b]), b], codec)
                errs.append(cer(chunk[i][1], pred, fold=k.startswith("real") or k == "val_sroie"))
        res[k] = round(float(np.mean(errs)), 5)
    syn = [v for k, v in res.items() if k.startswith("synth")]
    if syn:
        res["synth_mean"] = round(float(np.mean(syn)), 5)
    if "real_funsd" in res:
        res["real_mean"] = round((res["real_funsd"] + res["real_sroie"]) / 2, 5)
    net.train()
    return res


def batches(data, rng, size):
    idx = sorted(range(len(data)), key=lambda i: data[i][0].shape[1] + rng.random() * 30)
    bs = [idx[i:i + size] for i in range(0, len(idx), size)]
    rng.shuffle(bs)
    return bs


def train(a):
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    m = load(MODEL)
    codec = decode.Codec(m)
    run = FT / a.run
    run.mkdir(parents=True, exist_ok=True)
    data = pickle.load(open(FT / a.data, "rb"))
    sets = eval_sets(m) if a.eval == "local" else val_sets(data, m)
    real = set(a.real_src) | ({"sroie"} if a.sroie else set())
    sets = {k: v for k, v in sets.items() if k == "val_synth" or k[4:] in real or a.eval == "local"}
    data = [d for d in data if not d[4] and (d[3] == "synth" or d[3] in real)]
    print(f"{len(data)} training lines ({sum(d[3] != 'synth' for d in data)} real: {sorted(real)}); eval sets "
          f"{ {k: len(v) for k, v in sets.items()} }", flush=True)
    dev = torch.device(a.device)
    net = TessNet(m).to(dev)
    net.init = {k: v.to(dev) for k, v in net.init.items()}
    opt = torch.optim.Adam(net.parameters(), lr=a.lr)
    step = 0
    ck = run / "last.pt"
    if ck.exists():
        st = torch.load(ck, weights_only=False, map_location=dev)
        net.load_state_dict(st["net"]); opt.load_state_dict(st["opt"]); step = st["step"]
        print(f"resumed at step {step}", flush=True)
    log = open(run / "log.jsonl", "a")
    gen = torch.Generator(device=dev).manual_seed(1234)
    if step == 0:
        r = evaluate(net, codec, sets, torch.Generator(device=dev).manual_seed(99))
        log.write(json.dumps({"step": 0, **r}) + "\n"); log.flush()
        print("step 0", r, flush=True)
    rng = random.Random(step)
    queue = []
    t0, losses = time.time(), []
    started = time.time()
    while step < a.steps:
        if a.max_minutes and (time.time() - started) / 60 > a.max_minutes:
            print(f"stopping at step {step}: --max-minutes {a.max_minutes} reached", flush=True)
            r = evaluate(net, codec, sets, torch.Generator(device=dev).manual_seed(99))
            r.update(step=step, stopped_early=True)
            log.write(json.dumps(r) + "\n"); log.flush()
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step}, ck)
            break
        if not queue:
            queue = batches(data, rng, a.batch)
        bi = queue.pop()
        items = [data[i] for i in bi]
        xs = [ref.normalise(im, False) for im, *_ in items]
        logp, Ls, order = net(xs, generator=gen)
        tg = [items[i][2] for i in order]
        targets = torch.as_tensor([c for t in tg for c in t], dtype=torch.long, device=dev)
        tl = torch.as_tensor([len(t) for t in tg], dtype=torch.long)
        loss = F.ctc_loss(logp, targets, Ls, tl, blank=m.null_char, reduction="mean", zero_infinity=True)
        frac = min(1.0, max(0.0, (step - 100) / max(1, a.steps - 100)))
        lr = a.lr * min(1.0, (step + 1) / 100) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        for g in opt.param_groups:
            g["lr"] = lr
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        step += 1
        losses.append(float(loss))
        if step % 20 == 0:
            print(f"step {step} loss {np.mean(losses[-20:]):.4f} {(time.time() - t0) / 20:.2f}s/step", flush=True)
            t0 = time.time()
        if step % a.eval_every == 0 or step == a.steps:
            r = evaluate(net, codec, sets, torch.Generator(device=dev).manual_seed(99))
            r.update(step=step, loss=round(float(np.mean(losses[-a.eval_every:])), 4))
            log.write(json.dumps(r) + "\n"); log.flush()
            print("eval", r, flush=True)
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "step": step}, ck)


def do_export(a):
    m = load(MODEL)
    net = TessNet(m)
    st = torch.load(FT / a.run / "last.pt", weights_only=False, map_location="cpu")
    net.load_state_dict(st["net"])
    out = FT / a.run / "eng.traineddata"
    export(net, m, out)
    print(f"exported step {st['step']} -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "train", "export"])
    ap.add_argument("--run", default="pilot1")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--eval-every", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--threads", type=int, default=3)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max-minutes", type=float, default=0)
    ap.add_argument("--eval", choices=["val", "local"], default="val")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--data", default="prep.pkl")
    ap.add_argument("--sroie", action="store_true", help="include SROIE training lines (src 'sroie')")
    ap.add_argument("--real-src", nargs="*", default=[], help="real-line sources to train on, e.g. cord")
    ap.add_argument("--extra", nargs="*", default=[], help="extra labels.jsonl files for prep")
    a = ap.parse_args()
    if a.cmd == "prep":
        d = prep(a.extra)
        pickle.dump(d, open(FT / a.data, "wb"))
        print(f"prepared {len(d)} lines -> {FT / a.data}")
    elif a.cmd == "train":
        train(a)
    else:
        do_export(a)
