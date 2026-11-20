"""Does EmbeddingGemma 2's frozen vision encoder keep enough glyph detail to read text?

Line-level CTC probe. Synthetic printed text lines (10 Google Fonts, wikitext spans, mild
degradation) are rendered at 48 px tall, patchified exactly as Gemma4ImageProcessor does
(16 px patches, no resize), and run through the frozen vision tower. A small trainable head
(LayerNorm, Linear 512, 2-layer BiLSTM 256, Linear to CTC frames) reads the last hidden
state two ways:

  prepool  every 16 px patch column (3 rows concatenated, 2304 dims), 3 CTC frames per column
  pooled   the 3x3-averaged 48 px soft tokens a language model would see, 9 frames per token

Control: the same architecture with random weights (prepool head only), so the head's own
capacity is visible. Character error rate (CER) on unseen wikitext spans in seen fonts, and on
one font never seen in training.

Run with no arguments on Kaggle (2x T4): it installs transformers, downloads the model, fonts
and text, then trains the pretrained variant on cuda:0 and the random one on cuda:1 at once.
`--smoke` runs a few CPU steps locally.
"""
import argparse
import json
import math
import os
import random
import re
import subprocess
import sys
import time
import unicodedata
from pathlib import Path

OUT = Path(os.environ.get("PROBE_OUT", "/kaggle/working"))
CACHE = Path(os.environ.get("PROBE_CACHE", "/kaggle/temp/probe"))
MODEL_ID = "google/embeddinggemma-2"
FONT_BASE = "https://raw.githubusercontent.com/google/fonts/main/"
TRAIN_FONTS = [
    "ofl/notoserif/NotoSerif%5Bwdth,wght%5D.ttf",
    "ofl/sourcecodepro/SourceCodePro%5Bwght%5D.ttf",
    "ofl/merriweather/Merriweather%5Bopsz,wdth,wght%5D.ttf",
    "ofl/lato/Lato-Regular.ttf",
    "ofl/opensans/OpenSans%5Bwdth,wght%5D.ttf",
    "ofl/ebgaramond/EBGaramond%5Bwght%5D.ttf",
    "ofl/courierprime/CourierPrime-Regular.ttf",
    "apache/robotoslab/RobotoSlab%5Bwght%5D.ttf",
    "ofl/ptsans/PT_Sans-Web-Regular.ttf",
]
HELDOUT_FONT = "ofl/crimsontext/CrimsonText-Regular.ttf"
TEXT_URL = "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-103-raw-v1/{}-00000-of-00001.parquet"

H, W, PS = 48, 1152, 16  # line image; 3 x 72 patches, 24 pooled soft tokens
COLS, ROWS = W // PS, H // PS
CHARSET = [chr(c) for c in range(32, 127)]
C2I = {c: i + 1 for i, c in enumerate(CHARSET)}  # 0 = CTC blank
MAX_CHARS = 60


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


# ---------------------------------------------------------------- setup (Kaggle)

def setup():
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.23.0"], check=True)
    import urllib.request
    fdir = CACHE / "fonts"
    fdir.mkdir(exist_ok=True)
    for p in TRAIN_FONTS + [HELDOUT_FONT]:
        dst = fdir / Path(p).name
        if not dst.exists():
            urllib.request.urlretrieve(FONT_BASE + p, dst)
    for split in ("validation", "test"):
        dst = CACHE / f"{split}.parquet"
        if not dst.exists():
            urllib.request.urlretrieve(TEXT_URL.format(split), dst)
    from huggingface_hub import snapshot_download
    snapshot_download(MODEL_ID, local_dir=str(CACHE / "eg2"),
                      allow_patterns=["*.json", "*.safetensors"])


# ---------------------------------------------------------------- data

def clean(s):
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = re.sub(r" ([,.;:!?)\]'])", r"\1", s)
    s = re.sub(r"([(\[]) ", r"\1", s)
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if c in C2I)
    return re.sub(r"\s+", " ", s).strip()


def load_lines(split):
    import pandas as pd
    df = pd.read_parquet(CACHE / f"{split}.parquet")
    lines = [clean(t) for t in df["text"] if len(t) > 80 and not t.strip().startswith("=")]
    return [l for l in lines if len(l) > 40]


def font_paths(held_out):
    fdir = CACHE / "fonts"
    return [str(fdir / Path(HELDOUT_FONT).name)] if held_out else [str(fdir / Path(p).name) for p in TRAIN_FONTS]


def render(text, font_path, rng):
    """Returns (uint8 HxW image, text actually drawn)."""
    from PIL import Image, ImageDraw, ImageFilter, ImageFont
    size = rng.randint(22, 30)
    font = ImageFont.truetype(font_path, size)
    x0 = rng.randint(2, 24)
    while text and (font.getlength(text) > W - x0 - 8 or len(text) > MAX_CHARS):
        text = text[:-1]
    text = text.rstrip()
    bg, fg = rng.randint(200, 255), rng.randint(0, 70)
    img = Image.new("L", (W, H), bg)
    d = ImageDraw.Draw(img)
    asc, desc = font.getmetrics()
    y0 = (H - (asc + desc)) // 2 + rng.randint(-3, 3)
    d.text((x0, y0), text, font=font, fill=fg)
    if rng.random() < 0.3:
        img = img.rotate(rng.uniform(-1.0, 1.0), resample=Image.BILINEAR, fillcolor=bg)
    if rng.random() < 0.5:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.9)))
    import numpy as np
    a = np.asarray(img, dtype=np.float32)
    if rng.random() < 0.5:
        a = a + np.random.default_rng(rng.randint(0, 2**31)).normal(0, rng.uniform(2, 10), a.shape)
    return np.clip(a, 0, 255).astype(np.uint8), text


def sample_text(lines, rng):
    l = lines[rng.randrange(len(lines))]
    n = rng.randint(15, MAX_CHARS)
    starts = [0] + [m.end() for m in re.finditer(" ", l)]
    s = rng.choice([st for st in starts if st < len(l) - 10] or [0])
    return l[s:s + n].strip() or l[:n]


def make_batch(lines, fonts, rng, bs):
    import numpy as np
    imgs, texts = [], []
    while len(imgs) < bs:
        a, t = render(sample_text(lines, rng), rng.choice(fonts), rng)
        if len(t) >= 5:
            imgs.append(a)
            texts.append(t)
    return np.stack(imgs), texts


def make_eval_set(split, held_out, n, seed):
    rng = random.Random(seed)
    lines = load_lines(split)
    imgs, texts = make_batch(lines, font_paths(held_out), rng, n)
    return imgs, texts


# ---------------------------------------------------------------- model

def patchify(imgs, device):
    """uint8 (B,H,W) -> pixel_values (B, ROWS*COLS, 768) in [0,1], positions (B, N, 2) as (x, y).
    Matches transformers' Gemma4 convert_image_to_patches (row-major patches, (ps, ps, C) inside)."""
    import torch
    x = torch.from_numpy(imgs).to(device).float().div(255.0)
    x = x.unsqueeze(1).expand(-1, 3, -1, -1)  # grey -> RGB
    b = x.shape[0]
    x = x.reshape(b, 3, ROWS, PS, COLS, PS).permute(0, 2, 4, 3, 5, 1).reshape(b, ROWS * COLS, PS * PS * 3)
    yy, xx = torch.meshgrid(torch.arange(ROWS), torch.arange(COLS), indexing="ij")
    pos = torch.stack([xx, yy], -1).reshape(1, -1, 2).expand(b, -1, -1).to(device)
    return x, pos


def load_tower(variant, device):
    import torch
    from transformers import AutoConfig, AutoModel
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2Model,
    )
    path = str(CACHE / "eg2")
    if variant == "pretrained":
        m = EmbeddingGemma2Model.from_pretrained(path, dtype=torch.float32, audio_config=None)
        tower = m.vision_tower
        del m
    else:
        cfg = AutoConfig.from_pretrained(path)
        torch.manual_seed(0)
        tower = AutoModel.from_config(cfg.vision_config)
    return tower.to(device).eval().requires_grad_(False)


def encode(tower, imgs, device):
    """Frozen last hidden state before pooling: (B, ROWS, COLS, 768)."""
    import torch
    pv, pos = patchify(imgs, device)
    pad = torch.zeros(pv.shape[:2], dtype=torch.bool, device=device)
    with torch.no_grad():
        emb = tower.patch_embedder(pv, pos, pad)
        h = tower.encoder(inputs_embeds=emb, attention_mask=~pad, pixel_position_ids=pos).last_hidden_state
    return h.float().reshape(pv.shape[0], ROWS, COLS, -1)


def views(h):
    b, r, c, d = h.shape
    prepool = h.permute(0, 2, 1, 3).reshape(b, c, r * d)                       # (B, 72, 2304)
    pooled = h.reshape(b, r, c // 3, 3, d).mean(dim=(1, 3))                    # (B, 24, 768)
    return {"prepool": (prepool, 3), "pooled": (pooled, 9)}


def make_head(in_dim, up):
    from torch import nn

    class Head(nn.Module):
        def __init__(self):
            super().__init__()
            self.norm = nn.LayerNorm(in_dim)
            self.proj = nn.Linear(in_dim, 512)
            self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1)
            self.out = nn.Linear(512, (len(CHARSET) + 1) * up)
            self.up = up

        def forward(self, x):
            h, _ = self.rnn(self.proj(self.norm(x)))
            o = self.out(h)
            return o.reshape(x.shape[0], x.shape[1] * self.up, -1)  # (B, T, classes)

    return Head()


def greedy(logits):
    ids = logits.argmax(-1).tolist()
    outs = []
    for seq in ids:
        prev, s = 0, []
        for i in seq:
            if i != prev and i != 0:
                s.append(CHARSET[i - 1])
            prev = i
        outs.append("".join(s))
    return outs


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(preds, refs):
    return sum(lev(p, r) for p, r in zip(preds, refs)) / max(1, sum(len(r) for r in refs))


def evaluate(tower, heads, imgs, texts, device, bs=128):
    import torch
    preds = {k: [] for k in heads}
    for h in heads.values():
        h.eval()
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            v = views(encode(tower, imgs[i:i + bs], device))
            for k, h in heads.items():
                preds[k] += greedy(h(v[k][0]))
    for h in heads.values():
        h.train()
    return {k: {"cer": cer(p, texts), "exact": sum(a == b for a, b in zip(p, texts)) / len(texts),
                "samples": [[r, p_] for r, p_ in zip(texts[:12], p[:12])]} for k, p in preds.items()}


# ---------------------------------------------------------------- train

def run(variant, device, steps, bs, n_eval, smoke=False):
    import torch
    torch.manual_seed(0)
    t0 = time.time()
    tower = load_tower(variant, device)
    log(variant, "tower loaded", sum(p.numel() for p in tower.parameters()) / 1e6, "M params")
    want = ["prepool", "pooled"] if variant == "pretrained" else ["prepool"]
    dims = {"prepool": (ROWS * 768, 3), "pooled": (768, 9)}
    heads = {k: make_head(*dims[k]).to(device) for k in want}
    params = [p for h in heads.values() for p in h.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-3, weight_decay=0.01)
    warm = max(1, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ctc = torch.nn.CTCLoss(blank=0, zero_infinity=True)

    train_lines = load_lines("validation")
    fonts = font_paths(False)
    rng = random.Random(1 if variant == "pretrained" else 2)
    ev_seen = make_eval_set("test", False, n_eval, seed=10)
    ev_held = make_eval_set("test", True, n_eval // 2, seed=11)
    curve, losses = [], []
    res_path = OUT / f"result_{variant}.json"

    # Queue batches from a background thread so PIL rendering overlaps the GPU.
    import queue
    import threading
    q = queue.Queue(maxsize=8)

    def producer(seed):
        r = random.Random(seed)
        while True:
            q.put(make_batch(train_lines, fonts, r, bs))

    for k in range(2 if not smoke else 1):
        threading.Thread(target=producer, args=(rng.randint(0, 2**31) + k,), daemon=True).start()

    for step in range(1, steps + 1):
        imgs, texts = q.get()
        v = views(encode(tower, imgs, device))
        tgt = torch.tensor([C2I[c] for t in texts for c in t], dtype=torch.long)
        tlen = torch.tensor([len(t) for t in texts], dtype=torch.long)
        loss = 0.0
        for k, h in heads.items():
            lp = h(v[k][0]).log_softmax(-1).transpose(0, 1)  # (T, B, C)
            ilen = torch.full((lp.shape[1],), lp.shape[0], dtype=torch.long)
            loss = loss + ctc(lp.cpu() if lp.device.type == "mps" else lp, tgt.to(lp.device), ilen, tlen)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(float(loss))
        if step % 100 == 0 or smoke:
            log(variant, f"step {step} loss {sum(losses[-100:]) / len(losses[-100:]):.3f} "
                f"{(time.time() - t0) / step:.2f}s/step")
        if step % max(1, steps // 6) == 0 or step == steps:
            n = len(ev_seen[0]) if step == steps else min(400, len(ev_seen[0]))
            r = evaluate(tower, heads, ev_seen[0][:n], ev_seen[1][:n], device)
            point = {"step": step, "n": n, **{k: round(x["cer"], 4) for k, x in r.items()}}
            curve.append(point)
            log(variant, "eval", point)
            out = {"variant": variant, "device": str(device), "steps_done": step, "steps": steps,
                   "batch": bs, "curve": curve, "seen_fonts": r, "seconds": round(time.time() - t0)}
            if step == steps:
                out["heldout_font"] = evaluate(tower, heads, *ev_held, device)
                log(variant, "held-out font", {k: round(x["cer"], 4) for k, x in out["heldout_font"].items()})
            res_path.write_text(json.dumps(out, indent=1))
    return res_path


def verify_patchify():
    """My patchify + pool must match transformers' processor + vision tower exactly."""
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor
    rng = random.Random(0)
    a, _ = render("The quick brown fox, 1234.", font_paths(False)[0], rng)
    tower = load_tower("pretrained", "cpu")
    ip = AutoImageProcessor.from_pretrained(str(CACHE / "eg2"))
    pp = ip(images=[Image.fromarray(a).convert("RGB")], do_resize=False, max_soft_tokens=280, return_tensors="pt")
    with torch.no_grad():
        ref = tower(pixel_values=pp["pixel_values"], pixel_position_ids=pp["image_position_ids"]).last_hidden_state
    h = encode(tower, a[None], "cpu")
    mine = views(h)["pooled"][0][0] * math.sqrt(768)
    diff = (ref[: mine.shape[0]] - mine).abs().max().item()
    log("patchify check: max abs diff vs transformers path", diff, "ref norm", ref.norm().item())
    assert diff < 1e-3 * max(1.0, ref.abs().max().item()), diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-setup", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.smoke:
        verify_patchify()
        for v in ("pretrained", "random"):
            run(v, "cpu", steps=6, bs=8, n_eval=16, smoke=True)
        return
    if a.variant:
        run(a.variant, a.device, a.steps, a.bs, a.n_eval)
        return
    if not a.skip_setup:
        setup()
    import torch
    log("torch", torch.__version__, "cuda", torch.cuda.is_available(), torch.cuda.device_count())
    verify_patchify()
    ngpu = torch.cuda.device_count()
    import threading

    def relay(proc, name):
        # Workers print to the parent's stdout, prefixed, so Kaggle's live log stream
        # (kaggle_job.py logs) shows their progress; the per-worker file is a copy.
        with open(OUT / f"log_{name}.txt", "w") as f:
            for line in proc.stdout:
                f.write(line)
                f.flush()
                print(f"[{name}] {line}", end="", flush=True)

    procs, relays = [], []
    for i, v in enumerate(["pretrained", "random"]):
        dev = f"cuda:{i % max(1, ngpu)}" if ngpu else "cpu"
        p = subprocess.Popen([sys.executable, "-u", __file__, "--variant", v, "--device", dev,
                              "--steps", str(a.steps), "--bs", str(a.bs), "--n-eval", str(a.n_eval)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        procs.append(p)
        relays.append(threading.Thread(target=relay, args=(p, v)))
        relays[-1].start()
    codes = [p.wait() for p in procs]
    for t in relays:
        t.join()
    log("exit codes", codes)


if __name__ == "__main__":
    main()
