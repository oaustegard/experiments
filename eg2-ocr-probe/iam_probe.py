"""Handwriting: the same frozen-encoder CTC probe on IAM text lines (Teklia/IAM-line).

Each line image (IAM normalises them to 128 px tall) is resized to 64 px tall keeping its aspect ratio
(4 rows of 16 px patches, width capped at 2048 px) and padded with white to the batch's widest line. The
frozen EmbeddingGemma 2 vision tower encodes every train and test line once; the features of each line's
own columns are cached in fp16. A BiLSTM CTC head (as in probe.py, 3 frames per patch column) then trains
on the cache. Reports test CER on raw text and on text with IAM's spaced punctuation closed up.

Kaggle (no arguments): installs transformers, downloads model and data, writes /kaggle/working/iam_result.json.
`--smoke` runs a few lines on CPU against PROBE_CACHE.
"""
import argparse
import io
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

OUT = Path(os.environ.get("PROBE_OUT", "/kaggle/working"))
CACHE = Path(os.environ.get("PROBE_CACHE", "/kaggle/temp/probe"))
IAM_URL = "https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data/{}.parquet"
H, PS, MAX_W, UP = 64, 16, 2048, 3
ROWS = H // PS


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def setup():
    CACHE.mkdir(parents=True, exist_ok=True)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers==5.23.0"], check=True)
    import urllib.request
    for s in ("train", "test"):
        dst = CACHE / f"iam_{s}.parquet"
        if not dst.exists():
            urllib.request.urlretrieve(IAM_URL.format(s), dst)
    from huggingface_hub import snapshot_download
    snapshot_download("google/embeddinggemma-2", local_dir=str(CACHE / "eg2"), allow_patterns=["*.json", "*.safetensors"])
    log("setup done")


def norm(t):
    """Close up IAM's tokenised spacing so engines that print normal punctuation are not charged for it."""
    t = " ".join(t.split())
    t = re.sub(r" ([,.;:!?)'])", r"\1", t)
    return re.sub(r"([(]) ", r"\1", t)


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


def load_split(split, limit=None):
    import numpy as np
    import pandas as pd
    from PIL import Image
    df = pd.read_parquet(CACHE / f"iam_{split}.parquet")
    if limit:
        df = df.iloc[:limit]
    imgs, texts = [], []
    for _, r in df.iterrows():
        im = Image.open(io.BytesIO(r["image"]["bytes"])).convert("L")
        w = max(PS, round(im.size[0] * H / im.size[1]))
        if w > MAX_W:  # very long lines: shrink to fit, keeping the aspect ratio
            im = im.resize((MAX_W, max(PS, round(im.size[1] * MAX_W / im.size[0]))), Image.BICUBIC)
            canvas = Image.new("L", (MAX_W, H), 255)
            canvas.paste(im, (0, (H - im.size[1]) // 2))
            im = canvas
        else:
            im = im.resize((w, H), Image.BICUBIC)
        imgs.append(np.asarray(im, dtype=np.uint8))
        texts.append(r["text"])
    return imgs, texts


def encode_all(tower, imgs, device, bs=32):
    """Returns per line a (cols, ROWS*768) fp16 CPU tensor: its own columns only."""
    import numpy as np
    import torch
    order = sorted(range(len(imgs)), key=lambda i: imgs[i].shape[1])
    feats = [None] * len(imgs)
    t0 = time.time()
    for b0 in range(0, len(order), bs):
        idx = order[b0:b0 + bs]
        cols = [math.ceil(imgs[i].shape[1] / PS) for i in idx]
        C = max(cols)
        batch = np.full((len(idx), H, C * PS), 255, np.uint8)
        for j, i in enumerate(idx):
            batch[j, :, :imgs[i].shape[1]] = imgs[i]
        x = torch.from_numpy(batch).to(device).float().div(255.0).unsqueeze(1).expand(-1, 3, -1, -1)
        n = x.shape[0]
        pv = x.reshape(n, 3, ROWS, PS, C, PS).permute(0, 2, 4, 3, 5, 1).reshape(n, ROWS * C, PS * PS * 3)
        yy, xx = torch.meshgrid(torch.arange(ROWS), torch.arange(C), indexing="ij")
        pos = torch.stack([xx, yy], -1).reshape(1, -1, 2).expand(n, -1, -1).to(device)
        pad = torch.zeros(pv.shape[:2], dtype=torch.bool, device=device)
        with torch.no_grad():
            h = tower.encoder(inputs_embeds=tower.patch_embedder(pv, pos, pad), attention_mask=~pad,
                              pixel_position_ids=pos).last_hidden_state
        h = h.reshape(n, ROWS, C, -1).permute(0, 2, 1, 3).reshape(n, C, -1).half().cpu()
        for j, i in enumerate(idx):
            feats[i] = h[j, :cols[j]].clone()
        if (b0 // bs) % 20 == 0:
            log(f"encoded {b0 + len(idx)}/{len(imgs)} lines, {time.time() - t0:.0f}s")
    return feats


def run(device, steps, bs, train_limit=None, test_limit=None):
    import torch
    t_all = time.time()
    tr_imgs, tr_texts = load_split("train", train_limit)
    te_imgs, te_texts = load_split("test", test_limit)
    charset = sorted(set("".join(tr_texts)))
    c2i = {c: i + 1 for i, c in enumerate(charset)}
    log(f"train {len(tr_imgs)} test {len(te_imgs)} lines, {len(charset)} characters; "
        f"median width {sorted(a.shape[1] for a in tr_imgs)[len(tr_imgs) // 2]} px")
    from transformers.models.embedding_gemma2.modeling_embedding_gemma2 import (
        EmbeddingGemma2Model,
    )
    m = EmbeddingGemma2Model.from_pretrained(str(CACHE / "eg2"), dtype=torch.float32, audio_config=None)
    tower = m.vision_tower.to(device).eval().requires_grad_(False)
    del m
    tr_f = encode_all(tower, tr_imgs, device)
    te_f = encode_all(tower, te_imgs, device)
    del tower
    t_enc = time.time() - t_all
    log(f"encoding done in {t_enc:.0f}s")

    from torch import nn

    class Head(nn.Module):
        def __init__(self, d, ncls):
            super().__init__()
            self.norm, self.proj = nn.LayerNorm(d), nn.Linear(d, 512)
            self.drop = nn.Dropout(0.2)
            self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True, batch_first=True, dropout=0.2)
            self.out = nn.Linear(512, ncls * UP)

        def forward(self, x):
            h, _ = self.rnn(self.drop(self.proj(self.norm(x))))
            return self.out(h).reshape(x.shape[0], x.shape[1] * UP, -1)

    head = Head(tr_f[0].shape[1], len(charset) + 1).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=0.01)
    warm = max(1, steps // 15)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    ctc = nn.CTCLoss(blank=0, zero_infinity=True)

    def pad_batch(fs):
        L = max(f.shape[0] for f in fs)
        x = torch.zeros(len(fs), L, fs[0].shape[1], dtype=torch.float32)
        for j, f in enumerate(fs):
            x[j, :f.shape[0]] = f.float()
        return x.to(device), torch.tensor([f.shape[0] * UP for f in fs])

    def decode(fs):
        head.eval()
        preds = []
        with torch.no_grad():
            for b0 in range(0, len(fs), 64):
                x, lens = pad_batch(fs[b0:b0 + 64])
                ids = head(x).argmax(-1).cpu().tolist()
                for seq, n in zip(ids, lens.tolist()):
                    prev, s = 0, []
                    for i in seq[:n]:
                        if i != prev and i != 0:
                            s.append(charset[i - 1])
                        prev = i
                    preds.append("".join(s))
        head.train()
        return preds

    rng = random.Random(0)
    losses, curve = [], []
    t0 = time.time()
    for step in range(1, steps + 1):
        idx = rng.sample(range(len(tr_f)), min(bs, len(tr_f)))
        x, ilen = pad_batch([tr_f[i] for i in idx])
        tgt = torch.tensor([c2i[c] for i in idx for c in tr_texts[i]], dtype=torch.long)
        tlen = torch.tensor([len(tr_texts[i]) for i in idx])
        lp = head(x).log_softmax(-1).transpose(0, 1)
        loss = ctc(lp, tgt.to(device), ilen.to(device), tlen.to(device))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        opt.step()
        sched.step()
        losses.append(loss.item())
        if step % 100 == 0:
            log(f"step {step} loss {sum(losses[-100:]) / 100:.3f} {(time.time() - t0) / step:.3f}s/step")
        if step % max(1, steps // 8) == 0 or step == steps:
            p = decode(te_f[:500])
            point = {"step": step, "test500_cer": round(cer(p, te_texts[:500]), 4)}
            curve.append(point)
            log("eval", point)
    p_te = decode(te_f)
    p_tr = decode(tr_f[:500])
    res = {
        "lines": {"train": len(tr_f), "test": len(te_f)}, "steps": steps, "batch": bs, "height_px": H,
        "test_cer_raw": cer(p_te, te_texts),
        "test_cer_norm": cer([norm(x) for x in p_te], [norm(x) for x in te_texts]),
        "test_exact_norm": sum(norm(a) == norm(b) for a, b in zip(p_te, te_texts)) / len(te_texts),
        "train500_cer_raw": cer(p_tr, tr_texts[:500]),
        "curve": curve, "encode_seconds": round(t_enc), "train_seconds": round(time.time() - t0),
        "samples": [[r, q] for r, q in zip(te_texts[:15], p_te[:15])],
        "test_preds": p_te,
    }
    log({k: v for k, v in res.items() if k not in ("samples", "test_preds", "curve")})
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "iam_result.json").write_text(json.dumps(res, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--steps", type=int, default=2000)
    a = ap.parse_args()
    if a.smoke:
        run("cpu", steps=8, bs=4, train_limit=16, test_limit=8)
        return
    setup()
    import torch
    log("torch", torch.__version__, "cuda", torch.cuda.is_available())
    run("cuda:0" if torch.cuda.is_available() else "cpu", a.steps, 32)


if __name__ == "__main__":
    main()
