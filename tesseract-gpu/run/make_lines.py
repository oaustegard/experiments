#!/usr/bin/env python3
"""Build the synthetic printed-text line benchmark.

Base lines are real sentences (eng, fra: Wikipedia via the Hugging Face
datasets server; grc: Homer's Iliad from Perseus canonical-greekLit). Each
base line is rendered with a random system font that covers every character,
then written under eight degradation conditions:

    clean, blur, noise, jpeg, lowres, rotate, shade, invert

Outputs (under data/lines/):
    <lang>/<id>_<cond>.png     8-bit greyscale line images
    labels.jsonl               one row per image: id, file, lang, cond, text, font, size
    _corpus/                   cached raw corpus responses (regenerable)

Usage:  python3 run/make_lines.py [--out DIR]
"""

import argparse
import hashlib
import io
import json
import random
import re
import subprocess
import sys
import time
import unicodedata
import urllib.request
from pathlib import Path

import numpy as np
from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "data" / "lines"
SEED = 20261119
COUNTS = {"eng": 300, "fra": 150, "grc": 150}
CONDS = ["clean", "blur", "noise", "jpeg", "lowres", "rotate", "shade", "invert"]
LEN_MIN, LEN_MAX = 20, 70

HF_ROWS = "https://datasets-server.huggingface.co/rows"
WIKI_CONFIG = {"eng": "20231101.en", "fra": "20231101.fr"}
WIKI_ROWS = {"eng": 6_400_000, "fra": 2_600_000}  # approximate split sizes
WIKI_REQUESTS = 40          # random 20-row windows per language
WIKI_ROWS_PER_REQ = 20
PERSEUS_URL = ("https://raw.githubusercontent.com/PerseusDL/canonical-greekLit/"
               "master/data/tlg0012/tlg001/tlg0012.tlg001.perseus-grc2.xml")

SOURCES = {
    "eng": {"url": "https://huggingface.co/datasets/wikimedia/wikipedia (config 20231101.en)",
            "license": "CC BY-SA 3.0 and GFDL (Wikipedia; dataset card)"},
    "fra": {"url": "https://huggingface.co/datasets/wikimedia/wikipedia (config 20231101.fr)",
            "license": "CC BY-SA 3.0 and GFDL (Wikipedia; dataset card)"},
    "grc": {"url": PERSEUS_URL,
            "license": "CC BY-SA 4.0 (Perseus canonical-greekLit README)"},
}


def fetch(url, cache: Path, timeout=120) -> bytes:
    if cache.exists():
        return cache.read_bytes()
    cache.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "benchmark-builder/1.0"})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                data = r.read()
            break
        except OSError as e:  # 5xx and resets are transient on these endpoints
            if attempt == 4:
                raise
            print(f"retrying {url[:60]}: {e}", file=sys.stderr)
            time.sleep(2 ** attempt)
    cache.write_bytes(data)
    return data


def wiki_paragraphs(lang: str, corpus: Path):
    """Paragraphs from random Wikipedia articles (cached per request)."""
    rng = random.Random(f"{SEED}:{lang}:offsets")
    starts = sorted(rng.randrange(0, WIKI_ROWS[lang] - WIKI_ROWS_PER_REQ)
                    for _ in range(WIKI_REQUESTS))
    for off in starts:
        url = (f"{HF_ROWS}?dataset=wikimedia%2Fwikipedia&config={WIKI_CONFIG[lang]}"
               f"&split=train&offset={off}&length={WIKI_ROWS_PER_REQ}")
        raw = fetch(url, corpus / lang / f"rows_{off}.json")
        for row in json.loads(raw)["rows"]:
            for para in row["row"]["text"].split("\n"):
                yield para


SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[\"«“A-ZÀ-Ý])")
# printable Latin text with French punctuation; no digits, markup or brackets
LATIN_OK = re.compile(r"^[A-Za-zÀ-ÿ ,.;:!?'’\"«»()\-–]+$")


def latin_sentences(paragraphs):
    out, seen = [], set()
    for para in paragraphs:
        for s in SENT_SPLIT.split(para.strip()):
            s = " ".join(s.split())
            if not (LEN_MIN <= len(s) <= LEN_MAX):
                continue
            if not s[-1] in ".!?":
                continue
            if not LATIN_OK.match(s) or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def iliad_lines(corpus: Path):
    raw = fetch(PERSEUS_URL, corpus / "grc" / "tlg0012.tlg001.perseus-grc2.xml").decode("utf8")
    out, seen = [], set()
    for m in re.finditer(r"<l\b[^>]*>(.*?)</l>", raw, flags=re.S):
        s = re.sub(r"<[^>]+>", "", m.group(1))
        s = unicodedata.normalize("NFC", " ".join(s.split()))
        if LEN_MIN <= len(s) <= LEN_MAX and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def base_lines(lang: str, corpus: Path):
    if lang == "grc":
        pool = iliad_lines(corpus)
    else:
        pool = latin_sentences(wiki_paragraphs(lang, corpus))
    pool = sorted(pool)  # stable order before the seeded sample
    need = COUNTS[lang]
    if len(pool) < need:
        sys.exit(f"{lang}: only {len(pool)} candidate lines, need {need}")
    return random.Random(f"{SEED}:{lang}:sample").sample(pool, need)


def font_index():
    """Return [(path, cmap_keys)] for every usable outline font."""
    files = subprocess.run(["fc-list", "--format", "%{file}\n"], capture_output=True,
                           text=True, check=True).stdout.split("\n")
    fonts = []
    for f in sorted({f for f in files if f.endswith((".ttf", ".otf"))}):
        try:
            cmap = TTFont(f, lazy=True).getBestCmap() or {}
        except Exception:
            continue
        if cmap:
            fonts.append((f, frozenset(cmap.keys())))
    return fonts


def covering_fonts(fonts, text):
    need = {ord(c) for c in text}
    return [f for f, cm in fonts if need <= cm]


def render(text, font_path, size, rng):
    font = ImageFont.truetype(font_path, size)
    pad = rng.randint(4, 12)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    bg = rng.randint(235, 255)
    ink = rng.randint(0, 60)
    img = Image.new("L", (x1 - x0 + 2 * pad, y1 - y0 + 2 * pad), bg)
    ImageDraw.Draw(img).text((pad - x0, pad - y0), text, font=font, fill=ink)
    return img, font


def xheight(font):
    x0, y0, x1, y1 = font.getbbox("x")
    return max(1, y1 - y0)


def conditions(clean: Image.Image, font, rng, nrng):
    out = {"clean": clean}
    out["blur"] = clean.filter(ImageFilter.GaussianBlur(radius=rng.uniform(0.6, 1.2)))

    a = np.asarray(clean, dtype=np.float32)
    sd = rng.uniform(12, 25)
    out["noise"] = Image.fromarray(np.clip(a + nrng.normal(0, sd, a.shape), 0, 255).astype(np.uint8), "L")

    buf = io.BytesIO()
    clean.save(buf, format="JPEG", quality=rng.randint(15, 30))
    buf.seek(0)
    out["jpeg"] = Image.open(buf).convert("L")

    target = rng.uniform(6, 8)
    scale = target / xheight(font)
    w, h = clean.size
    out["lowres"] = clean.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)

    angle = rng.uniform(-2, 2)
    out["rotate"] = clean.rotate(angle, resample=Image.BICUBIC, expand=True,
                                 fillcolor=int(np.asarray(clean)[0, 0]))

    # linear illumination gradient, horizontal or vertical, random direction
    lo, hi = rng.uniform(0.45, 0.7), 1.0
    horizontal = rng.random() < 0.5
    n = a.shape[1] if horizontal else a.shape[0]
    ramp = np.linspace(lo, hi, n, dtype=np.float32)
    if rng.random() < 0.5:
        ramp = ramp[::-1]
    illum = np.broadcast_to(ramp[None, :], a.shape) if horizontal else np.broadcast_to(ramp[:, None], a.shape)
    s = a * illum
    m = s.mean()
    out["shade"] = Image.fromarray(np.clip(m + 0.4 * (s - m), 0, 255).astype(np.uint8), "L")

    out["invert"] = Image.fromarray(255 - a.astype(np.uint8), "L")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()
    out: Path = args.out
    corpus = out / "_corpus"

    fonts = font_index()
    print(f"usable outline fonts: {len(fonts)}", file=sys.stderr)

    rows = []
    for lang in ["eng", "fra", "grc"]:
        lines = base_lines(lang, corpus)
        print(f"{lang}: {len(lines)} base lines sampled", file=sys.stderr)
        (out / lang).mkdir(parents=True, exist_ok=True)
        for i, text in enumerate(lines, 1):
            uid = f"{lang}{i:04d}"
            rng = random.Random(f"{SEED}:{uid}")
            nrng = np.random.default_rng(int(hashlib.sha256(f"{SEED}:{uid}".encode()).hexdigest()[:8], 16))
            cands = covering_fonts(fonts, text)
            if not cands:
                sys.exit(f"{uid}: no system font covers all characters of {text!r}")
            font_path = rng.choice(cands)
            size = rng.randint(18, 40)
            clean, font = render(text, font_path, size, rng)
            imgs = conditions(clean, font, rng, nrng)
            for cond in CONDS:
                rel = f"{lang}/{uid}_{cond}.png"
                imgs[cond].save(out / rel, format="PNG")
                rows.append({"id": uid, "file": rel, "lang": lang, "cond": cond, "text": text,
                             "font": Path(font_path).name, "size": size})
    with (out / "labels.jsonl").open("w", encoding="utf8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} rows to {out / 'labels.jsonl'}", file=sys.stderr)


if __name__ == "__main__":
    main()
