"""Training lines for fine-tuning Tesseract's eng model: rendered text with document-style damage.

    python3 run/ft/make_train.py --n 24000 --jobs 3      # -> data/ft/lines/*.png + data/ft/labels.jsonl

Text, all within eng's unicharset:
  wiki  sentences and fragments of WikiText-103 (validation and test splits, Wikipedia text, CC BY-SA), digits
        and punctuation allowed; any sentence in data/lines/labels.jsonl or data/real/labels.jsonl is excluded.
  doc   generated receipt and form strings: item lines with prices, totals, dates and times, phone and fax
        numbers, invoice and tax numbers, addresses, form labels; words drawn from the wiki vocabulary.
Rendering: every installed outline font that covers the text (serif, sans, mono, typewriter, bitmap-style),
14 to 48 px, random ink and paper levels, random padding.
Damage (independent draws, about one line in five left clean): dot-matrix dots, ink bleed or thinning, blur,
noise, JPEG, low resolution, small rotation, lighting gradient with contrast loss, fading, bitonal thresholding,
thermal-paper streaks, form rules (underline or box), fragments of neighbouring lines at the top or bottom edge,
speckle, inversion.

Nothing here is drawn from FUNSD or SROIE; those stay evaluation data (SROIE training lines, if used, come from
run/ft/sroie_train.py with its own document split).
"""
import argparse
import io
import json
import random
import re
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "run"))
from make_lines import covering_fonts  # noqa: E402


def font_index():
    """[(path, cmap keys)] for every outline font under the system font dirs and $FT_FONT_DIR (fc-list is not
    needed, so this also runs on a Kaggle machine)."""
    import glob
    import os
    from fontTools.ttLib import TTFont
    dirs = ["/usr/share/fonts", "/usr/local/share/fonts", os.path.expanduser("~/.fonts"), os.environ.get("FT_FONT_DIR", "")]
    files = sorted({f for d in dirs if d for f in glob.glob(f"{d}/**/*", recursive=True) if f.lower().endswith((".ttf", ".otf"))})
    fonts = []
    for f in files:
        try:
            cmap = TTFont(f, lazy=True, fontNumber=0).getBestCmap() or {}
        except Exception:
            continue
        if cmap:
            fonts.append((f, frozenset(cmap.keys())))
    return fonts

OUT = ROOT / "data/ft"
SEED = "ft-20261120"
CHARSET = set(' CHES-RIKNGB85F,(/LT)OY.DAMUP[]970142W3<>"VX\'~!JQZ+@&’=_€™“|?:6{}$;\\—”*#»®%£«°©§¥¢‘'
              'inculatemosghbzvqfrwpdkxyjé')
SKIP_FONTS = ("NotoColorEmoji", "OpenSymbol", "unifont_upper", "unifont_csur", "Unifont-JP")


# ------------------------------------------------------------------ text

def wiki_text():
    """Paragraphs of English Wikipedia text: WikiText-103 (Salesforce/wikitext, CC BY-SA), validation and test
    splits, with its tokenisation (' @-@ ', ' , ', ...) undone. The rows endpoint the evaluation set came from
    answered 503 on 2026-11-20, and this also keeps training text away from the evaluation articles."""
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    paras = []
    for split in ("validation", "test"):
        path = hf_hub_download("Salesforce/wikitext", f"wikitext-103-raw-v1/{split}-00000-of-00001.parquet",
                               repo_type="dataset", cache_dir=str(OUT / "_corpus"))
        for t in pq.read_table(path).column("text").to_pylist():
            t = t.strip()
            if len(t) < 40 or t.startswith("="):
                continue
            paras.append(detok(t))
    return paras


def detok(t):
    for a, b in ((" @-@ ", "-"), (" @,@ ", ","), (" @.@ ", "."), (" ,", ","), (" .", "."), (" ;", ";"), (" :", ":"),
                 (" 's", "'s"), (" n't", "n't"), ("( ", "("), (" )", ")"), (" ?", "?"), (" !", "!"), (" %", "%"),
                 ("$ ", "$"), (" 're", "'re"), (" 've", "'ve"), (" 'd", "'d"), (" 'll", "'ll"), (" 'm", "'m")):
        t = t.replace(a, b)
    t = re.sub(r'" (.*?) "', r'"\1"', t)
    return t


class Excluded:
    """Membership by SHA-1 of the whitespace-normalised text: run/ft/eval_text_sha1.txt holds the hashes of every
    evaluation text (data/lines and data/real), so the evaluation transcriptions need not ship with the code."""
    def __init__(self):
        import hashlib
        self._h = hashlib.sha1
        self.hashes = set((Path(__file__).parent / "eval_text_sha1.txt").read_text().split())

    def __contains__(self, t):
        return self._h(" ".join(t.split()).encode()).hexdigest() in self.hashes


def norm_chars(s):
    s = s.replace("–", "-").replace("…", "...").replace(" ", " ")
    return " ".join(s.split())


def wiki_lines(paras, exclude, rng):
    out = []
    for p in paras:
        for sent in re.split(r"(?<=[.!?])\s+", norm_chars(p)):
            if not sent or sent in exclude or not set(sent) <= CHARSET:
                continue
            if len(sent) <= 70:
                if len(sent) >= 6:
                    out.append(sent)
                continue
            words = sent.split()
            i = 0
            while i < len(words):  # fragments of 2..12 words, like lines cut from a column
                k = rng.randint(2, 12)
                frag = " ".join(words[i:i + k])
                if 6 <= len(frag) <= 70:
                    out.append(frag)
                i += k
    return out


def doc_line(rng, vocab):
    W = lambda: rng.choice(vocab)
    up = lambda s: s.upper() if rng.random() < 0.5 else s
    money = lambda: f"{rng.randint(0, 999)}.{rng.randint(0, 99):02d}" if rng.random() < 0.8 else f"{rng.randint(1, 99)},{rng.randint(0, 999):03d}.{rng.randint(0, 99):02d}"
    cur = lambda: rng.choice(["", "", "$", "RM ", "RM", "£", "€", "USD ", "S$"])
    date = lambda: rng.choice([f"{rng.randint(1, 28):02d}/{rng.randint(1, 12):02d}/{rng.randint(1990, 2025)}",
                               f"{rng.randint(1, 28):02d}-{rng.randint(1, 12):02d}-{rng.randint(10, 25)}",
                               f"{rng.randint(1, 28)} {rng.choice(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])} {rng.randint(1990, 2025)}",
                               f"{rng.randint(1990, 2025)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"])
    time_ = lambda: f"{rng.randint(0, 23):02d}:{rng.randint(0, 59):02d}" + (f":{rng.randint(0, 59):02d}" if rng.random() < 0.4 else "") + rng.choice(["", "", " AM", " PM"])
    num = lambda k: "".join(str(rng.randint(0, 9)) for _ in range(k))
    kind = rng.randint(0, 13)
    if kind == 0:
        return up(f"{rng.randint(1, 12)} {rng.choice(['x', 'X', '@', ''])} {W()} {W()}".replace("  ", " ")) + f" {cur()}{money()}"
    if kind == 1:
        return up(f"{W()} {W()}") + " " * rng.randint(1, 4) + f"{money()}"
    if kind == 2:
        return up(rng.choice(["Total", "Sub Total", "Subtotal", "Grand Total", "Cash", "Change", "Rounding", "Tax", "GST 6%", "Service Charge", "Total Incl. GST", "Amount Due", "Balance"])) + rng.choice([": ", " ", " : "]) + f"{cur()}{money()}"
    if kind == 3:
        return up(rng.choice(["Date", "Time", "Date/Time", "Invoice Date", "Due Date"])) + rng.choice([": ", " ", ":"]) + date() + (" " + time_() if rng.random() < 0.5 else "")
    if kind == 4:
        return up(rng.choice(["Tel", "Tel No", "Phone", "Fax", "Fax No.", "H/P"])) + rng.choice([": ", ". ", " "]) + rng.choice([f"{num(2)}-{num(4)} {num(4)}", f"({num(3)}) {num(3)}-{num(4)}", f"+{num(2)} {num(2)} {num(4)} {num(4)}", f"{num(3)}-{num(7)}"])
    if kind == 5:
        return up(rng.choice(["Invoice No", "INV#", "Receipt #", "Bill No", "Order No", "Ref", "Doc No", "Cashier", "Table", "GST ID", "GST Reg No", "Co. Reg. No", "Account No."])) + rng.choice([": ", " ", "#", ": #"]) + rng.choice([num(rng.randint(4, 12)), f"{rng.choice('ABCDEFGHKMPRST')}{num(rng.randint(4, 9))}", f"{num(6)}-{rng.choice('ABCDKPTUVW')}"])
    if kind == 6:
        return up(f"No. {rng.randint(1, 200)}, {W()} {rng.choice(['Street', 'Road', 'Avenue', 'Jalan', 'Lane'])} {rng.randint(1, 20)}" + (f", {num(5)} {W()}" if rng.random() < 0.5 else ""))
    if kind == 7:
        return rng.choice(["Name", "Date", "Signature", "Title", "Company", "Address", "Phone", "Fax", "Subject", "From", "To", "CC", "Re", "Approved by", "Department", "Brand", "Product", "Code", "Quantity", "Remarks", "Comments", "Yes", "No", "N/A"]) + rng.choice([":", ":", "", " :", "."])
    if kind == 8:
        return f"{W()} {W()} ({num(rng.randint(1, 4))}%)" if rng.random() < 0.5 else f"{rng.randint(1, 99)}.{rng.randint(0, 9)}% {W()}"
    if kind == 9:
        return f"{W()}@{W()}.{rng.choice(['com', 'org', 'net', 'com.my', 'co.uk'])}" if rng.random() < 0.5 else f"www.{W()}{W()}.{rng.choice(['com', 'org', 'net'])}"
    if kind == 10:
        return up(" ".join(W() for _ in range(rng.randint(1, 5))))
    if kind == 11:
        return f"{num(rng.randint(2, 4))} {W()} {num(rng.randint(1, 3))}/{num(rng.randint(1, 3))} {W()}"
    if kind == 12:
        return up(f"{W()} {rng.choice(['Sdn Bhd', 'Inc.', 'Ltd', 'LLC', 'Co.', 'Corp.', '& Sons'])}")
    return f"{rng.randint(1, 99)} {W()} {W()} {rng.choice(['-', '/', '|', '*', '#'])} {num(rng.randint(2, 6))}"


# ------------------------------------------------------------------ rendering and damage

def render(text, font_path, size, rng):
    font = ImageFont.truetype(font_path, size)
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    x0, y0, x1, y1 = probe.textbbox((0, 0), text, font=font)
    padx, pady = rng.randint(0, 14), rng.randint(0, 10)
    bg, ink = rng.randint(185, 255), rng.randint(0, 90)
    img = Image.new("L", (x1 - x0 + 2 * padx + 1, y1 - y0 + 2 * pady + 1), bg)
    ImageDraw.Draw(img).text((padx - x0, pady - y0), text, font=font, fill=ink)
    return img, bg, ink, (padx, pady, x1 - x0, y1 - y0)


def dot_matrix(img, bg, ink, size, rng):
    a = np.asarray(img, np.float32)
    on = a < (bg + ink) / 2
    cell = max(2, round(size / rng.uniform(10, 13)))
    h, w = a.shape
    out = Image.new("L", (w, h), bg)
    d = ImageDraw.Draw(out)
    r = cell * rng.uniform(0.35, 0.55)
    for y in range(0, h, cell):
        for x in range(0, w, cell):
            if on[y:y + cell, x:x + cell].mean() > 0.3:
                cx, cy = x + cell / 2, y + cell / 2
                d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=ink)
    return out


def damage(img, bg, ink, size, box, rng, nrng, others):
    tags = []
    if rng.random() < 0.18:
        return img, ["clean"]
    before = img
    if rng.random() < 0.12 and size >= 26:
        img = dot_matrix(img, bg, ink, size, rng); tags.append("dots")
    if rng.random() < 0.15:
        img = img.filter(ImageFilter.MinFilter(3)); tags.append("bleed")
    elif rng.random() < 0.1 and size >= 24 and "dots" not in tags:
        img = img.filter(ImageFilter.MaxFilter(3)); tags.append("thin")
    if ink_fraction(img, bg, ink) < 0.4 * ink_fraction(before, bg, ink):  # the strokes did not survive
        img, tags = before, []
    if rng.random() < 0.2:  # form rules
        d = ImageDraw.Draw(img)
        padx, pady, tw, th = box
        y = min(img.height - 1, pady + th + rng.randint(1, 4))
        d.line((0, y, img.width, y), fill=ink, width=rng.randint(1, 2))
        if rng.random() < 0.4:
            d.rectangle((0, 0, img.width - 1, img.height - 1), outline=ink, width=1)
        tags.append("rule")
    if rng.random() < 0.12 and others:  # fragment of a neighbouring line cut by the crop
        frag, fh = others
        a = np.asarray(img).copy()
        k = rng.randint(2, max(3, fh // 3))
        part = np.asarray(frag)[-k:, :a.shape[1]] if rng.random() < 0.5 else np.asarray(frag)[:k, :a.shape[1]]
        wpart = part.shape[1]
        if rng.random() < 0.5:
            a[:k, :wpart] = np.minimum(a[:k, :wpart], part[:k])
        else:
            a[-k:, :wpart] = np.minimum(a[-k:, :wpart], part[-k:])
        img = Image.fromarray(a); tags.append("neighbour")
    if rng.random() < 0.25:
        img = img.rotate(rng.uniform(-2.5, 2.5), resample=Image.BICUBIC, expand=True, fillcolor=bg); tags.append("rotate")
    if rng.random() < 0.3:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.4))); tags.append("blur")
    a = np.asarray(img, np.float32)
    if rng.random() < 0.25:  # lighting gradient and contrast loss
        horizontal = rng.random() < 0.6
        n = a.shape[1] if horizontal else a.shape[0]
        ramp = np.linspace(rng.uniform(0.45, 0.8), 1.0, n, dtype=np.float32)
        if rng.random() < 0.5:
            ramp = ramp[::-1]
        a = a * (ramp[None, :] if horizontal else ramp[:, None])
        m = a.mean(); a = m + rng.uniform(0.35, 0.8) * (a - m); tags.append("shade")
    if rng.random() < 0.15:  # faded
        a = 255 - (255 - a) * rng.uniform(0.25, 0.55); tags.append("faded")
    if rng.random() < 0.12:  # thermal streaks
        if rng.random() < 0.5:
            a = a + nrng.normal(0, 12, (1, a.shape[1]))
        else:
            a = a + nrng.normal(0, 12, (a.shape[0], 1))
        tags.append("streaks")
    if rng.random() < 0.3:
        a = a + nrng.normal(0, rng.uniform(4, 28), a.shape); tags.append("noise")
    if rng.random() < 0.08:
        sp = nrng.random(a.shape)
        a[sp < 0.01] = 0; a[sp > 0.99] = 255; tags.append("speckle")
    a = np.clip(a, 0, 255)
    if rng.random() < 0.12 and "blur" not in tags and "shade" not in tags:  # bitonal (fax, Check 21-style images)
        a = np.where(a < (a.min() + a.max()) / 2 + rng.uniform(-15, 15), 0, 255).astype(np.float32); tags.append("bitonal")
    img = Image.fromarray(a.astype(np.uint8))
    if rng.random() < 0.25:
        scale = rng.uniform(5, 9) / max(6, size * 0.5)
        if scale < 0.95:
            img = img.resize((max(4, round(img.width * scale)), max(4, round(img.height * scale))), Image.LANCZOS); tags.append("lowres")
    if rng.random() < 0.25:
        buf = io.BytesIO(); img.save(buf, format="JPEG", quality=rng.randint(10, 50)); buf.seek(0)
        img = Image.open(buf).convert("L"); tags.append("jpeg")
    if rng.random() < 0.05:
        img = Image.fromarray(255 - np.asarray(img)); tags.append("invert")
    return img, tags or ["mild"]


def ink_fraction(img, bg, ink):
    return float((np.asarray(img, np.float32) < (bg + ink) / 2).mean())


def legible(img):
    """Enough contrast between ink and paper, and some but not most pixels on the ink side."""
    a = np.asarray(img, np.float32)
    p2, p50, p98 = np.percentile(a, [2, 50, 98])
    c = max(p50 - p2, p98 - p50)
    if c < 40:
        return False
    far = np.abs(a - p50) > c / 2
    return 0.01 < far.mean() < 0.6


_FONTS = None


def work(args):
    i, text, seed = args
    global _FONTS
    if _FONTS is None:
        _FONTS = [(f, cm) for f, cm in font_index() if not any(s in f for s in SKIP_FONTS)]
    rng = random.Random(seed)
    nrng = np.random.default_rng(i + 1_000_003)
    fonts = covering_fonts(_FONTS, text)
    if not fonts:
        return None
    font = rng.choice(fonts)
    size = rng.randint(14, 48)
    try:
        img, bg, ink, box = render(text, font, size, rng)
        other, *_ = render(text[::-1], font, size, rng)
        clean_img = img
        for _ in range(3):
            img, tags = damage(clean_img, bg, ink, size, box, rng, nrng, (other, other.height))
            if legible(img):
                break
        else:
            img, tags = clean_img, ["clean"]
    except Exception as e:  # a font that cannot render this text: skip the line, say so
        return {"i": i, "error": repr(e)[:120]}
    name = f"{i:06d}.png"
    img.save(OUT / "lines" / name)
    return {"i": i, "file": f"lines/{name}", "text": text, "font": Path(font).name, "size": size, "tags": tags}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=24000)
    ap.add_argument("--jobs", type=int, default=3)
    ap.add_argument("--doc-share", type=float, default=0.45)
    a = ap.parse_args()
    (OUT / "lines").mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    exclude = Excluded()
    paras = wiki_text()
    wl = wiki_lines(paras, exclude, rng)
    vocab = sorted({w.strip(".,;:!?()\"'") for s in wl for w in s.split() if w.strip(".,;:!?()\"'").isalpha() and 2 <= len(w) <= 14})
    print(f"{len(paras)} paragraphs, {len(wl)} wiki lines, vocab {len(vocab)}", flush=True)
    texts = []
    for _ in range(a.n):
        if rng.random() < a.doc_share:
            t = norm_chars(doc_line(rng, vocab))
        else:
            t = rng.choice(wl)
        if set(t) <= CHARSET and t.strip() and t not in exclude:
            texts.append(t)
    jobs = [(i, t, f"{SEED}:{i}") for i, t in enumerate(texts)]
    rows, errs = [], 0
    with Pool(a.jobs) as pool:
        for k, r in enumerate(pool.imap_unordered(work, jobs, chunksize=64)):
            if r is None or "error" in r:
                errs += 1
                continue
            rows.append(r)
            if k % 4000 == 0:
                print(f"{k}/{len(jobs)}", flush=True)
    rows.sort(key=lambda r: r["i"])
    with open(OUT / "labels.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {len(rows)} lines, {errs} skipped", flush=True)


if __name__ == "__main__":
    main()
