"""Cheap line-level arms for reading text lines with Tesseract's LSTM, run on the GPU engine.

Every arm changes only the image before the network (prepare), or chooses among readings the
network already produced (vote, pick_by_confidence). Nothing here changes the weights or
Tesseract's decoder.

prepare(gray, flags, H, inverted) -> uint8 array at the model height H
    flags (all optional):
      invert   flip the line when ref.needs_invert says most pixels are dark
      deskew   rotate by the angle in [-3, 3] deg that maximises the row-profile variance
      stretch  linear map of the 2nd..98th percentile to 0..255
      pad      n pixels of background border (n in {2, 4, 8}) added before scaling
      upscale  lines under 20 px high: Lanczos straight to H instead of ref.scale_to_height
      tight    crop to the ink bounding box plus 4 px (ink found after flatfield)
      flatfield  flat-field the line (background from a grey closing)
    `inverted` is ref.needs_invert(gray) computed once per line by the caller.

confidence(probs) -> float or None
    mean over timesteps of the largest softmax value (None for an empty output).

pick_by_confidence(readings) -> index of the reading with the highest confidence.

vote(readings) -> str
    ROVER-style voting. The highest-confidence reading is the pivot; every other reading is
    aligned to it with a Levenshtein alignment. Each pivot position takes the character with the
    largest summed confidence (a deletion votes for nothing), and each gap between pivot
    characters keeps an inserted string only when one string holds more than half the total
    confidence.
"""

import math

import numpy as np
from PIL import Image
from rapidfuzz.distance import Levenshtein

from tessgpu import ref

SMALL_LINE = 20                       # px: lines shorter than this get the Lanczos path
DESKEW_RANGE = (-3.0, 3.0, 0.25)      # degrees: start, stop, step
PADS = (2, 4, 8)


def background(gray):
    """Median of the outermost pixel ring: the line crop's own background."""
    ring = np.concatenate([gray[0], gray[-1], gray[:, 0], gray[:, -1]])
    return int(np.median(ring))


def invert(gray):
    return (255 - gray).astype(np.uint8)


def stretch(gray):
    lo, hi = np.percentile(gray, [2, 98])
    if hi <= lo:
        return gray
    out = (gray.astype(np.float32) - lo) * (255.0 / (hi - lo))
    return np.clip(out, 0, 255).astype(np.uint8)


def pad(gray, n):
    if n <= 0:
        return gray
    return np.pad(gray, n, mode="constant", constant_values=background(gray))


def _ink_profile_var(gray, thr):
    ink = (gray < thr).astype(np.float32)
    rows = ink.sum(axis=1)
    return float(rows.var()) if rows.size > 1 else 0.0


def skew_angle(gray):
    """Angle (degrees, PIL convention) whose rotation makes the text rows most distinct."""
    lo, hi = np.percentile(gray, [2, 98])
    thr = (lo + hi) / 2.0
    bg = background(gray)
    best_a, best_v = 0.0, -1.0
    for a in np.arange(DESKEW_RANGE[0], DESKEW_RANGE[1] + 1e-9, DESKEW_RANGE[2]):
        r = np.asarray(Image.fromarray(gray).rotate(float(a), resample=Image.BICUBIC,
                                                    expand=True, fillcolor=bg))
        v = _ink_profile_var(r, thr)
        if v > best_v + 1e-12:
            best_a, best_v = float(a), v
    return best_a


def rotate(gray, angle):
    if angle == 0.0:
        return gray
    bg = background(gray)
    return np.asarray(Image.fromarray(gray).rotate(angle, resample=Image.BICUBIC,
                                                   expand=True, fillcolor=bg))


def _lanczos_to_height(gray, H):
    h, w = gray.shape
    nw = max(1, int(round(w * H / h)))
    return np.asarray(Image.fromarray(gray).resize((nw, H), Image.LANCZOS))


def tight(g, pad=4):
    """Crop to the ink bounding box plus pad px. Copied from run/crop_check.py (tight)."""
    f = flatfield(g)  # dark ink on light paper after flatfield
    ink = f < 128
    ys, xs = np.where(ink)
    if len(ys) == 0:
        return g
    y0, y1 = max(0, ys.min() - pad), min(g.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(g.shape[1], xs.max() + pad + 1)
    return g[y0:y1, x0:x1]


def flatfield(gray):
    """Flat-field: grey closing (window ~1.5x line height) and box blur as background, then
    gray / background and a 1..99 percentile stretch. Output is dark ink on light paper.
    Copied from run/flatfield_check.py (flatfield)."""
    from scipy import ndimage
    g = gray.astype(np.float32)
    if ref.needs_invert(gray):
        g = 255 - g
    k = max(9, int(1.5 * g.shape[0]) | 1)
    bg = ndimage.grey_closing(g, size=(k, k))
    bg = ndimage.uniform_filter(bg, size=k)
    out = g / np.maximum(bg, 1) * 255
    lo, hi = np.percentile(out, [1, 99])
    if hi > lo:
        out = (out - lo) * 255 / (hi - lo)
    return np.clip(out, 0, 255).astype(np.uint8)


def prepare(gray, flags, H, inverted=None):
    """The line as the network should see it: uint8 at height H.

    tight and flatfield run first (tight finds the ink with flatfield). flatfield output is
    already dark-on-light, so invert is not applied on top of it."""
    g = np.asarray(gray, np.uint8)
    if flags.get("tight"):
        g = tight(g)
    if flags.get("flatfield"):
        g = flatfield(g)
    small = g.shape[0] < SMALL_LINE
    if flags.get("invert") and not flags.get("flatfield"):
        if inverted is None:
            inverted = ref.needs_invert(g)
        if inverted:
            g = invert(g)
    if flags.get("deskew"):
        g = rotate(g, skew_angle(g))
    if flags.get("stretch"):
        g = stretch(g)
    if flags.get("pad"):
        g = pad(g, int(flags["pad"]))
    if flags.get("upscale") and small:
        return _lanczos_to_height(g, H)
    return ref.scale_to_height(g, H)


def confidence(probs):
    p = np.asarray(probs)
    if p.ndim != 2 or p.shape[0] == 0:
        return None
    return float(p.max(axis=1).mean())


def pick_by_confidence(confs):
    """Index of the highest confidence; readings with None confidence never win."""
    best, best_c = None, -math.inf
    for i, c in enumerate(confs):
        if c is not None and c > best_c:
            best, best_c = i, c
    return best


def _align(pivot, text):
    """For each pivot position: the text character aligned to it (or '' for a deletion), and
    for each gap before a pivot position (and after the last one): the inserted string."""
    n = len(pivot)
    sub = [""] * n
    ins = [""] * (n + 1)
    for tag, i1, i2, j1, j2 in Levenshtein.opcodes(pivot, text):
        if tag == "equal" or tag == "replace":
            span = min(i2 - i1, j2 - j1)
            for k in range(span):
                sub[i1 + k] = text[j1 + k]
            if i2 - i1 > span:                       # pivot chars with no partner
                for k in range(span, i2 - i1):
                    sub[i1 + k] = ""
            if j2 - j1 > span:                       # extra text chars: insert after the span
                ins[i1 + span] += text[j1 + span:j2]
        elif tag == "delete":
            for k in range(i1, i2):
                sub[k] = ""
        elif tag == "insert":
            ins[i1] += text[j1:j2]
    return sub, ins


def vote(readings, weighted=True):
    """readings: list of (text, confidence or None), first reading first.

    weighted=True: each reading votes with its confidence; the pivot is the most confident.
    weighted=False: every reading votes once and the first reading is the pivot, so a
    majority of three readings is well defined and ignores confidence."""
    if weighted:
        rs = [(t, c) for t, c in readings if c is not None]
        if not rs:
            return ""
        pivot_i = pick_by_confidence([c for _, c in rs])
        rs = [(t, c) for t, c in rs]
    else:
        rs = [(t, 1.0) for t, _ in readings]
        if not rs:
            return ""
        pivot_i = 0
    pivot = rs[pivot_i][0]
    total = sum(c for _, c in rs)
    n = len(pivot)
    sub_votes = [dict() for _ in range(n)]
    ins_votes = [dict() for _ in range(n + 1)]
    for text, w in rs:
        sub, ins = _align(pivot, text)
        for i in range(n):
            sub_votes[i][sub[i]] = sub_votes[i].get(sub[i], 0.0) + w
        for j in range(n + 1):
            if ins[j]:
                ins_votes[j][ins[j]] = ins_votes[j].get(ins[j], 0.0) + w
    out = []
    for j in range(n + 1):
        if ins_votes[j]:
            s, w = max(ins_votes[j].items(), key=lambda kv: kv[1])
            if w > total / 2:
                out.append(s)
        if j < n:
            votes = sub_votes[j]
            if votes:
                ch = max(votes.items(), key=lambda kv: (kv[1], kv[0] == pivot[j]))[0]
                out.append(ch)
    return "".join(out)
