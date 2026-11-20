"""Small image perturbations for test-time augmentation of Indra OCR.

Two places to perturb. PAGE maps a uint8 grayscale page to another before the
whole pipeline reads it, so detection, the script classifier and reading order
see the change too. CROP maps the recognizer's input batch, a float32 ink map
[B, 1, H, W] (1 = ink, 0 = paper; rec_preproc.crop_line computes 1 - pixel/255
with no per-crop normalization), so the line boxes stay fixed and every
variant's CTC posteriors cover the same timesteps and can be averaged.

Color: indra-ocr reads every page as grayscale (cv2.IMREAD_GRAYSCALE, or
pdfium's grayscale render), so a hue change reaches it only through the gray
conversion. The color variant is a different conversion, the darkest of the
three channels (prerender.py --mode min), not a function here.
"""
from __future__ import annotations

import numpy as np


def _lut(f):
    x = np.arange(256, dtype=np.float32) / 255.0
    return np.clip(np.round(255.0 * f(x)), 0, 255).astype(np.uint8)


def _gamma(g):
    lut = _lut(lambda p: p ** g)
    return lambda gray: lut[gray]


def _contrast(c):
    lut = _lut(lambda p: 0.5 + c * (p - 0.5))
    return lambda gray: lut[gray]


def _otsu(gray):
    import cv2

    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def _morph(op):
    def f(gray):
        import cv2

        return op(gray, np.ones((2, 2), np.uint8))

    return f


def _thicken(gray):  # dark ink grows: a min filter on the page
    import cv2

    return cv2.erode(gray, np.ones((2, 2), np.uint8))


def _thin(gray):  # dark ink shrinks: a max filter on the page
    import cv2

    return cv2.dilate(gray, np.ones((2, 2), np.uint8))


def _blur(gray):
    import cv2

    return cv2.GaussianBlur(gray, (0, 0), 1.0)


def _sharpen(gray):
    import cv2

    b = cv2.GaussianBlur(gray, (0, 0), 1.5).astype(np.float32)
    return np.clip(gray.astype(np.float32) * 1.8 - b * 0.8, 0, 255).astype(np.uint8)


def _scale(k):
    def f(gray):
        import cv2

        h, w = gray.shape
        interp = cv2.INTER_AREA if k < 1 else cv2.INTER_CUBIC
        return cv2.resize(gray, (max(1, round(w * k)), max(1, round(h * k))), interpolation=interp)

    return f


PAGE = {
    "gamma0.5": _gamma(0.5),  # midtones lighter: faint strokes fade
    "gamma2": _gamma(2.0),  # midtones darker: faint strokes darken and widen
    "contrast1.5": _contrast(1.5),
    "contrast0.6": _contrast(0.6),  # washed out: paper goes gray, ink goes gray
    "otsu": _otsu,
    "thicken": _thicken,
    "thin": _thin,
    "blur": _blur,
    "sharpen": _sharpen,
    "scale0.75": _scale(0.75),
    "scale1.33": _scale(4 / 3),
}

# Page variants whose output boxes are in another pixel frame: the scorer reads
# block boxes against the page size, so opb_eval reports image_size of the
# perturbed page (the adapter normalizes boxes by it).
RESCALES = {"scale0.75", "scale1.33"}


# ---- crop level: x is ink in [0, 1], 1 = ink -------------------------------------


def _cgamma(g):
    # In pixel terms p = 1 - ink: p' = p ** g.
    return lambda x: 1.0 - np.clip(1.0 - x, 0.0, 1.0) ** g


def _ccontrast(c):
    return lambda x: np.clip(0.5 + c * (x - 0.5), 0.0, 1.0)


def _cbin(x):
    # Per-line Otsu on the ink map (its valid width is unknown here; padding is 0
    # ink, i.e. paper, so it only adds to the paper mode).
    import cv2

    out = np.empty_like(x)
    for i in range(x.shape[0]):
        u8 = np.round(x[i, 0] * 255).astype(np.uint8)
        out[i, 0] = cv2.threshold(u8, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1] / 255.0
    return out


def _cmorph(kind):
    def f(x):
        import cv2

        k = np.ones((2, 2), np.uint8)
        op = cv2.dilate if kind == "thicken" else cv2.erode  # ink is bright here
        return np.stack([op(x[i, 0], k)[None] for i in range(x.shape[0])])

    return f


def _cblur(x):
    import cv2

    return np.stack([cv2.GaussianBlur(x[i, 0], (0, 0), 0.7)[None] for i in range(x.shape[0])])


def _csharpen(x):
    import cv2

    b = np.stack([cv2.GaussianBlur(x[i, 0], (0, 0), 1.0)[None] for i in range(x.shape[0])])
    return np.clip(1.8 * x - 0.8 * b, 0.0, 1.0)


CROP = {
    "id": lambda x: x,
    "gamma0.5": _cgamma(0.5),
    "gamma2": _cgamma(2.0),
    "contrast1.5": _ccontrast(1.5),
    "contrast0.6": _ccontrast(0.6),
    "otsu": _cbin,
    "thicken": _cmorph("thicken"),
    "thin": _cmorph("thin"),
    "blur": _cblur,
    "sharpen": _csharpen,
}
