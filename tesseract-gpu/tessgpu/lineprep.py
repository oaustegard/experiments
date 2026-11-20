"""Line clean-up before the recognizer: flat-field the lighting, then crop tightly to the ink.

clean_line(gray) is what web/lineprep.js ports (checked by web/test/check_lineprep.mjs):
  box from flatfield(gray) < 128, padded by 4 px; crop the original; flatfield the crop.
flatfield(gray): flip a light-on-dark line (ref.needs_invert); background = grey closing (k x k,
  k = max(9, 1.5 * height) | 1) smoothed by a k x k box filter; gray / background * 255; stretch the
  1st..99th percentile to 0..255.

Measured on eng tessdata_best with the exact engine (run/crop_check.py): CER on clean synthetic lines
1.06% -> 0.28%, shaded 1.48% -> 0.52%, real scans 14.2% -> 13.0% (case-insensitive).
"""
import numpy as np
from scipy import ndimage

from tessgpu import ref

PAD = 4


def flatfield(gray):
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


def tight_box(gray, pad=PAD):
    ys, xs = np.where(flatfield(gray) < 128)
    if len(ys) == 0:
        return None
    return (max(0, xs.min() - pad), max(0, ys.min() - pad),
            min(gray.shape[1], xs.max() + pad + 1), min(gray.shape[0], ys.max() + pad + 1))


def clean_line(gray):
    b = tight_box(gray)
    if b is None:
        return flatfield(gray)
    x0, y0, x1, y1 = b
    return flatfield(np.ascontiguousarray(gray[y0:y1, x0:x1]))
