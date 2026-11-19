"""NumPy reference for Tesseract's LSTM line recognizer, following the C++ op for op.

This is the spec the GPU ports (server/torch_engine.py, web/*.js) are checked against.
Sources: tesseract src/lstm/{input,networkio,convolve,maxpool,reversed,lstm,fullyconnected,
functions,lstmrecognizer}.cpp and ccutil/helpers.h (TRand).

Semantics reproduced exactly:
  * input normalisation: black/white from local extrema of the middle row (STATS::ile),
    v = (p - black) / ((white - black) / 2) - 1, int8 models quantise v * 128;
  * Convolve pads outside the image with values from Tesseract's seeded TRand, in the
    order Convolve::Forward draws them;
  * tanh / logistic through Tesseract's 4096-entry lookup tables with linear interpolation;
  * int8 models: int8 activations between layers (round(127 v), clipped), int32 dot
    products, bias * 127, per-row scales;
  * LSTM state clipped to +-100, summarising LSTM keeps the last step of each row.
Float models run in float64 here (Tesseract uses float32), so logits differ in the last
bits; argmax paths agree.
"""

import numpy as np

from .traineddata import LSTMS

TABLE = 4096
SCALE = 256.0
_IDX = np.arange(TABLE) / SCALE
TANH_T = np.tanh(_IDX)
LOGISTIC_T = 1.0 / (1.0 + np.exp(-_IDX))
STATE_CLIP = 100.0


def _lut(x, table):
    a = np.abs(x)
    xs = a * SCALE
    idx = np.minimum(xs.astype(np.int64), TABLE - 2)
    v = table[idx] + (table[idx + 1] - table[idx]) * (xs - idx)
    return np.where(a >= (TABLE - 1) / SCALE, 1.0, v)


def tanh(x):
    v = _lut(x, TANH_T)
    return np.where(x < 0, -v, v)


def logistic(x):
    v = _lut(x, LOGISTIC_T)
    return np.where(x < 0, 1.0 - v, v)


def int_cast_rounded(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, np.floor(x + 0.5), -np.floor(-x + 0.5)).astype(np.int64)


def quant(v):
    """NetworkIO::WriteTimeStepPart in int mode: clip to [-1, 1], round(127 v)."""
    return int_cast_rounded(np.clip(v, -1.0, 1.0) * 127)


class TRand:
    """ccutil/helpers.h TRand. Tesseract 5.0-5.3.x (the 5.3.4 that Ubuntu 24.04 ships) wraps
    std::minstd_rand; current master replaced it with a 64-bit LCG. Only Convolve's padding
    outside the image draws from it, so the choice moves border columns and nothing else."""
    MINSTD_M = 2147483647
    LCG_M = 6364136223846793005
    LCG_A = 1442695040888963407

    def __init__(self, seed=1, kind="minstd"):
        self.kind = kind
        seed &= 0xFFFFFFFFFFFFFFFF
        if kind == "minstd":
            self.state = seed % self.MINSTD_M or 1
        else:
            self.state = seed

    def int_rand(self):
        if self.kind == "minstd":
            self.state = self.state * 48271 % self.MINSTD_M
            return self.state
        self.state = (self.state * self.LCG_M + self.LCG_A) & 0xFFFFFFFFFFFFFFFF
        return self.state >> 33

    def signed_rand(self, rng):
        return rng * 2.0 * self.int_rand() / 2147483647 - rng

    @classmethod
    def for_line(cls, sample_iteration, kind="minstd"):
        """LSTMRecognizer::SetRandomSeed: seed = sample_iteration * 0x10000001, one draw."""
        r = cls(sample_iteration * 0x10000001, kind)
        r.int_rand()
        return r


def stats_ile(counts, frac):
    """STATS(0, 255)::ile."""
    total = int(counts.sum())
    if total == 0:
        return 0.0
    target = min(max(frac * total, 1.0), float(total))
    s = 0
    i = 0
    while i <= 255 and s < target:
        s += int(counts[i])
        i += 1
    return i - (s - target) / counts[i - 1]


def black_white(gray):
    """networkio.cpp ComputeBlackWhite: quartiles of the middle row's local extrema."""
    h, w = gray.shape
    mins = np.zeros(256, np.int64)
    maxes = np.zeros(256, np.int64)
    if w >= 3:
        row = gray[h // 2].astype(np.int64)
        prev, curr, nxt = row[:-2], row[1:-1], row[2:]
        lmin = ((curr < prev) & (curr <= nxt)) | ((curr <= prev) & (curr < nxt))
        lmax = ((curr > prev) & (curr >= nxt)) | ((curr >= prev) & (curr > nxt))
        np.add.at(mins, curr[lmin], 1)
        np.add.at(maxes, curr[lmax], 1)
    if mins.sum() == 0:
        mins[0] = 1
    if maxes.sum() == 0:
        maxes[255] = 1
    return stats_ile(mins, 0.25), stats_ile(maxes, 0.75)


def normalise(gray, int_mode):
    """NetworkIO::FromPixes + SetPixel for an 8-bit image already at the input height."""
    black, white = black_white(gray)
    black = np.float32(black)
    contrast = np.float32((white - black) / 2.0)
    if contrast <= 0:
        contrast = np.float32(1.0)
    v = (gray.astype(np.float32) - black) / contrast - np.float32(1.0)
    if int_mode:
        return np.clip(int_cast_rounded(128.0 * v.astype(np.float64)), -127, 127)
    return v.astype(np.float64)


def conv_pad_values(h, w, half_x, half_y, ni, rng, int_mode):
    """The values Convolve::Forward writes outside the image, in draw order.

    Returns a list of (y, x, feature_offset, values) for every out-of-image block.
    """
    ys = 2 * half_y + 1
    out = []

    def draw(n):
        vals = [rng.signed_rand(127.0 if int_mode else 1.0) for _ in range(n)]
        if int_mode:
            return int_cast_rounded(vals)
        return np.asarray(vals, np.float32).astype(np.float64)

    border = [(y, x) for y in range(h) for x in range(w)
              if y < half_y or y >= h - half_y or x < half_x or x >= w - half_x]
    for y, x in border:
        out_ix = 0
        for dx in range(-half_x, half_x + 1):
            if not 0 <= x + dx < w:
                out.append((y, x, out_ix, draw(ys * ni)))
            else:
                off = out_ix
                for dy in range(-half_y, half_y + 1):
                    if not 0 <= y + dy < h:
                        out.append((y, x, off, draw(ni)))
                    off += ni
            out_ix += ys * ni
    return out


def convolve(a, node, rng, int_mode, pads=None):
    hx, hy = node.half
    h, w, ni = a.shape
    kx, ky = 2 * hx + 1, 2 * hy + 1
    padded = np.zeros((h + 2 * hy, w + 2 * hx, ni), a.dtype)
    padded[hy:hy + h, hx:hx + w] = a
    out = np.empty((h, w, kx * ky * ni), a.dtype)
    i = 0
    for dx in range(kx):
        for dy in range(ky):
            out[:, :, i:i + ni] = padded[dy:dy + h, dx:dx + w]
            i += ni
    if pads is None:
        pads = conv_pad_values(h, w, hx, hy, ni, rng, int_mode)
    for y, x, off, vals in pads:
        out[y, x, off:off + len(vals)] = vals
    return out


def dense(x, wts, int_mode):
    """WeightMatrix::MatrixDotVector over the last axis (bias is the last column)."""
    if int_mode:
        q = wts.q.astype(np.int64)
        acc = x.astype(np.int64) @ q[:, :-1].T + q[:, -1] * 127
        return acc * wts.scale
    w = wts.w
    return x @ w[:, :-1].T + w[:, -1]


def fully_connected(a, node, int_mode):
    v = dense(a, node.weights, int_mode)
    t = node.type
    if t == "Tanh":
        v = tanh(v)
    elif t == "Logistic":
        v = logistic(v)
    elif t == "Relu":
        v = np.maximum(v, 0)
    elif t in ("Softmax", "SoftmaxNoCTC"):
        m = v.max(axis=-1, keepdims=True)
        e = np.exp(np.clip(v - m, -86, 0))
        return e / e.sum(axis=-1, keepdims=True)  # softmax output is always float
    elif t != "Linear":
        raise ValueError(t)
    return quant(v) if int_mode else v


def maxpool(a, node):
    sx, sy = node.scale
    h, w, f = a.shape
    ho, wo = h // sy, w // sx
    b = a[:ho * sy, :wo * sx].reshape(ho, sy, wo, sx, f)
    return b.max(axis=(1, 3))


def lstm(a, node, int_mode):
    """1-d LSTM along x for every row of a [H, W, ni]; rows are independent."""
    if node.is_2d or node.softmax is not None:
        raise NotImplementedError(node.type)
    h, w, ni = a.shape
    ns = node.ns
    g = node.gates
    state = np.zeros((h, ns))
    out_f = np.zeros((h, ns))
    prev = np.zeros((h, ns), np.int64) if int_mode else out_f
    outs = []
    for t in range(w):
        src = np.concatenate([a[:, t], prev], axis=1)
        ci = tanh(dense(src, g["CI"], int_mode))
        gi = logistic(dense(src, g["GI"], int_mode))
        gf = logistic(dense(src, g["GF1"], int_mode))
        go = logistic(dense(src, g["GO"], int_mode))
        state = np.clip(state * gf + ci * gi, -STATE_CLIP, STATE_CLIP)
        out_f = tanh(state) * go
        prev = quant(out_f) if int_mode else out_f
        outs.append(prev)
    seq = np.stack(outs, axis=1)
    if node.type == "SummLSTM":
        return seq[:, -1:]
    return seq


def run(node, a, ctx):
    t = node.type
    if t == "Series":
        for c in node.children:
            a = run(c, a, ctx)
        return a
    if t == "Input":
        return a
    if t == "Convolve":
        return convolve(a, node, ctx["rng"], ctx["int"], ctx.get("pads"))
    if t == "Maxpool":
        return maxpool(a, node)
    if t == "XYTranspose":
        return run(node.children[0], a.transpose(1, 0, 2), ctx).transpose(1, 0, 2)
    if t == "RTLReversed":
        return run(node.children[0], a[:, ::-1], ctx)[:, ::-1]
    if t == "TTBReversed":
        return run(node.children[0], a[::-1], ctx)[::-1]
    if t in LSTMS:
        return lstm(a, node, ctx["int"])
    if node.weights is not None:
        return fully_connected(a, node, ctx["int"])
    if t == "Dropout":
        return a
    raise NotImplementedError(t)


def forward(model, gray, rng=None):
    """gray: uint8 [H, W] at the model's input height. Returns softmax [T, classes]."""
    if gray.shape[0] != model.input_height:
        raise ValueError(f"line must be {model.input_height} px high, got {gray.shape[0]}")
    x = normalise(gray, model.int_mode)[:, :, None]
    if rng is None:
        rng = TRand.for_line(model.sample_iteration)
    out = run(model.network, x, {"rng": rng, "int": model.int_mode})
    return out[0]


def scale_to_height(gray, height):
    """Resize a line to the network height (PIL bilinear/area). Leptonica's pixScale is not
    reproduced, so only lines already at the input height give bit-level input parity."""
    from PIL import Image
    h, w = gray.shape
    if h == height:
        return gray
    f = height / h
    nw = max(1, int(round(w * f)))
    im = Image.fromarray(gray).resize((nw, height), Image.BOX if f < 1 else Image.BILINEAR)
    return np.asarray(im)
