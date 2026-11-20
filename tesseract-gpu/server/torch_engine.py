"""PyTorch engine for Tesseract's LSTM line recogniser: many text lines per call.

    eng = TorchTess("models/best/eng.traineddata", device="cpu", mode="exact")
    probs = eng.recognize(images)     # list of float32 [T_i, C]
    texts = eng.texts(images)         # list of str (greedy CTC, tessgpu.decode)

images are uint8 greyscale arrays of any height; each is scaled to the model height with
ref.scale_to_height and normalised with ref.normalise on the CPU. Lines are sorted longest
first and packed into one padded batch: the conv/pool stages run on chunks of lines, the
x-LSTMs run over the whole batch with the batch shrinking to the still-running prefix.

Modes
  "exact"  reproduces tessgpu.ref.forward: the pad table from pad_table_fast, LUT tanh and
           logistic (Tesseract's 4096-entry tables), state clip +-100, and for int8 models
           round-half-away activations between layers with integer dot products in float32
           (exact: every partial sum stays below 2^24) times the per-row scales.
  "cudnn"  torch.nn.LSTM (PackedSequence for the x-layers, reversed layer flipped within
           each line's length) with Tesseract's gate blocks mapped to the cuDNN gate order;
           int8 weights dequantised as q * scale * 127; plain tanh/sigmoid, no clipping,
           no activation quantisation.

The engine only needs tessgpu/ on the import path (run from the repository root).
"""

import numpy as np
import torch
from torch import nn

from tessgpu import decode, ref
from tessgpu.traineddata import load

CHUNK = 32                       # lines per conv/pool chunk (bounds the [c,H,W,16] tensor)
_MASK64 = (1 << 64) - 1
_MINSTD_M = 2147483647           # ccutil TRand: std::minstd_rand
_MINSTD_A = 48271
_STATE_CACHE = {}                # sample_iteration -> [last_state, np.int64 states]


# ----------------------------------------------------------------------------------------
# Convolve padding

def _minstd_states(sample_iteration, n):
    """The first n states of the one random stream every line of a model shares.

    LSTMRecognizer::SetRandomSeed seeds TRand with sample_iteration * 0x10000001 and draws
    once; Convolve then draws padding values from that stream in border order. Each line
    starts from the same seed, so the stream is computed once per model and each (H, W)
    takes a prefix of it."""
    ent = _STATE_CACHE.get(sample_iteration)
    if ent is None:
        seed = (sample_iteration * 0x10000001) & _MASK64
        s = (seed % _MINSTD_M) or 1
        ent = [s * _MINSTD_A % _MINSTD_M, np.zeros(0, np.int64)]
        _STATE_CACHE[sample_iteration] = ent
    last, arr = ent
    if len(arr) < n:
        grow = max(n, 2 * len(arr), 4096)
        new = []
        s = last
        for _ in range(grow - len(arr)):
            s = s * _MINSTD_A % _MINSTD_M
            new.append(s)
        ent[0] = s
        ent[1] = np.concatenate([arr, np.asarray(new, np.int64)])
    return ent[1][:n]


def _conv_half(model):
    return model.network.children[1].children[0].half    # (half_x, half_y)


def pad_table_fast(model, H, W):
    """Vectorised equal of ref.pad_table(H, W, TRand.for_line(sample_iteration), int_mode).

    Returns float64 [n_border, kx*ky]: one row per out-of-image-touching position in raster
    order, slot k = (dx+half_x)*ky + (dy+half_y). A slot is drawn exactly when its dx or its
    dy leaves the image (Convolve draws a whole dx block when x is outside); the k-th draw
    overall is element k of the model's stream, so the table is a boolean scatter."""
    hx, hy = _conv_half(model)
    kx, ky = 2 * hx + 1, 2 * hy + 1
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    border = (yy < hy) | (yy >= H - hy) | (xx < hx) | (xx >= W - hx)
    by, bx = yy[border], xx[border]
    dxs = np.arange(-hx, hx + 1)
    dys = np.arange(-hy, hy + 1)
    ox = (bx[:, None] + dxs[None, :] < 0) | (bx[:, None] + dxs[None, :] >= W)   # [nb, kx]
    oy = (by[:, None] + dys[None, :] < 0) | (by[:, None] + dys[None, :] >= H)   # [nb, ky]
    drawn = (ox[:, :, None] | oy[:, None, :]).reshape(len(by), kx * ky)
    n = int(drawn.sum())
    states = _minstd_states(model.sample_iteration, n)
    if model.int_mode:
        raw = 127.0 * 2.0 * states / _MINSTD_M - 127.0          # rng * 2 * k / M - rng
        vals = np.where(raw >= 0, np.floor(raw + 0.5), -np.floor(-raw + 0.5))
    else:
        raw = 1.0 * 2.0 * states / _MINSTD_M - 1.0
        vals = raw.astype(np.float32).astype(np.float64)
    table = np.zeros((len(by), kx * ky))
    table[drawn] = vals
    return table


# ----------------------------------------------------------------------------------------
# Weights

def _mats(weights, int_mode, mode):
    """Stack Weights objects row-wise into (WT, Bd, S, no, ni).

    WT is [ni, no] (already transposed), Bd the bias, S the per-row scale. exact int8:
    integer codes with bias * 127 and the row scale applied after the integer accumulate;
    everything else: float weights with S = 1. cudnn int8: dequantised q * scale * 127."""
    if int_mode and mode == "exact":
        Q = np.concatenate([w.q for w in weights]).astype(np.float32)
        S = np.concatenate([w.scale for w in weights]).astype(np.float32)
        Wd = Q[:, :-1]
        Bd = Q[:, -1] * np.float32(127.0)
    else:
        if int_mode:
            F = np.concatenate([w.q.astype(np.float64) * w.scale[:, None] * 127.0
                                for w in weights])
        else:
            F = np.concatenate([w.w for w in weights]).astype(np.float64)
        Wd = F[:, :-1].astype(np.float32)
        Bd = F[:, -1].astype(np.float32)
        S = np.ones(F.shape[0], np.float32)
    return Wd.T.copy(), Bd, S


GATES = ("CI", "GI", "GF1", "GO")          # Tesseract block order; CI=tanh, others logistic


class _Affine:
    def __init__(self, WT, Bd, S, dev):
        self.WT = torch.as_tensor(WT, device=dev)
        self.Bd = torch.as_tensor(Bd, device=dev)
        self.S = torch.as_tensor(S, device=dev)


class _Lstm(_Affine):
    def __init__(self, node, int_mode, mode, dev):
        ns = node.ns
        WT, Bd, S = _mats([node.gates[g] for g in GATES], int_mode, mode)
        self.ns = ns
        self.ni = WT.shape[0] - ns
        self.WinT = torch.as_tensor(WT[:self.ni], device=dev)
        self.WrecT = torch.as_tensor(WT[self.ni:], device=dev)
        self.Bd = torch.as_tensor(Bd, device=dev)
        self.S = torch.as_tensor(S, device=dev)
        self.module = None
        if mode == "cudnn":
            # Tesseract blocks CI, GI, GF1, GO -> torch gate order i, f, g, o = GI, GF1, CI, GO
            order = [1, 2, 0, 3]
            Wf = np.concatenate([WT[:self.ni].T, WT[self.ni:].T], axis=1)   # [4ns, ni+ns]
            Wf = Wf[np.concatenate([np.arange(k * ns, (k + 1) * ns) for k in order])]
            bf = Bd[np.concatenate([np.arange(k * ns, (k + 1) * ns) for k in order])]
            mod = nn.LSTM(self.ni, ns, batch_first=True)
            with torch.no_grad():
                mod.weight_ih_l0.copy_(torch.as_tensor(Wf[:, :self.ni]))
                mod.weight_hh_l0.copy_(torch.as_tensor(Wf[:, self.ni:]))
                mod.bias_ih_l0.copy_(torch.as_tensor(bf))
                mod.bias_hh_l0.zero_()
            self.module = mod.to(dev).eval()


# ----------------------------------------------------------------------------------------
# Activations

def _quant(v):
    """ref.quant / NetworkIO::WriteTimeStepPart: clip to [-1, 1], round(127 v) half away."""
    x = torch.clamp(v, -1.0, 1.0) * 127.0
    return torch.sign(x) * torch.floor(x.abs() + 0.5)


def _lut(x, table):
    a = x.abs()
    xs = a * 256.0
    idx = torch.clamp(torch.floor(xs).long(), max=4094)
    v = table[idx] + (table[idx + 1] - table[idx]) * (xs - idx)
    return torch.where(a >= 4095.0 / 256.0, torch.ones_like(v), v)


# ----------------------------------------------------------------------------------------
# Engine

class TorchTess:
    def __init__(self, model_path, device="cpu", mode="exact"):
        if mode not in ("exact", "cudnn"):
            raise ValueError(f"mode must be 'exact' or 'cudnn', got {mode!r}")
        self.model = load(model_path)
        self.mode = mode
        self.device = torch.device(device)
        self.H = self.model.input_height
        self.int_mode = self.model.int_mode
        self.C = self.model.num_classes
        self.codec = decode.Codec(self.model)
        self.half = _conv_half(self.model)
        self.K = (2 * self.half[0] + 1) * (2 * self.half[1] + 1)
        kids = self.model.network.children
        dev = self.device
        conv_w = kids[1].children[1].weights
        self.conv = _Affine(*_mats([conv_w], self.int_mode, mode), dev)
        self.y = _Lstm(kids[3].children[0], self.int_mode, mode, dev)
        self.x1 = _Lstm(kids[4], self.int_mode, mode, dev)
        self.x2 = _Lstm(kids[5].children[0], self.int_mode, mode, dev)
        self.x3 = _Lstm(kids[6], self.int_mode, mode, dev)
        self.out = _Affine(*_mats([kids[7].weights], self.int_mode, mode), dev)
        self.nsy = self.y.ns
        self.tanh_t = torch.as_tensor(ref.TANH_T, dtype=torch.float32, device=dev)
        self.log_t = torch.as_tensor(ref.LOGISTIC_T, dtype=torch.float32, device=dev)
        # caches that depend only on the model, or on (H, W)
        self.scale = 1.0 / 127.0 if (mode == "cudnn" and self.int_mode) else 1.0
        self._pv = None                      # pad draws of the model's stream, scaled
        self._pad_cache = {}                 # W -> pad_table_fast(H, W), keep=True only
        ky = 2 * self.half[1] + 1
        k = torch.arange(self.K, device=dev)
        self._dxk = k // ky - self.half[0]   # slot k = (dx+hx)*ky + (dy+hy)
        self._dyk = k % ky - self.half[1]
        ys = torch.arange(self.H, device=dev)
        yn = ys[:, None] + self._dyk[None, :]
        self._outy = (yn < 0) | (yn >= self.H)                              # [H, K]

    # activations: LUT in exact mode, plain functions in cudnn mode
    def _tanh(self, x):
        if self.mode == "cudnn":
            return torch.tanh(x)
        v = _lut(x, self.tanh_t)
        return torch.where(x < 0, -v, v)

    def _sig(self, x):
        if self.mode == "cudnn":
            return torch.sigmoid(x)
        v = _lut(x, self.log_t)
        return torch.where(x < 0, 1.0 - v, v)

    def _act(self, v):
        """Activation quantisation between layers (exact int8 only)."""
        return _quant(v) if (self.mode == "exact" and self.int_mode) else v

    def _affine(self, x, m):
        return (x @ m.WT + m.Bd) * m.S

    # LSTM over [N, T, ni]; Ls is the (descending) length of each row
    def _lstm_exact(self, m, X, Ls):
        N, T, _ = X.shape
        ns = m.ns
        h = torch.zeros(N, ns, device=self.device)
        c = torch.zeros(N, ns, device=self.device)
        out = torch.zeros(N, T, ns, device=self.device)
        for t in range(T):
            n = int((Ls > t).sum())
            if n == 0:
                break
            pre = (X[:n, t] @ m.WinT + h[:n] @ m.WrecT + m.Bd) * m.S
            ci = self._tanh(pre[:, :ns])
            gi = self._sig(pre[:, ns:2 * ns])
            gf = self._sig(pre[:, 2 * ns:3 * ns])
            go = self._sig(pre[:, 3 * ns:])
            cn = torch.clamp(c[:n] * gf + ci * gi, -100.0, 100.0)
            o = self._act(self._tanh(cn) * go)
            c[:n] = cn
            h[:n] = o
            out[:n, t] = o
        return out

    def _lstm_cudnn(self, m, X, Ls):
        N, T, _ = X.shape
        lengths = torch.as_tensor(np.asarray(Ls, np.int64))
        packed = nn.utils.rnn.pack_padded_sequence(X, lengths, batch_first=True)
        out_p, _ = m.module(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out_p, batch_first=True, total_length=T)
        return out

    def _lstm(self, m, X, Ls):
        if self.mode == "cudnn":
            return self._lstm_cudnn(m, X, Ls)
        return self._lstm_exact(m, X, Ls)

    def _lstm_reversed(self, m, X, Ls):
        """Reversed layer: each row is flipped within its own length, run, flipped back."""
        N, T, F = X.shape
        t = np.arange(T)[None, :]
        L = np.asarray(Ls)[:, None]
        idx = torch.as_tensor(np.where(t < L, L - 1 - t, t), device=self.device)
        Y = self._lstm(m, X.gather(1, idx[:, :, None].expand(N, T, F)), Ls)
        return Y.gather(1, idx[:, :, None].expand(N, T, Y.shape[2]))

    def _lstm_y(self, seq):
        """Y-summarising LSTM over the Hp rows of each column: last step only."""
        Ls = np.full(seq.shape[0], seq.shape[1], np.int64)
        return self._lstm(self.y, seq, Ls)[:, -1]

    def _pad_table(self, W):
        """pad_table_fast for one width, cached per (H, W). Used only by the keep=True path."""
        t = self._pad_cache.get(W)
        if t is None:
            t = self._pad_cache[W] = pad_table_fast(self.model, self.H, W)
        return t

    def _pad_stream(self, n):
        """First n pad draws of the model's stream as float32, times the input scale.

        Every line draws from the same stream prefix in border order, so one array serves
        all (H, W). It grows geometrically and is cached on the device."""
        if self._pv is None or self._pv.shape[0] < n:
            have = 0 if self._pv is None else self._pv.shape[0]
            m = max(n, 2 * have, 4096)
            st = _minstd_states(self.model.sample_iteration, m)
            if self.int_mode:
                raw = 127.0 * 2.0 * st / _MINSTD_M - 127.0
                vals = np.where(raw >= 0, np.floor(raw + 0.5), -np.floor(-raw + 0.5))
            else:
                raw = 1.0 * 2.0 * st / _MINSTD_M - 1.0
                vals = raw.astype(np.float32).astype(np.float64)
            self._pv = torch.as_tensor(vals.astype(np.float32) * np.float32(self.scale),
                                       device=self.device)
        return self._pv

    def _windows(self, V, off, Wb, Wp):
        """Conv windows [c, H, Wp, K] for one chunk, with no per-line Python.

        V: flat scaled buffer [line pixels ..., 0]; off [c]: each line's start in V;
        Wb [c]: each line's width; Wp: chunk width (>= every Wb, a multiple of 3). Slot
        k = (dx+hx)*ky + (dy+hy). In-image slots read the padded image; out-of-image slots
        take the Convolve pad draws in border raster order; columns >= Wb are zero."""
        H, K = self.H, self.K
        hx, hy = self.half
        kx, ky = 2 * hx + 1, 2 * hy + 1
        dev = self.device
        c = int(Wb.shape[0])
        zero = V.shape[0] - 1
        # zero-padded image P [c, H+2hy, Wp+2hx], gathered from V
        r = torch.arange(H + 2 * hy, device=dev) - hy
        col = torch.arange(Wp + 2 * hx, device=dev) - hx
        ok = ((r >= 0) & (r < H))[None, :, None] & \
             ((col[None, :] >= 0) & (col[None, :] < Wb[:, None]))[:, None, :]
        rr = r.clamp(0, H - 1)
        cc = torch.minimum(col.clamp(min=0)[None, :], Wb[:, None] - 1)
        flat = off[:, None, None] + rr[None, :, None] * Wb[:, None, None] + cc[:, None, :]
        P = V[torch.where(ok, flat, zero)]
        win = P.unfold(1, ky, 1).unfold(2, kx, 1)                         # [c,H,Wp,ky,kx]
        win = win.permute(0, 1, 2, 4, 3).reshape(c, H, Wp, K)              # slot = ix*ky+iy
        # which slots leave the image, and the pad draws they take
        xs = torch.arange(Wp, device=dev)
        nx = xs[None, :, None] + self._dxk[None, None, :]                  # [1,Wp,K]
        outx = (nx < 0) | (nx >= Wb[:, None, None])                        # [c,Wp,K]
        out = self._outy[None, :, None, :] | outx[:, None, :, :]           # [c,H,Wp,K]
        valid = (xs[None, :] < Wb[:, None])[:, None, :, None]              # [c,1,Wp,1]
        drawn = out & valid
        fd = drawn.reshape(c, -1)
        pv = self._pad_stream(int(fd.sum(dim=1).max()))
        pos = (torch.cumsum(fd, dim=1) - fd.long()).clamp_(max=pv.shape[0] - 1)
        vals = pv[pos].reshape(c, H, Wp, K)
        return torch.where(drawn, vals, torch.where(valid, win, 0.0))

    @torch.inference_mode()
    def _stages(self, xs, keep=False):
        """xs: normalised float arrays [H, W_i] in input order. Returns a dict of lists in
        input order: probs (float32 [T_i, C]), and with keep=True every stage per line."""
        H = self.H
        Hp = H // 3
        n_all = len(xs)
        res = {"probs": [np.zeros((0, self.C), np.float32)] * n_all}
        if keep:
            for name in ("in", "pads", "conv", "pool", "ylstm", "x1", "x2", "x3"):
                res[name] = [None] * n_all
            for i in range(n_all):
                res["in"][i] = xs[i]
        lines = [i for i in range(n_all) if xs[i].shape[1] >= 3]   # shorter lines: no frames
        lines.sort(key=lambda i: -xs[i].shape[1])
        if not lines:
            return res
        Wl = [xs[i].shape[1] for i in lines]
        Ls = np.asarray([w // 3 for w in Wl], np.int64)
        B = len(lines)
        in_off = np.concatenate([[0], np.cumsum([H * w for w in Wl])]).astype(np.int64)
        V = np.concatenate([xs[i].ravel() for i in lines] + [np.zeros(1)])
        V = torch.as_tensor(V.astype(np.float32) * np.float32(self.scale), device=self.device)
        Wl_t = torch.as_tensor(np.asarray(Wl, np.int64), device=self.device)
        off_t = torch.as_tensor(in_off[:-1], device=self.device)

        Tmax = -(-Wl[0] // 3)
        YL = torch.zeros(B, Tmax, self.nsy, device=self.device)
        F = self.conv.WT.shape[1]
        for c0 in range(0, B, CHUNK):
            c1 = min(B, c0 + CHUNK)
            c = c1 - c0
            Wp = -(-Wl[c0] // 3) * 3
            Tc = Wp // 3
            win = self._windows(V, off_t[c0:c1], Wl_t[c0:c1], Wp)
            feat = self._act(self._tanh(self._affine(win, self.conv)))      # [c,H,Wp,16]
            pooled = feat.reshape(c, Hp, 3, Tc, 3, F).amax(dim=(2, 4))     # [c,Hp,Tc,16]
            seq = pooled.permute(0, 2, 1, 3).reshape(c * Tc, Hp, F)
            YL[c0:c1, :Tc] = self._lstm_y(seq).reshape(c, Tc, self.nsy)
            if keep:
                feat_np, pooled_np = feat.cpu().numpy(), pooled.cpu().numpy()
                for j in range(c):
                    b = c0 + j
                    i = lines[b]
                    res["conv"][i] = feat_np[j, :, :Wl[b]]
                    res["pool"][i] = pooled_np[j, :, :Ls[b]]
                    res["pads"][i] = self._pad_table(Wl[b])

        x1 = self._lstm(self.x1, YL, Ls)
        x2 = self._lstm_reversed(self.x2, x1, Ls)
        x3 = self._lstm(self.x3, x2, Ls)
        v = self._affine(x3, self.out)
        m = v.max(dim=-1, keepdim=True).values
        e = torch.exp(torch.clamp(v - m, -86.0, 0.0))
        probs = (e / e.sum(dim=-1, keepdim=True)).cpu().numpy()          # one transfer
        for b in range(B):
            res["probs"][lines[b]] = probs[b, :int(Ls[b])]
        if keep:
            YL_np, x1_np, x2_np, x3_np = (t.cpu().numpy() for t in (YL, x1, x2, x3))
            for b in range(B):
                i = lines[b]
                t = int(Ls[b])
                res["ylstm"][i] = YL_np[b, :t]
                res["x1"][i] = x1_np[b, :t]
                res["x2"][i] = x2_np[b, :t]
                res["x3"][i] = x3_np[b, :t]
        return res

    def _normalise(self, images):
        out = []
        for im in images:
            g = ref.scale_to_height(np.asarray(im, np.uint8), self.H)
            out.append(ref.normalise(g, self.int_mode))
        return out

    def recognize(self, images):
        """list of float32 [T_i, C] softmax arrays, in input order."""
        return self._stages(self._normalise(images))["probs"]

    def texts(self, images):
        return [decode.greedy(p, self.codec) for p in self.recognize(images)]
