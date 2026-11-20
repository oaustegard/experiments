"""Tesseract's float LSTM network as a trainable PyTorch module, and export back into a .traineddata file.

    from tessgpu.traineddata import load
    from tessgpu.tessnet import TessNet, encode_text, export
    m = load("models/best/eng.traineddata")
    net = TessNet(m)
    logp, Ls = net([x1, x2, ...])         # x_i: float32 [H, W_i], normalised as ref.normalise(float mode)
    # logp [T, B, C] log-softmax, Ls [B] frames (W_i // 3); CTC blank = m.null_char
    export(net, m, "out.traineddata")     # same file layout, trained weights patched in

Topology (all tessdata LSTM models): Input H -> Convolve 3x3 (out-of-image slots draw uniform [-1, 1], as
Tesseract's training and inference both fill them from TRand) -> Tanh FC to 16 -> Maxpool 3x3 -> y-summarising
LSTM -> forward LSTM -> reversed LSTM -> forward LSTM -> Softmax FC. LSTM gates map Tesseract CI, GI, GF1, GO onto
torch's g, i, f, o; Tesseract's lookup-table tanh and logistic become the plain functions (server/torch_engine.py
cudnn mode: same text as the exact mode on 100% of 1,024 eng best lines).

Export writes each weight as the file's own float64 value plus (trained float32 - initial float32), so a weight
that did not move is written back bit for bit; an untrained network exports a byte-identical file.
Float (tessdata_best) models only.
"""
import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ORDER = [1, 2, 0, 3]  # torch gate blocks i, f, g, o <- Tesseract blocks GI, GF1, CI, GO (CI=0, GI=1, GF1=2, GO=3)
GATES = ("CI", "GI", "GF1", "GO")


def _lstm_nodes(m):
    k = m.network.children
    return {"y": k[3].children[0], "x1": k[4], "x2": k[5].children[0], "x3": k[6]}


def _to_torch_lstm(node):
    ns = node.ns
    W = np.concatenate([node.gates[g].w for g in GATES])          # [4ns, ni+ns+1], blocks CI, GI, GF1, GO
    W = W[np.concatenate([np.arange(b * ns, (b + 1) * ns) for b in ORDER])]
    ni = W.shape[1] - ns - 1
    mod = nn.LSTM(ni, ns, batch_first=True)
    with torch.no_grad():
        mod.weight_ih_l0.copy_(torch.as_tensor(W[:, :ni], dtype=torch.float32))
        mod.weight_hh_l0.copy_(torch.as_tensor(W[:, ni:ni + ns], dtype=torch.float32))
        mod.bias_ih_l0.copy_(torch.as_tensor(W[:, -1], dtype=torch.float32))
        mod.bias_hh_l0.zero_()
    return mod


class TessNet(nn.Module):
    def __init__(self, m):
        super().__init__()
        if m.int_mode:
            raise ValueError("TessNet trains float (tessdata_best) models only")
        k = m.network.children
        self.H = m.input_height
        cw = k[1].children[1].weights.w                              # [16, 10], slot k = ix*3 + iy, bias last
        self.conv = nn.Linear(cw.shape[1] - 1, cw.shape[0])
        ow = k[7].weights.w                                          # [C, ns+1]
        self.out = nn.Linear(ow.shape[1] - 1, ow.shape[0])
        with torch.no_grad():
            self.conv.weight.copy_(torch.as_tensor(cw[:, :-1], dtype=torch.float32))
            self.conv.bias.copy_(torch.as_tensor(cw[:, -1], dtype=torch.float32))
            self.out.weight.copy_(torch.as_tensor(ow[:, :-1], dtype=torch.float32))
            self.out.bias.copy_(torch.as_tensor(ow[:, -1], dtype=torch.float32))
        nodes = _lstm_nodes(m)
        self.y, self.x1, self.x2, self.x3 = (_to_torch_lstm(nodes[n]) for n in ("y", "x1", "x2", "x3"))
        self.null_char = m.null_char
        self.init = {n: p.detach().clone() for n, p in self.named_parameters()}

    def forward(self, xs, generator=None, pads=None):
        """xs: list of float32 arrays or tensors [H, W_i], any order. Returns (logp [T, B, C], Ls [B], order):
        batch row b holds input order[b]; rows are sorted by width, longest first.
        pads (checks only): per input, Tesseract's pad table (server.torch_engine.pad_table_fast) to use for the
        out-of-image slots instead of random draws."""
        dev = self.conv.weight.device
        order = sorted(range(len(xs)), key=lambda i: -xs[i].shape[1])
        Ws = [int(xs[i].shape[1]) for i in order]
        B, H, Wm = len(xs), self.H, Ws[0]
        X = torch.rand(B, H + 2, Wm + 2, generator=generator, device=dev) * 2 - 1
        for b, i in enumerate(order):
            X[b, 1:H + 1, 1:Ws[b] + 1] = torch.as_tensor(xs[i], dtype=torch.float32, device=dev)
        win = X.unfold(1, 3, 1).unfold(2, 3, 1)                       # [B, H, Wm, 3 (y), 3 (x)]
        win = win.permute(0, 1, 2, 4, 3).reshape(B, H, Wm, 9)          # slot = ix*3 + iy
        if pads is not None:
            win = win.clone()
            for b, i in enumerate(order):
                W = Ws[b]
                yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
                border = (yy < 1) | (yy >= H - 1) | (xx < 1) | (xx >= W - 1)
                by, bx = yy[border], xx[border]
                d = np.arange(-1, 2)
                ox = (bx[:, None] + d[None, :] < 0) | (bx[:, None] + d[None, :] >= W)
                oy = (by[:, None] + d[None, :] < 0) | (by[:, None] + d[None, :] >= H)
                drawn = torch.as_tensor((ox[:, :, None] | oy[:, None, :]).reshape(len(by), 9))
                cur = win[b, torch.as_tensor(by), torch.as_tensor(bx)]
                tab = torch.as_tensor(pads[i], dtype=torch.float32, device=dev)
                win[b, torch.as_tensor(by), torch.as_tensor(bx)] = torch.where(drawn, tab, cur)
        feat = torch.tanh(self.conv(win))
        T = Wm // 3
        Hp = H // 3
        pooled = feat[:, :Hp * 3, :T * 3].reshape(B, Hp, 3, T, 3, -1).amax(dim=(2, 4))   # [B, Hp, T, 16]
        seq = pooled.permute(0, 2, 1, 3).reshape(B * T, Hp, -1)
        yo = self.y(seq)[0][:, -1].reshape(B, T, -1)
        Ls = torch.as_tensor([w // 3 for w in Ws], dtype=torch.int64)
        x1 = self._run(self.x1, yo, Ls)
        x2 = self._reversed(self.x2, x1, Ls)
        x3 = self._run(self.x3, x2, Ls)
        logp = F.log_softmax(self.out(x3), dim=-1)
        return logp.transpose(0, 1), Ls, order

    @staticmethod
    def _run(mod, X, Ls):
        packed = nn.utils.rnn.pack_padded_sequence(X, Ls, batch_first=True, enforce_sorted=True)
        out, _ = mod(packed)
        return nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=X.shape[1])[0]

    def _reversed(self, mod, X, Ls):
        B, T, Fd = X.shape
        t = torch.arange(T)[None, :]
        L = Ls[:, None]
        idx = torch.where(t < L, L - 1 - t, t).to(X.device)
        Y = self._run(mod, X.gather(1, idx[:, :, None].expand(B, T, Fd)), Ls)
        return Y.gather(1, idx[:, :, None].expand(B, T, Y.shape[2]))

    # ---------------------------------------------------------------- export

    def _delta(self, name):
        return (dict(self.named_parameters())[name].detach().cpu() - self.init[name].cpu()).double().numpy()

    def tess_updates(self, m):
        """[(Weights object, new float64 [no, ni+1])] for every weight matrix in the network."""
        k = m.network.children
        ups = []
        cw = k[1].children[1].weights
        d = np.concatenate([self._delta("conv.weight"), self._delta("conv.bias")[:, None]], axis=1)
        ups.append((cw, cw.w + d))
        ow = k[7].weights
        d = np.concatenate([self._delta("out.weight"), self._delta("out.bias")[:, None]], axis=1)
        ups.append((ow, ow.w + d))
        for key, node in _lstm_nodes(m).items():
            ns = node.ns
            dih = self._delta(f"{key}.weight_ih_l0")
            dhh = self._delta(f"{key}.weight_hh_l0")
            db = self._delta(f"{key}.bias_ih_l0") + self._delta(f"{key}.bias_hh_l0")
            for tb, tess_block in enumerate(ORDER):
                g = node.gates[GATES[tess_block]]
                rows = slice(tb * ns, (tb + 1) * ns)
                d = np.concatenate([dih[rows], dhh[rows], db[rows][:, None]], axis=1)
                ups.append((g, g.w + d))
        return ups


def export(net, m, out_path):
    buf = bytearray(open(m.path, "rb").read())
    n = struct.unpack_from("<I", buf, 0)[0]
    base = struct.unpack_from(f"<{n}q", buf, 4)[17]
    for w, new in net.tess_updates(m):
        if w.off is None or new.shape != w.w.shape:
            raise ValueError("weight matrix without an offset or with a changed shape")
        b = np.ascontiguousarray(new, dtype="<f8").tobytes()
        buf[base + w.off: base + w.off + len(b)] = b
    with open(out_path, "wb") as f:
        f.write(buf)


def encode_text(text, m, _cache={}):
    """Text -> CTC target codes: greedy longest match over the unicharset, each unichar through the recoder.
    Returns None when a character is not in the unicharset."""
    key = id(m)
    if key not in _cache:
        by_str = {s: i for i, s in enumerate(m.unicharset) if i >= 3 or i == 0}
        _cache[key] = (by_str, max(len(s) for s in by_str))
    by_str, longest = _cache[key]
    ids, i = [], 0
    while i < len(text):
        for L in range(min(longest, len(text) - i), 0, -1):
            u = by_str.get(text[i:i + L])
            if u is not None:
                ids.append(u)
                i += L
                break
        else:
            return None
    codes = []
    for u in ids:
        codes.extend(m.recoder[u])
    return codes
