"""Read a Tesseract 4/5 .traineddata file: the LSTM network, its unicharset and recoder.

Follows tesseract src/lstm/network.cpp (CreateFromFile), weightmatrix.cpp, lstm.cpp,
plumbing.cpp, convolve.cpp, reconfig.cpp, fullyconnected.cpp, ccutil/unicharcompress.h and
lstmrecognizer.cpp (DeSerialize). Pure Python + NumPy; nothing from Tesseract is executed.
"""

import struct
from dataclasses import dataclass, field

import numpy as np

TYPE_NAMES = [
    "Invalid", "Input", "Convolve", "Maxpool", "Parallel", "Replicated", "ParBidiLSTM",
    "DepParUDLSTM", "Par2dLSTM", "Series", "Reconfig", "RTLReversed", "TTBReversed",
    "XYTranspose", "LSTM", "SummLSTM", "Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu",
    "Linear", "Softmax", "SoftmaxNoCTC", "LSTMSoftmax", "LSTMBinarySoftmax", "TensorFlow",
    "Dropout",
]
PLUMBING = {"Parallel", "Replicated", "ParBidiLSTM", "DepParUDLSTM", "Par2dLSTM", "Series",
            "RTLReversed", "TTBReversed", "XYTranspose"}
FULLY = {"Logistic", "LinLogistic", "LinTanh", "Tanh", "Relu", "Linear", "Softmax", "SoftmaxNoCTC"}
LSTMS = {"LSTM", "SummLSTM", "LSTMSoftmax", "LSTMBinarySoftmax"}
NF_LAYER_SPECIFIC_LR = 64
TS_ENABLED = 1
COMPONENTS = {17: "lstm", 18: "lstm-punc-dawg", 19: "lstm-word-dawg", 20: "lstm-number-dawg",
              21: "lstm-unicharset", 22: "lstm-recoder", 23: "version"}


class Reader:
    def __init__(self, buf, off=0):
        self.b = buf
        self.o = off

    def take(self, fmt):
        v = struct.unpack_from("<" + fmt, self.b, self.o)
        self.o += struct.calcsize("<" + fmt)
        return v[0] if len(v) == 1 else v

    def string(self):
        n = self.take("I")
        s = bytes(self.b[self.o:self.o + n])
        self.o += n
        return s.decode("utf-8")

    def array(self, dtype, n):
        a = np.frombuffer(self.b, dtype=dtype, count=n, offset=self.o).copy()
        self.o += a.nbytes
        return a

    def remaining(self):
        return len(self.b) - self.o


@dataclass
class Weights:
    """One weight matrix: rows = outputs, cols = inputs + 1 (bias last).

    float models: w is float64 [no, ni+1].
    int8 models: q is int8 [no, ni+1] and scale is float64 [no] with the /127 Tesseract
    applies at load, so q * scale[:, None] is the float equivalent of each row.
    """
    int_mode: bool
    w: np.ndarray = None
    q: np.ndarray = None
    scale: np.ndarray = None
    off: int = None   # float models: byte offset of w inside the lstm component (tessgpu.tessnet.export)

    def dense(self):
        if self.int_mode:
            return self.q.astype(np.float64) * self.scale[:, None]
        return self.w

    @property
    def shape(self):
        return (self.q if self.int_mode else self.w).shape


def read_weights(r, training):
    mode = r.take("B")
    int_mode = bool(mode & 1)
    adam = bool(mode & 4)
    if not mode & 128:
        raise ValueError("old-format weight matrix (pre-double) not supported")
    if int_mode:
        d1, d2 = r.take("II")
        r.take("b")  # empty_ value
        q = r.array(np.int8, d1 * d2).reshape(d1, d2)
        n = r.take("I")
        scale = r.array(np.float64, n)[:d1] / 127.0
        return Weights(True, q=q, scale=scale)
    d1, d2 = r.take("II")
    r.take("d")
    off = r.o
    w = r.array(np.float64, d1 * d2).reshape(d1, d2)
    if training:  # updates_ (and Adam's dw_sq_sum_) follow in a training dump
        for _ in range(2 if adam else 1):
            e1, e2 = r.take("II")
            r.take("d")
            r.array(np.float64, e1 * e2)
    return Weights(False, w=w, off=off)


@dataclass
class Node:
    type: str
    name: str
    ni: int
    no: int
    flags: int = 0
    num_weights: int = 0
    training: bool = False
    children: list = field(default_factory=list)
    shape: tuple = None          # Input: (batch, height, width, depth, loss_type)
    half: tuple = None           # Convolve: (half_x, half_y)
    scale: tuple = None          # Maxpool/Reconfig: (x_scale, y_scale)
    gates: dict = None           # LSTM: CI, GI, GF1, GO[, GFS] -> Weights
    na: int = 0
    ns: int = 0
    is_2d: bool = False
    softmax: "Node" = None       # LSTMSoftmax built-in output layer
    weights: Weights = None      # FullyConnected

    def walk(self, depth=0):
        yield depth, self
        for c in self.children:
            yield from c.walk(depth + 1)
        if self.softmax is not None:
            yield from self.softmax.walk(depth + 1)

    def describe(self):
        lines = []
        for d, n in self.walk():
            extra = ""
            if n.type == "Input":
                extra = f" shape={n.shape}"
            elif n.type == "Convolve":
                extra = f" half={n.half}"
            elif n.type in ("Maxpool", "Reconfig"):
                extra = f" scale={n.scale}"
            elif n.type in LSTMS:
                g = n.gates["CI"]
                extra = f" ns={n.ns} na={n.na} 2d={n.is_2d} int8={g.int_mode}"
            elif n.weights is not None:
                extra = f" w={n.weights.shape} int8={n.weights.int_mode}"
            lines.append(f"{'  ' * d}{n.type} '{n.name}' ni={n.ni} no={n.no}{extra}")
        return "\n".join(lines)


def read_network(r):
    t = r.take("b")
    if t == 0:
        tname = r.string()
    else:
        tname = TYPE_NAMES[t]
    training = r.take("b") == TS_ENABLED
    r.take("b")  # needs_to_backprop
    flags, ni, no, nw = r.take("iiii")
    name = r.string()
    n = Node(tname, name, ni, no, flags, nw, training)
    if tname == "Input":
        n.shape = r.take("iiiii")
    elif tname == "Convolve":
        n.half = r.take("ii")
        n.no = ni * (2 * n.half[0] + 1) * (2 * n.half[1] + 1)
    elif tname in ("Maxpool", "Reconfig"):
        n.scale = r.take("ii")
        n.no = ni if tname == "Maxpool" else ni * n.scale[0] * n.scale[1]
    elif tname in PLUMBING:
        size = r.take("I")
        n.children = [read_network(r) for _ in range(size)]
        if flags & NF_LAYER_SPECIFIC_LR:
            k = r.take("I")
            r.array(np.float32, k)
        if tname == "Series":
            n.no = n.children[-1].no
        elif tname in ("RTLReversed", "TTBReversed", "XYTranspose"):
            n.no = n.children[0].no
        else:
            n.no = sum(c.no for c in n.children)
    elif tname in LSTMS:
        n.na = r.take("i")
        nf = no if tname == "LSTMSoftmax" else (
            int(np.ceil(np.log2(no))) if tname == "LSTMBinarySoftmax" else 0)
        n.gates = {}
        for gname in ("CI", "GI", "GF1", "GO", "GFS"):
            if gname == "GFS" and not n.is_2d:
                continue
            n.gates[gname] = read_weights(r, training)
            if gname == "CI":
                n.ns = n.gates["CI"].shape[0]
                n.is_2d = n.na - nf == ni + 2 * n.ns
        if tname in ("LSTMSoftmax", "LSTMBinarySoftmax"):
            n.softmax = read_network(r)
    elif tname in FULLY:
        n.weights = read_weights(r, training)
    elif tname == "Dropout":
        pass
    else:
        raise ValueError(f"unsupported layer type {tname}")
    return n


def read_unicharset(text):
    """lstm-unicharset text: count line, then one entry per line; 'NULL' is the space."""
    lines = text.split("\n")
    n = int(lines[0].strip())
    chars = []
    for line in lines[1:1 + n]:
        tok = line.split(" ", 1)[0]
        chars.append(" " if tok == "NULL" else tok)
    return chars


def read_recoder(buf):
    r = Reader(buf)
    n = r.take("I")
    codes = []
    for _ in range(n):
        r.take("b")  # self_normalized
        length = r.take("I")
        codes.append(tuple(int(x) for x in r.array(np.int32, length)))
    assert r.remaining() == 0, r.remaining()
    return codes


@dataclass
class Model:
    path: str
    version: str
    network: Node
    network_str: str
    null_char: int
    unicharset: list
    recoder: list           # unichar id -> tuple of codes
    components: dict
    int_mode: bool
    sample_iteration: int = 0
    training_flags: int = 0

    @property
    def simple_text(self):
        """LSTMRecognizer::SimpleTextOutput: no TF_COMPRESS_UNICHARSET (64) flag."""
        return (self.training_flags & 64) == 0

    @property
    def input_height(self):
        return self.network.children[0].shape[1]

    @property
    def num_classes(self):
        return self.network.no


def components(buf):
    n = struct.unpack_from("<I", buf, 0)[0]
    offs = struct.unpack_from(f"<{n}q", buf, 4)
    out = {}
    for i, o in enumerate(offs):
        if o < 0:
            continue
        nxt = next((offs[j] for j in range(i + 1, n) if offs[j] >= 0), len(buf))
        out[i] = buf[o:nxt]
    return out


def load(path):
    buf = open(path, "rb").read()
    comps = components(buf)
    if 17 not in comps:
        raise ValueError(f"{path}: no LSTM component")
    r = Reader(comps[17])
    net = read_network(r)
    include_charsets = not (21 in comps and 22 in comps)
    if include_charsets:
        raise ValueError("charsets embedded in the LSTM component are not supported")
    network_str = r.string()
    training_flags, training_iteration, sample_iteration, null_char = r.take("iiii")
    r.take("fff")  # learning_rate, momentum, adam_beta
    if r.remaining() != 0:
        raise ValueError(f"{path}: {r.remaining()} trailing bytes in the LSTM component")
    uni = read_unicharset(comps[21].decode("utf-8"))
    rec = read_recoder(comps[22])
    int_mode = any(n.gates["CI"].int_mode for _, n in net.walk() if n.type in LSTMS)
    version = comps.get(23, b"").decode("utf-8", "replace")
    return Model(path, version, net, network_str, null_char, uni, rec,
                 {COMPONENTS.get(k, str(k)): len(v) for k, v in comps.items()}, int_mode,
                 sample_iteration, training_flags)


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        m = load(p)
        print(p, m.version, m.network_str, "null_char", m.null_char,
              "unichars", len(m.unicharset), "codes", len(m.recoder), "int8", m.int_mode)
        print(m.network.describe())
