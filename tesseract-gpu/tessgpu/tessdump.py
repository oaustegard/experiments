"""Read the .bin files written by the jailed tess_dump harness (Tesseract 5.3.4's own
LSTMRecognizer::RecognizeLine + RecodeBeamSearch, no dictionary).

Layout: int32 T, int32 C, float32 probs[T*C], int32 n, int32 unichar_ids[n],
then int32 TI, int32 FI, int32 int_mode, float32 inputs[TI*FI] (row-major over y, x).
"""
import numpy as np


def read(path):
    b = open(path, "rb").read()
    t, c = np.frombuffer(b, np.int32, 2)
    probs = np.frombuffer(b, np.float32, t * c, 8).reshape(t, c)
    o = 8 + 4 * t * c
    n = int(np.frombuffer(b, np.int32, 1, o)[0])
    ids = np.frombuffer(b, np.int32, n, o + 4).tolist()
    o += 4 + 4 * n
    out = {"probs": probs, "ids": ids}
    if o < len(b):
        ti, fi, im = np.frombuffer(b, np.int32, 3, o)
        out["inputs"] = np.frombuffer(b, np.float32, ti * fi, o + 12).reshape(ti, fi)
        out["int_mode"] = bool(im)
    return out


def read_dict(path):
    """Read a .dict.bin written by harness/tess_dump_dict (dictionary on and off, same outputs).

    Layout (int32 unless noted): T, C, float32 probs[T*C]; n0, ids_nodict[n0];
    n, ids_dict[n], xcoords[n]; nw, then per word: len, permuter, nbytes, utf8 text.
    """
    b = open(path, "rb").read()
    t, c = np.frombuffer(b, np.int32, 2)
    probs = np.frombuffer(b, np.float32, t * c, 8).reshape(t, c)
    o = 8 + 4 * t * c

    def ints(k):
        nonlocal o
        v = np.frombuffer(b, np.int32, k, o).tolist()
        o += 4 * k
        return v

    n0 = ints(1)[0]
    ids0 = ints(n0)
    n = ints(1)[0]
    ids = ints(n)
    xs = ints(n)
    nw = ints(1)[0]
    words = []
    for _ in range(nw):
        ln, perm, nb = ints(3)
        words.append({"len": ln, "perm": perm, "text": b[o:o + nb].decode("utf-8")})
        o += nb
    if o != len(b):
        raise ValueError(f"{path}: {len(b) - o} trailing bytes")
    return {"probs": probs, "ids_nodict": ids0, "ids": ids, "xcoords": xs, "words": words}
