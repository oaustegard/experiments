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
