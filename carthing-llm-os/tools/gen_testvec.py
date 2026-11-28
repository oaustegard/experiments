"""Write the test-vector blob the kernel checks itself against.

Weights and activations are quantized by gguf-py (the Python port of ggml's
reference quantizers that ships with llama.cpp), so the kernel's own
quantizers and dot products are checked against llama.cpp's formats, not
against themselves. Layout matches `testvec` in src/main.c.
"""
import struct
import sys
from pathlib import Path

import numpy as np
from gguf import GGMLQuantizationType as T
from gguf.quants import quantize, dequantize

ROWS, COLS = 1000, 1024   # 1000 rows: does not split evenly over 4 cores


def main(out: Path) -> None:
    rng = np.random.default_rng(0)
    w = (rng.standard_normal((ROWS, COLS)) * 0.05).astype(np.float32)
    x = rng.standard_normal(COLS).astype(np.float32)

    wq = quantize(w, T.Q4_0)                 # (ROWS, COLS/32*18) uint8
    xq = quantize(x[None, :], T.Q8_0)[0]     # (COLS/32*34,) uint8

    # llama.cpp's vec_dot_q4_0_q8_0: per block, integer sum times d_w * d_x.
    wd = dequantize(wq, T.Q4_0).astype(np.float64)   # (q - 8) * d_w
    xd = dequantize(xq[None, :], T.Q8_0)[0].astype(np.float64)
    yref = (wd @ xd).astype(np.float32)
    yf32 = (w.astype(np.float64) @ x.astype(np.float64)).astype(np.float32)

    parts = [w.tobytes(), wq.tobytes(), x.tobytes(), xq.tobytes(), yref.tobytes(), yf32.tobytes()]
    hdr_fmt = "<8s9I"
    off = struct.calcsize(hdr_fmt)
    offs = []
    for p in parts:
        off = (off + 15) & ~15
        offs.append(off)
        off += len(p)
    blob = bytearray(off)
    struct.pack_into(hdr_fmt, blob, 0, b"Q4TV0001", ROWS, COLS, *offs, off)
    for o, p in zip(offs, parts):
        blob[o:o + len(p)] = p
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(blob)
    print(f"{out}: {len(blob)} bytes, {ROWS}x{COLS}")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
