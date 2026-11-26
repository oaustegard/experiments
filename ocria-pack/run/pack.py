"""Pack OCRia-nano-1's model files into one bundle the page turns back into the files LiteParse loads.

Every float32 tensor of 1024 or more elements is stored as int8 per row (row = the first axis: a conv's
or a linear layer's output channel), symmetric, levels -L..L, with a float32 scale per row. Everything
else in the file (graph protobuf, small tensors, headers, rec_latin's own int8 tensors) is kept byte for
byte. Unpacking writes each dequantized row back into its place, so the result is the original file with
only the large weights rounded: same size, same layout, loadable by LiteParse's wasm runtime unchanged.

Where tensors are: an ONNX file's float initializers are found through the onnx package (their raw_data
located in the file in initializer order); tablenet.bin is "TNET", a u32 header length, a JSON header
listing each tensor's shape and element offset, then the float32 data.

Bundle: b"OCPK", u32 header length, JSON header, then the blob (kept bytes, int8 data, float32 scales).
Header: {"levels": L, "files": [{"name", "size", "kept": [[file_off, len, blob_off]...],
"q": [[file_off, rows, cols, q_off, s_off]...]}]}.

    python3 run/pack.py --models models --files det.onnx,router.onnx,... --levels 127 \
        --bundle packs/i8/ocria-latin.pack [--unpacked arms/i8]
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import numpy as np

MIN_ELEMS = 1024


def onnx_regions(path: Path) -> list[tuple[int, tuple[int, ...]]]:
    import onnx

    raw = path.read_bytes()
    m = onnx.load(str(path), load_external_data=False)
    out, pos = [], 0
    for t in m.graph.initializer:
        if not t.raw_data:
            continue
        at = raw.find(t.raw_data, pos)
        assert at >= 0, f"{path.name}: {t.name} raw_data not found"
        pos = at + len(t.raw_data)
        if t.data_type == onnx.TensorProto.FLOAT:
            out.append((at, tuple(t.dims)))
    return out


def tnet_regions(path: Path) -> list[tuple[int, tuple[int, ...]]]:
    raw = path.read_bytes()
    assert raw[:4] == b"TNET"
    n = struct.unpack("<I", raw[4:8])[0]
    head = json.loads(raw[8:8 + n])
    base = 8 + n
    return [(base + 4 * t["offset"], tuple(t["shape"])) for t in head["tensors"]]


def regions(path: Path):
    """Float tensors to quantize; none for a file that is neither ONNX nor TNET (a charset or meta JSON is
    carried whole, so one bundle holds everything a group of models needs)."""
    if path.suffix == ".bin":
        return tnet_regions(path)
    if path.suffix == ".onnx":
        return onnx_regions(path)
    return []


def pack(models: Path, files: list[str], levels: int):
    blob_kept, blob_q, blob_s = bytearray(), bytearray(), bytearray()
    head = {"levels": levels, "files": []}
    err = {}
    for name in files:
        raw = (models / name).read_bytes()
        qs = []
        for off, shape in regions(models / name):
            n = int(np.prod(shape)) if shape else 1
            if n < MIN_ELEMS:
                continue
            w = np.frombuffer(raw, "<f4", n, off).astype(np.float32)
            rows = shape[0] if len(shape) > 1 else 1
            cols = n // rows
            w2 = w.reshape(rows, cols)
            amax = np.abs(w2).max(axis=1)
            scale = np.where(amax > 0, amax / levels, 1.0).astype(np.float32)
            q = np.clip(np.rint(w2 / scale[:, None]), -levels, levels).astype(np.int8)
            deq = q.astype(np.float32) * scale[:, None]
            err[f"{name}:{off}"] = float(np.abs(deq - w2).max() / (np.abs(w2).max() + 1e-12))
            qs.append([off, rows, cols, len(blob_q), len(blob_s)])
            blob_q += q.tobytes()
            blob_s += scale.astype("<f4").tobytes()
        qs.sort()
        kept, cur = [], 0
        for off, rows, cols, *_ in qs:
            if off > cur:
                kept.append([cur, off - cur, len(blob_kept)])
                blob_kept += raw[cur:off]
            cur = off + 4 * rows * cols
        if cur < len(raw):
            kept.append([cur, len(raw) - cur, len(blob_kept)])
            blob_kept += raw[cur:]
        head["files"].append({"name": name, "size": len(raw), "kept": kept, "q": qs})
    # blob = kept | pad | q | pad | scales; offsets in the header are made absolute to the blob start
    def pad4(b: bytearray):
        b += b"\0" * (-len(b) % 4)
    pad4(blob_kept)
    q0 = len(blob_kept)
    pad4(blob_q)
    s0 = q0 + len(blob_q)
    for f in head["files"]:
        for e in f["q"]:
            e[3] += q0
            e[4] += s0
    hj = json.dumps(head, separators=(",", ":")).encode()
    hj += b" " * (-(8 + len(hj)) % 4)
    bundle = b"OCPK" + struct.pack("<I", len(hj)) + hj + bytes(blob_kept) + bytes(blob_q) + bytes(blob_s)
    return bundle, err


def unpack(bundle: bytes) -> dict[str, bytes]:
    assert bundle[:4] == b"OCPK"
    n = struct.unpack("<I", bundle[4:8])[0]
    head = json.loads(bundle[8:8 + n])
    blob = memoryview(bundle)[8 + n:]
    out = {}
    for f in head["files"]:
        buf = bytearray(f["size"])
        for off, ln, bo in f["kept"]:
            buf[off:off + ln] = blob[bo:bo + ln]
        for off, rows, cols, qo, so in f["q"]:
            q = np.frombuffer(blob, np.int8, rows * cols, qo).reshape(rows, cols)
            s = np.frombuffer(blob, "<f4", rows, so)
            buf[off:off + 4 * rows * cols] = (q.astype(np.float32) * s[:, None]).astype("<f4").tobytes()
        out[f["name"]] = bytes(buf)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", type=Path, required=True)
    ap.add_argument("--files", required=True)
    ap.add_argument("--levels", type=int, default=127)
    ap.add_argument("--bundle", type=Path, required=True)
    ap.add_argument("--unpacked", type=Path, help="also write the unpacked files (plus the other model files)")
    a = ap.parse_args()
    files = a.files.split(",")
    bundle, err = pack(a.models, files, a.levels)
    a.bundle.parent.mkdir(parents=True, exist_ok=True)
    a.bundle.write_bytes(bundle)
    import gzip

    gz = len(gzip.compress(bundle, 9))
    print(f"{a.bundle.name}: {len(bundle)} B, gzip {gz} B; max rel err {max(err.values()):.4f}")
    if a.unpacked:
        a.unpacked.mkdir(parents=True, exist_ok=True)
        files_out = unpack(bundle)
        for p in a.models.iterdir():
            if p.is_file() and p.name not in files_out:
                (a.unpacked / p.name).write_bytes(p.read_bytes())
        for name, b in files_out.items():
            (a.unpacked / name).write_bytes(b)


if __name__ == "__main__":
    main()
