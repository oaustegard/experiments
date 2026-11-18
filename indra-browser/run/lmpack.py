"""The n-gram tables (lm*.npz) made small: fewer orders, quantized values,
delta-coded keys. `arm` writes an npz in the package's own format holding
exactly the values the packed file decodes to, so the reference pipeline can
score it; `size` says what the packed file costs.

Packed layout per table (k_n with v_n, hk_n with hb_n, n < order): the sorted
uint64 keys as LEB128 varints of their gaps, then the values as b-bit codes
into a per-table codebook of 2^b float32 levels (1-D k-means, quantile start).

    python3 run/lmpack.py arm  lm.npz out/lm.npz --order 3 --vbits 4
    python3 run/lmpack.py size lm.npz --order 3 --vbits 4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def kmeans1d(x: np.ndarray, k: int, iters: int = 25) -> np.ndarray:
    """k levels for the values x (sorted Lloyd iterations); fewer if x has fewer distinct values."""
    u = np.unique(x)
    if len(u) <= k:
        return u.astype(np.float32)
    xs = np.sort(x.astype(np.float64))
    c = np.quantile(xs, (np.arange(k) + 0.5) / k)
    for _ in range(iters):
        edges = (c[1:] + c[:-1]) / 2
        idx = np.searchsorted(edges, xs)
        sums = np.bincount(idx, weights=xs, minlength=k)
        cnt = np.bincount(idx, minlength=k)
        nc = np.where(cnt > 0, sums / np.maximum(cnt, 1), c)
        if np.allclose(nc, c):
            break
        c = np.sort(nc)
    return c.astype(np.float32)


def quant(v: np.ndarray, bits: int):
    if bits >= 32 or len(v) == 0:
        return v.astype(np.float32), None, None
    lv = kmeans1d(v, 2 ** bits)
    edges = (lv[1:] + lv[:-1]) / 2
    codes = np.searchsorted(edges, v).astype(np.uint16)
    return lv[codes], codes, lv


def varint(d: np.ndarray) -> bytes:
    x = d.astype(np.uint64)
    nb = np.ones(len(x), np.int64)
    t = x >> np.uint64(7)
    while (t > 0).any():
        nb += t > 0
        t >>= np.uint64(7)
    buf = np.zeros(int(nb.sum()), np.uint8)
    pos = np.concatenate([[0], np.cumsum(nb)[:-1]]).astype(np.int64)
    for i in range(int(nb.max()) if len(nb) else 0):
        m = nb > i
        b = ((x[m] >> np.uint64(7 * i)) & np.uint64(0x7F)).astype(np.uint8)
        b |= np.where(nb[m] > i + 1, 0x80, 0).astype(np.uint8)
        buf[pos[m] + i] = b
    return buf.tobytes()


def pack_bits(codes: np.ndarray, bits: int) -> bytes:
    if codes is None or len(codes) == 0:
        return b""
    b = ((codes[:, None] >> np.arange(bits)) & 1).astype(np.uint8).ravel()
    return np.packbits(b, bitorder="little").tobytes()


def build(src: str, order: int | None, vbits: int, hbits: int | None = None):
    z = np.load(src)
    N = int(z["order"])
    order = min(order or N, N)
    hbits = vbits if hbits is None else hbits
    out = {"order": np.int64(order), "vocab_size": z["vocab_size"]}
    if "bits" in z:
        out["bits"] = z["bits"]
    blobs = {"keys": [], "vals": []}
    for n in range(order):
        for kk, vv, b in (("k", "v", vbits), ("hk", "hb", hbits)):
            k, v = z[f"{kk}{n}"], z[f"{vv}{n}"]
            fake, codes, lv = quant(v, b)
            out[f"{kk}{n}"], out[f"{vv}{n}"] = k, fake.astype(np.float32)
            blobs["keys"].append(varint(np.diff(np.concatenate([[0], k]).astype(np.uint64))))
            blobs["vals"].append((lv.tobytes() if lv is not None else b"") + (pack_bits(codes, b) if codes is not None else fake.astype(np.float32).tobytes()))
    return out, blobs


def size(blobs) -> dict:
    import brotli

    keys = b"".join(blobs["keys"])
    vals = b"".join(blobs["vals"])
    kb, vb = brotli.compress(keys, quality=9, lgwin=24), brotli.compress(vals, quality=9, lgwin=24)
    return {"keys_raw": len(keys), "vals_raw": len(vals), "keys_br": len(kb), "vals_br": len(vb),
            "total_raw": len(keys) + len(vals), "total_br": len(kb) + len(vb)}


def write(src: str, dst: str, order: int | None, vbits: int, hbits: int | None = None):
    """The packed file the browser reads (web/lm.js): a u32 header length, the JSON header,
    then per table its varint key gaps, its float32 levels and one byte per value code (or
    float32 values when unquantized), each 4-byte aligned."""
    z = np.load(src)
    N = int(z["order"])
    order = min(order or N, N)
    hbits = vbits if hbits is None else hbits
    body = bytearray()

    def put(b: bytes) -> int:
        while len(body) % 4:
            body.append(0)
        off = len(body)
        body.extend(b)
        return off

    tables = []
    for n in range(order):
        for kk, vv, b in (("k", "v", vbits), ("hk", "hb", hbits)):
            k, v = z[f"{kk}{n}"], z[f"{vv}{n}"]
            _, codes, lv = quant(v, b)
            t = {"name": f"{kk}{n}", "n": int(len(k)),
                 "keys": put(varint(np.diff(np.concatenate([[0], k]).astype(np.uint64))))}
            t["keys_len"] = len(body) - t["keys"]
            if codes is None:
                t["f32"] = put(v.astype(np.float32).tobytes())
            else:
                t["levels"] = put(lv.astype(np.float32).tobytes())
                t["nlevels"] = int(len(lv))
                t["codes"] = put(codes.astype(np.uint8).tobytes())
            tables.append(t)
    head = json.dumps({"format": "indra-lm/1", "order": order, "vocab_size": int(z["vocab_size"]),
                       "bits": int(z["bits"]) if "bits" in z else 11, "tables": tables}).encode()
    pad = (-(4 + len(head))) % 4
    with open(dst, "wb") as f:
        f.write(np.uint32(len(head) + pad).tobytes() + head + b" " * pad + bytes(body))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["arm", "size", "write"])
    ap.add_argument("src")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--order", type=int)
    ap.add_argument("--vbits", type=int, default=32)
    ap.add_argument("--hbits", type=int)
    a = ap.parse_args()
    if a.cmd == "write":
        write(a.src, a.dst, a.order, a.vbits, a.hbits)
        print(a.dst, Path(a.dst).stat().st_size)
        return
    out, blobs = build(a.src, a.order, a.vbits, a.hbits)
    if a.cmd == "arm":
        np.savez(a.dst, **out)
    print(json.dumps({"src": a.src, "order": int(out["order"]), "vbits": a.vbits, "hbits": a.hbits or a.vbits} | size(blobs)))


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------- trie format (indra-lm/2)

def _leb(vals) -> bytes:
    return varint(np.asarray(vals, np.uint64))


def _bits(codes: np.ndarray, b: int) -> bytes:
    if codes is None or len(codes) == 0:
        return b""
    u = codes.astype(np.uint32)
    return np.packbits(((u[:, None] >> np.arange(b)) & 1).astype(np.uint8).ravel(), bitorder="little").tobytes()


def write_trie(src: str, dst: str, order: int | None, vbits: int, hbits: int | None = None):
    """The tables as a trie, which the browser rebuilds into the same sorted keys:
    per history length n, the histories (hk_n) as one bit per (n-1)-gram key (every history of
    length n is a seen (n-1)-gram, but the all-line-start one, kept as an extra key), and the
    n-grams (k_n) as, per history in order, a count and the tokens' gaps; then the values as
    b-bit codes into a float32 codebook. Keys are about a third smaller than gap-coded keys."""
    z = np.load(src)
    N = int(z["order"])
    order = min(order or N, N)
    hbits = vbits if hbits is None else hbits
    bits = int(z["bits"]) if "bits" in z else 11
    mask = np.uint64((1 << bits) - 1)
    body = bytearray()

    def put(b: bytes) -> list:
        while len(body) % 4:
            body.append(0)
        off = len(body)
        body.extend(b)
        return [off, len(b)]

    tables = []
    for n in range(order):
        k = z[f"k{n}"].astype(np.uint64)
        hk = z[f"hk{n}"].astype(np.uint64)
        t = {"n": n, "nk": int(len(k)), "nhk": int(len(hk))}
        if n == 0:
            t["extras"] = [str(int(v)) for v in hk]
        else:
            prev = z[f"k{n-1}"].astype(np.uint64)
            sel = np.isin(prev, hk)
            t["bitmap"] = put(np.packbits(sel.astype(np.uint8), bitorder="little").tobytes())
            t["extras"] = [str(int(v)) for v in hk[~np.isin(hk, prev)]]
        hist = k >> np.uint64(bits)
        tok = (k & mask).astype(np.int64)
        pos = np.searchsorted(hk, hist)
        assert (hk[pos] == hist).all()
        cnt = np.bincount(pos, minlength=len(hk))
        gaps = tok.copy()
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        first = np.zeros(len(tok), bool)
        first[starts[cnt > 0]] = True
        gaps[~first] = tok[~first] - tok[np.where(~first)[0] - 1]
        assert (gaps > 0).all()
        t["counts"] = put(_leb(cnt))
        t["gaps"] = put(_leb(gaps))
        for name, vals, b in (("v", z[f"v{n}"], vbits), ("hb", z[f"hb{n}"], hbits)):
            _, codes, lv = quant(vals, b)
            if codes is None:
                t[name] = {"f32": put(vals.astype(np.float32).tobytes())}
            else:
                t[name] = {"levels": put(lv.astype(np.float32).tobytes()), "bits": b, "codes": put(_bits(codes, b))}
        tables.append(t)
    head = json.dumps({"format": "indra-lm/2", "order": order, "vocab_size": int(z["vocab_size"]),
                       "bits": bits, "tables": tables}).encode()
    pad = (-(4 + len(head))) % 4
    with open(dst, "wb") as f:
        f.write(np.uint32(len(head) + pad).tobytes() + head + b" " * pad + bytes(body))


def read_trie(path: str) -> dict:
    """Back to {k_n, hk_n, v_n, hb_n} (the check that write_trie loses nothing but value precision)."""
    raw = open(path, "rb").read()
    hl = int(np.frombuffer(raw[:4], np.uint32)[0])
    head = json.loads(raw[4:4 + hl])
    body = raw[4 + hl:]
    bits = head["bits"]

    def leb(off, ln, count):
        out = np.zeros(count, np.uint64)
        b = body[off:off + ln]
        i = p = 0
        while i < count:
            v = s = 0
            while True:
                c = b[p]; p += 1
                v |= (c & 0x7F) << s; s += 7
                if c < 0x80:
                    break
            out[i] = v
            i += 1
        return out

    def vals(spec, count):
        if "f32" in spec:
            o, ln = spec["f32"]
            return np.frombuffer(body[o:o + ln], np.float32)
        o, ln = spec["levels"]
        lv = np.frombuffer(body[o:o + ln], np.float32)
        o, ln = spec["codes"]
        bb = np.unpackbits(np.frombuffer(body[o:o + ln], np.uint8), bitorder="little")[: count * spec["bits"]]
        codes = (bb.reshape(count, spec["bits"]).astype(np.int64) << np.arange(spec["bits"])).sum(1)
        return lv[codes]

    out = {}
    prev = None
    for t in head["tables"]:
        n = t["n"]
        extras = np.array([int(e) for e in t["extras"]], np.uint64)
        if n == 0:
            hk = extras
        else:
            o, ln = t["bitmap"]
            sel = np.unpackbits(np.frombuffer(body[o:o + ln], np.uint8), bitorder="little")[: len(prev)].astype(bool)
            hk = np.sort(np.concatenate([prev[sel], extras]))
        cnt = leb(*t["counts"], t["nhk"]).astype(np.int64)
        gaps = leb(*t["gaps"], t["nk"]).astype(np.int64)
        hist = np.repeat(hk, cnt)
        starts = np.concatenate([[0], np.cumsum(cnt)[:-1]])
        grp = np.repeat(np.arange(len(hk)), cnt)
        tok = np.cumsum(gaps)
        base = np.concatenate([[0], tok])[starts]  # cumulative sum before each group
        tok = tok - base[grp]
        k = (hist << np.uint64(bits)) | tok.astype(np.uint64)
        out[f"k{n}"], out[f"hk{n}"] = k, hk
        out[f"v{n}"], out[f"hb{n}"] = vals(t["v"], t["nk"]), vals(t["hb"], t["nhk"])
        prev = k
    return out
