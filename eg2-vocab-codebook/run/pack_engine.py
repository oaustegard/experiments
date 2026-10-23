"""Build the WGSL engine's weight file with an rhq token table (run/rhq.py) in place of the
3-bit grouped table, and the renumbered tokenizer that goes with it.

Usage: python3 -I run/pack_engine.py --model DIR --rows ROWS.npy --table TABLE.npy --n3 N --tokenizer TOK.json --out DIR
  --model  a dir with manifest.json + its data file (eg2-wgsl-engine/run/manifest.py output, t3w)
  --rows   original EG2 ids of the vocabulary (sorted), --table their float rows (HF scale)
  --n3     rows to store at 3 bits: the n3 most frequent (.work/freq.npy); added tokens always
The vocabulary is renumbered so the 3-bit rows come first (each tier in original-id order); the
tokenizer.json is rewritten to match (vocab, added tokens, post-processor ids). Every non-table
tensor is copied byte for byte. Writes model_rhq.bin, manifest.json, tokenizer.json, ids.npy
(new id -> original EG2 id).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import rhq  # noqa: E402

ALIGN = 256


def walk(m, fn, path=()):
    """Apply fn to every tensor spec (dict with offset/length) outside the table."""
    if isinstance(m, dict):
        if "offset" in m and "length" in m:
            fn(m)
            return
        for k, v in m.items():
            if path == () and k == "table":
                continue
            walk(v, fn, path + (k,))
    elif isinstance(m, list):
        for v in m:
            walk(v, fn, path + ("[]",))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--n3", type=int, required=True)
    ap.add_argument("--freq", default=str(Path(__file__).resolve().parent.parent / ".work" / "freq.npy"))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    src = Path(a.model)
    man = json.loads((src / "manifest.json").read_text())
    data = (src / man["data"]).read_bytes()
    rows = np.load(a.rows)
    X = np.load(a.table).astype(np.float32)
    assert len(X) == len(rows)
    tj = json.loads(Path(a.tokenizer).read_text())
    added = {t["id"] for t in tj["added_tokens"]}  # ids in the pruned tokenizer = positions in rows
    f = np.load(a.freq)[rows].astype(np.float64)
    pri = f.copy()
    pri[list(added)] = np.inf
    top = np.sort(np.argsort(-pri, kind="stable")[:a.n3])
    rest = np.setdiff1d(np.arange(len(rows)), top)
    perm = np.concatenate([top, rest])  # new id -> old (pruned) id
    newid = np.empty(len(rows), np.int64)
    newid[perm] = np.arange(len(rows))

    c3, s3 = rhq.encode(X[top], 3)
    c2, s2 = rhq.encode(X[rest], 2)
    out = bytearray()

    def put(b):
        nonlocal out
        out += b"\0" * ((-len(out)) % ALIGN)
        off = len(out)
        out += bytes(b)
        return off

    def remap(t):
        t["offset"] = put(data[t["offset"]:t["offset"] + t["length"]])
    walk(man, remap)
    signs = rhq.signs()
    sbits = np.packbits((signs > 0).astype(np.uint8), bitorder="little")
    tbl = {"kind": "rhq", "n": int(len(rows)), "n3": int(len(top)), "seed": 0x5EED}
    for name, arr, dt, dims in [("codes3", rhq.pack(c3, 3), "UINT8", [len(top), 192]),
                                ("codes2", rhq.pack(c2, 2), "UINT8", [len(rest), 128]),
                                ("scales", np.concatenate([s3, s2]).astype("<f2"), "FLOAT16", [len(rows)]),
                                ("signs", sbits, "UINT8", [64]),
                                ("levels3", rhq.lloyd_max(3).astype("<f4"), "FLOAT", [8]),
                                ("levels2", rhq.lloyd_max(2).astype("<f4"), "FLOAT", [4])]:
        b = np.ascontiguousarray(arr).tobytes()
        tbl[name] = {"offset": put(b), "length": len(b), "dims": dims, "dtype": dt}
    out += b"\0" * ((-len(out)) % 4)
    man["table"] = tbl
    man["data"] = "model_rhq.bin"
    man["source"] = None
    dst = Path(a.out)
    dst.mkdir(parents=True, exist_ok=True)
    (dst / "model_rhq.bin").write_bytes(out)
    (dst / "manifest.json").write_text(json.dumps(man, indent=1))
    np.save(dst / "ids.npy", rows[perm])
    np.save(dst / "table_decoded.npy", np.concatenate([rhq.decode(c3, s3, 3), rhq.decode(c2, s2, 2)]))

    inv = {i: t for t, i in tj["model"]["vocab"].items()}
    tj["model"]["vocab"] = {inv[int(o)]: int(n) for n, o in enumerate(perm)}
    for t in tj["added_tokens"]:
        t["id"] = int(newid[t["id"]])
    for sp in (tj.get("post_processor") or {}).get("special_tokens", {}).values():
        sp["ids"] = [int(newid[i]) for i in sp["ids"]]
    (dst / "tokenizer.json").write_text(json.dumps(tj, ensure_ascii=False))
    table_b = sum(tbl[k]["length"] for k in ("codes3", "codes2", "scales", "signs", "levels3", "levels2"))
    print(json.dumps({"rows": len(rows), "n3": len(top), "table_mb": table_b / 1e6, "file_mb": len(out) / 1e6}))


if __name__ == "__main__":
    main()
