"""Benchmark payload codecs for Tiny Little Office links.

Measures, per held-out document, the characters a link payload takes under:
compression (deflate+v1 dict, deflate+trained dict of several sizes, zstd+trained
dict, brotli) x text encoding (base64url, base62 in 32-byte blocks).
"""
import json, math, sys, zlib
from collections import Counter
import brotli, zstandard

HERE = sys.argv[1]
corpus = json.load(open(f"{HERE}/corpus.json"))
train = [d["text"].encode() for d in corpus["train"]]
test = [(d["name"], d["text"].encode()) for d in corpus["test"]]
V1 = open(sys.argv[2], "rb").read()


def deflate(raw, zdict=None):
    c = zlib.compressobj(9, zlib.DEFLATED, -15, 9, **({"zdict": zdict} if zdict else {}))
    return c.compress(raw) + c.flush()


def train_dict(docs, size, kmin=6, kmax=48):
    """COVER-lite: pick substrings shared by many documents, score (len-3)*docfreq,
    greedily without overlap, best placed last (shortest back-reference distance)."""
    df = Counter()
    for d in docs:
        seen = set()
        for k in (6, 8, 12, 16, 24, 32, 48):
            if k < kmin or k > kmax:
                continue
            for i in range(0, max(0, len(d) - k + 1)):
                seen.add(d[i:i + k])
        df.update(seen)
    cands = sorted(((k, n) for k, n in df.items() if n >= 3), key=lambda x: -(len(x[0]) - 3) * x[1])
    chosen, total = [], 0
    for s, n in cands:
        if total >= size:
            break
        if any(s in c for c in chosen):
            continue
        chosen = [c for c in chosen if c not in s]
        chosen.append(s)
        total = sum(map(len, chosen))
    chosen.sort(key=lambda s: df[s] * (len(s) - 3))  # most valuable last
    blob = b"".join(chosen)
    return blob[-size:]


def b64len(n):
    return math.ceil(n * 8 / 6)


def b62len(n):
    full, rest = divmod(n, 32)
    return full * 43 + (math.ceil(rest * 8 / math.log2(62)) if rest else 0)


def best(raw, *codecs):
    return min(len(c(raw)) for c in codecs)


dicts = {sz: train_dict(train, sz) for sz in (2048, 4096, 8192, 16384, 32768)}
zd = {sz: zstandard.train_dictionary(sz, train * 4) for sz in (4096, 16384)}

schemes = {
    "v1: deflate + v1 dict (1.6 KB)": lambda r: best(r, lambda x: deflate(x, V1), deflate, lambda x: x),
    "deflate, no dict": lambda r: best(r, deflate, lambda x: x),
    "brotli q11 (built-in dict)": lambda r: len(brotli.compress(r, quality=11, lgwin=22)),
}
for sz, d in dicts.items():
    schemes[f"deflate + trained dict {sz//1024} KB"] = (lambda d: lambda r: best(r, lambda x: deflate(x, d), deflate))(d)
schemes["deflate + v1 + trained 16 KB (v1 words last)"] = lambda r: best(r, lambda x: deflate(x, (dicts[16384] + V1)[-32768:]))
for sz, d in zd.items():
    cc = zstandard.ZstdCompressor(level=22, dict_data=d, write_checksum=False, write_content_size=False, write_dict_id=False)
    schemes[f"zstd 22 + trained dict {sz//1024} KB"] = (lambda cc: lambda r: len(cc.compress(r)))(cc)

rows = []
base = None
for name, f in schemes.items():
    sizes = [f(r) for _, r in test]
    b64 = sum(b64len(s) for s in sizes)
    b62 = sum(b62len(s) for s in sizes)
    if base is None:
        base = b64
    rows.append((name, sum(sizes), b64, b62, 100 * (b64 - base) / base))
raw_total = sum(len(r) for _, r in test)
print(f"held-out: {len(test)} sample docs, {raw_total} bytes of TLO; dictionaries trained on {len(train)} template/test docs\n")
print(f"{'scheme':48} {'bytes':>6} {'b64 chars':>9} {'b62 chars':>9} {'vs v1':>7}")
for n, by, b64, b62, d in rows:
    print(f"{n:48} {by:6} {b64:9} {b62:9} {d:+6.1f}%")
json.dump({"dicts": {k: v.hex() for k, v in dicts.items()}}, open(f"{HERE}/dicts.json", "w"))
