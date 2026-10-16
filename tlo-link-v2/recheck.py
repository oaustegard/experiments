"""Recheck the numbers RESULTS.md quotes against the data, in a few seconds.
Exit 1 if any quoted number no longer matches. Rebuild data/ with corpus.py and
bench2.py/bench3.py first if the corpus or dictionary changed."""
import base64, gzip, json, math, re, statistics, sys, zlib
from pathlib import Path
import codec_v2 as c

HERE = Path(__file__).parent
corpus = json.loads((HERE / "data/corpus.json").read_text())
test = [d["text"] for d in corpus["test"]]
V1, V2 = c.DICTS["y"], c.DICTS["w"]

def v1_payload(doc):  # today's site: shortest of y / z / r, base64url, "+" every 300
    raw = doc.encode()
    flag, data = min((("y", c.deflate(raw, V1)), ("z", c.deflate(raw, None)), ("r", raw)), key=lambda x: len(x[1]))
    body = flag + base64.urlsafe_b64encode(data).decode().rstrip("=")
    return "+".join(body[i:i + 300] for i in range(0, len(body), 300))

v1 = [len(v1_payload(d)) for d in test]
v2 = [len(c.encode(d)) for d in test]
v1d_b62 = [len(c.encode(d).replace("+", "")) for d in test]
got = {
    "test_docs": len(test),
    "train_docs": len(corpus["train"]),
    "v1_chars": sum(v1),
    "v2_chars": sum(v2),
    "v2_vs_v1_pct": round(100 * (sum(v2) - sum(v1)) / sum(v1), 1),
    "median_doc_pct": round(statistics.median(100 * (b - a) / a for a, b in zip(v1, v2)), 1),
    "dict_gz_kb_added": round((len(gzip.compress(V2, 9)) - len(gzip.compress(V1, 9))) / 1024, 1),
    "b62_overhead_pct": round(100 * (43 / (32 * 8 / 6) - 1), 2),  # 43 chars vs 42.67 base64 chars per 32 bytes
}
claimed = json.loads(re.search(r"<!-- recheck (\{.*?\}) -->", (HERE / "RESULTS.md").read_text(), re.S).group(1))
bad = {k: (claimed.get(k), v) for k, v in got.items() if claimed.get(k) != v}
print(json.dumps(got))
if bad:
    print("MISMATCH (claimed, measured):", bad)
    sys.exit(1)
print("RESULTS.md numbers match the data")
