"""Check RESULTS.md's quoted numbers against the committed artifacts.

Sub-minute fixture per the repo's trust conventions: every headline constant
in the prose is re-read from the results JSONs, so the writeup and the data
cannot drift apart. Run from this directory: ``python3 recheck.py``.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FAIL = 0


def check(label: str, cond: bool) -> None:
    global FAIL
    print(("PASS  " if cond else "FAIL  ") + label)
    if not cond:
        FAIL += 1


def get(rows, **kw):
    m = [r for r in rows if all(r.get(k) == v for k, v in kw.items())]
    assert len(m) == 1, (kw, len(m))
    return m[0]


pb = json.load(open(HERE / "results_partb_leaf.json"))
lat = json.load(open(HERE / "results_latency_leaf.json"))
vb = json.load(open(HERE / "results_vs_bekko.json"))

R = pb["retrieval"]
check("n = 179 per distribution", pb["n"] == 179)

# retrieval table, full width
check("jina blog R@10 = 0.631", abs(get(R, model="jina-v5-nano-q4", dist="blog", dim=768)["r@10"] - 0.631) < 5e-4)
check("jina code R@10 = 0.978", abs(get(R, model="jina-v5-nano-q4", dist="code", dim=768)["r@10"] - 0.978) < 5e-4)
check("jina code R@1 = 0.888", abs(get(R, model="jina-v5-nano-q4", dist="code", dim=768)["r@1"] - 0.888) < 5e-4)
check("leaf fp32 blog R@10 = 0.564", abs(get(R, model="leaf-mt-fp32", dist="blog", dim=1024)["r@10"] - 0.564) < 5e-4)
check("leaf fp32 code R@10 = 0.866", abs(get(R, model="leaf-mt-fp32", dist="code", dim=1024)["r@10"] - 0.866) < 5e-4)
check("leaf fp32 code R@1 = 0.581", abs(get(R, model="leaf-mt-fp32", dist="code", dim=1024)["r@1"] - 0.581) < 5e-4)
check("leaf int8 blog R@10 = 0.542", abs(get(R, model="leaf-mt-int8", dist="blog", dim=1024)["r@10"] - 0.542) < 5e-4)
check("leaf int8 code R@10 = 0.872", abs(get(R, model="leaf-mt-int8", dist="code", dim=1024)["r@10"] - 0.872) < 5e-4)
check("leaf q4 code R@10 = 0.838", abs(get(R, model="leaf-mt-q4", dist="code", dim=1024)["r@10"] - 0.838) < 5e-4)

# model sizes
check("leaf int8 = 23.7 MB", pb["model_mb"]["leaf-mt-int8"] == 23.7)
check("leaf fp32 = 89.1 MB", pb["model_mb"]["leaf-mt-fp32"] == 89.1)
check("jina = 131.6 MB", pb["model_mb"]["jina-v5-nano-q4"] == 131.6)

# headline paired claims
c = get(pb["claims"], claim="leaf fp32 @1024 vs jina @768, code R@1")
check("code R@1 delta = -0.307, 2/57, p<1e-5",
      abs(c["delta"] + 0.307) < 5e-4 and c["wins"] == 2 and c["losses"] == 57 and c["p"] < 1e-5)
c = get(pb["claims"], claim="leaf fp32 @1024 vs jina @768, code R@10")
check("code R@10 delta = -0.112, p<1e-4", abs(c["delta"] + 0.112) < 5e-4 and c["p"] < 1e-4)
c = get(pb["claims"], claim="leaf fp32 @1024 vs jina @768, blog R@10")
check("blog R@10 delta = -0.067, p=0.036", abs(c["delta"] + 0.067) < 5e-4 and abs(c["p"] - 0.036) < 5e-3)

# fidelity
f = get(pb["fidelity"], model="leaf-mt-int8", dist="blog")
check("int8 per-doc cosine ~0.989 (blog)", abs(f["per_doc_cosine_vs_fp32"] - 0.989) < 1e-3)

# latency table
def L(model, threads):
    return get(lat, model=model, threads=threads)

check("1t leaf int8 query = 7.3 ms", abs(L("leaf-mt-int8", 1)["query_ms"] - 7.3) < 0.05)
check("1t jina query = 140.0 ms", abs(L("jina-v5-nano-q4", 1)["query_ms"] - 140.0) < 0.05)
check("19.2x = jina/leaf-int8 1t", abs(L("jina-v5-nano-q4", 1)["query_ms"] / L("leaf-mt-int8", 1)["query_ms"] - 19.2) < 0.05)
check("1t leaf int8 tokens/s = 9055", abs(L("leaf-mt-int8", 1)["tokens_per_s"] - 9055) < 1)
check("1t q4 slower than fp32 (18.3 vs 15.4)",
      L("leaf-mt-q4", 1)["query_ms"] > L("leaf-mt-fp32", 1)["query_ms"])

# vs-bekko: paired tie + same-session latency
for cl in vb["claims"]:
    check(f"not significant: {cl['claim']}", not cl["significant"])
b = get(vb["retrieval"], model="bekko-a8m", dist="blog")
check("bekko-a8m blog R@10 reproduces 0.575", abs(b["r@10"] - 0.575) < 5e-4)
b = get(vb["retrieval"], model="bekko-a8m", dist="code")
check("bekko-a8m code R@10 reproduces 0.888", abs(b["r@10"] - 0.888) < 5e-4)
lb = {r["model"]: r["query_ms"] for r in vb["latency"]}
check("same-session 8.0 vs 10.8 ms", abs(lb["leaf-mt-int8"] - 8.0) < 0.05 and abs(lb["bekko-a8m"] - 10.8) < 0.05)

# negative control: a claim that SHOULD be significant, inverted
c = get(pb["claims"], claim="leaf fp32 @1024 vs jina @768, code R@1")
check("negative control: decisive cell is not 'noise'", c["significant"])

# codec head-to-head (bench_headtohead_leaf.py)
hh = json.load(open(HERE / "results_headtohead_leaf.json"))
HR, HC = hh["rows"], hh["claims"]
check("remax k=1 blog @128B = 0.503",
      abs(get(HR, dist="blog", arm="remax", dim=1024, param=1)["r@10"] - 0.503) < 5e-4)
check("vendor binary blog @128B = 0.547",
      abs(get(HR, dist="blog", arm="vendor-binary", dim=1024)["r@10"] - 0.547) < 5e-4)
check("binary-asym blog @128B = 0.559 (panel max)",
      abs(get(HR, dist="blog", arm="binary-asym", dim=1024)["r@10"] - 0.559) < 5e-4)
check("binary-asym d=512 blog = 0.542 = fp32 full",
      abs(get(HR, dist="blog", arm="binary-asym", dim=512)["r@10"]
          - get(HR, dist="blog", arm="matryoshka-fp32", dim=1024)["r@10"]) < 5e-4)
c = get(HC, claim="[blog] remax k=1 vs vendor binary @128B (d=1024)")
check("remax vs vendor binary blog: -0.045, n.s.",
      abs(c["delta"] + 0.045) < 5e-4 and not c["significant"])
c = get(HC, claim="[blog] remex 2-bit @1024 (~260B) vs MRL d=64 fp32 (256B)")
check("quantize-before-truncate blog: +0.073, p=0.015",
      abs(c["delta"] - 0.073) < 5e-4 and c["significant"])
c = get(HC, claim="[code] remex 2-bit @1024 (~260B) vs MRL d=64 fp32 (256B)")
check("quantize-before-truncate code: +0.117, p<1e-4",
      abs(c["delta"] - 0.117) < 5e-4 and c["p"] < 1e-4)
c = get(HC, claim="[blog] remex 2-bit @1024 (~260B) vs fp32 @1024 (4096B)")
check("remex 2-bit ~= uncompressed (blog, n.s.)", not c["significant"])
check("remax stack beats width at 128B: d=512 k=2 > d=1024 k=1 (blog)",
      get(HR, dist="blog", arm="remax", dim=512, param=2)["r@10"]
      > get(HR, dist="blog", arm="remax", dim=1024, param=1)["r@10"])

print(f"\n{FAIL} failure(s)")
sys.exit(1 if FAIL else 0)
