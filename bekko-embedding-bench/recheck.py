#!/usr/bin/env python3
"""Check RESULTS.md prose against the committed result artifacts.

Sub-5-minute fixture, no network, no models: it re-derives every headline
number from results_*.json and asserts the claims the writeup actually makes.
Run after editing either the prose or the results.

    python3 recheck.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.environ.get("REMEX_ROOT", "/home/user/remex"))

HERE = Path(__file__).resolve().parent
FAIL: list[str] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"{'ok  ' if cond else 'FAIL'}  {label}{('  — ' + detail) if detail else ''}")
    if not cond:
        FAIL.append(label)


def mean(key: str, rows: list[dict]) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else float("nan")


def main() -> int:
    # ── the writeup must not contradict itself ──────────────────────────────
    # An earlier revision left the n=6 reversal in the top-of-file headline
    # while the body retracted it 100 lines down. A reader stops at the
    # headline, so guard the retracted phrasings by text.
    doc = (HERE / "RESULTS.md").read_text()
    for banned in ("Part A reverses the 2026-07-04/05 verdict",
                   "the decision gate passes"):
        check(f"RESULTS.md headline free of retracted claim: {banned[:40]!r}",
              banned not in doc)
    check("RESULTS.md headline states the n=59 outcome",
          "does not replicate" in doc and "n=59" in doc)

    # ── Part A ──────────────────────────────────────────────────────────────
    A = json.load(open(HERE / "results_parta.json"))
    inst = json.load(open(HERE / "instances.json"))

    ast8 = [r for r in A if r["mode"] == "ast" and r["variant"] == "a8m"]
    check("Part A instance set re-mined to n>=50", len(inst) >= 50, f"n={len(inst)}")
    check("grep baseline r@5 ~0.596 at n=59", abs(mean("rg_r5", ast8) - 0.596) < 0.01,
          f"{mean('rg_r5', ast8):.3f}")
    # THE HEADLINE REVERSED AT n=59. Guard it so the n=6 version cannot return.
    check("n=59: dense/a8m does NOT beat grep (n=6 headline retracted)",
          mean("dense_r5", ast8) <= mean("rg_r5", ast8) + 0.01,
          f"dense {mean('dense_r5', ast8):.3f} vs rg {mean('rg_r5', ast8):.3f}")
    check("RRF remains directionally best in every cell",
          all(mean("rrf_r5", [r for r in A if r["mode"] == m and r["variant"] == v])
              > mean("rg_r5", [r for r in A if r["mode"] == m and r["variant"] == v])
              for m in ("ast", "flat") for v in ("a8m", "a25m")))

    # gate now depends on the cell -- that IS the finding
    poor = [r for r in ast8 if r["n_idents"] == 0]
    rich = [r for r in ast8 if r["n_idents"] > 0]
    check("identifier-poor stratum is STILL n=1 at n=59 (base rate corroborated)",
          len(poor) == 1, f"{len(poor)}/{len(ast8)}")
    check("GATE FAILS on ast/a8m at n=59 (regresses on the rich stratum)",
          mean("dense_r5", rich) < mean("rg_r5", rich),
          f"rich dense {mean('dense_r5', rich):.3f} vs rg {mean('rg_r5', rich):.3f}")

    # encoder axis is now REAL (it was called noise at n=6)
    ast25 = [r for r in A if r["mode"] == "ast" and r["variant"] == "a25m"]
    check("encoder axis a25m>a8m is real at n=59 (reverses the n=6 call)",
          mean("dense_r5", ast25) > mean("dense_r5", ast8) + 0.03,
          f"{mean('dense_r5', ast25):.3f} vs {mean('dense_r5', ast8):.3f}")

    # ── code-trained encoder ────────────────────────────────────────────────
    CE = {r["model"]: r for r in json.load(open(HERE / "results_codeemb.json"))}
    check("code-trained encoder does NOT beat the general one at r@5",
          CE["jina-code"]["dense_r5"] <= CE["bekko-a25m"]["dense_r5"],
          f"jina-code {CE['jina-code']['dense_r5']:.3f} vs "
          f"bekko-a25m {CE['bekko-a25m']['dense_r5']:.3f}")
    check("all encoders cluster near the grep baseline",
          all(abs(CE[m]["dense_r5"] - CE[m]["rg_r5"]) < 0.08 for m in CE),
          " ".join(f"{m}={CE[m]['dense_r5']:.3f}" for m in CE))

    # ── quantization on CODE (not just the blog corpus) ─────────────────────
    Q = {(r["codec"], r["param"], r["dim"]): r
         for r in json.load(open(HERE / "results_code_quant.json"))}
    q1, q2 = Q[("remex", 1, 384)], Q[("remex", 2, 384)]
    f384, f64 = Q[("fp32", 32, 384)], Q[("fp32", 32, 64)]
    check("CODE: quantize-wide beats truncate-narrow (1-bit@48B vs fp32 d=64@256B)",
          q1["r@5"] > f64["r@5"] and q1["bytes"] < f64["bytes"],
          f"{q1['r@5']:.3f}@{q1['bytes']}B vs {f64['r@5']:.3f}@{f64['bytes']}B")
    check("CODE: 2-bit@96B matches uncompressed@1536B",
          abs(q2["r@5"] - f384["r@5"]) < 0.03,
          f"{q2['r@5']:.3f} vs {f384['r@5']:.3f}")
    check("CODE: 1-bit is NOT free vs full fp32 (honest limit)",
          q1["r@5"] < f384["r@5"], f"{q1['r@5']:.3f} vs {f384['r@5']:.3f}")
    check("CODE: best dense arm still only ties grep",
          max(r["r@5"] for r in Q.values()) < 0.70,
          f"best r@5 {max(r['r@5'] for r in Q.values()):.3f} vs grep 0.596")

    # ── Part B ──────────────────────────────────────────────────────────────
    B = json.load(open(HERE / "results_partb.json"))
    for f in B["fidelity"]:
        check(f"int8-embtable fidelity holds ({f['model']}/{f['dist']})",
              f["per_doc_cosine_vs_own_fp32"] > 0.9994,
              f"{f['per_doc_cosine_vs_own_fp32']:.5f}")
    check("second distribution was actually measured",
          {f["dist"] for f in B["fidelity"]} == {"blog", "code"})

    J = json.load(open(HERE / "results_jina.json"))
    jr = {(r["dist"], r["dim"]): r["r@10"] for r in J["retrieval"]}
    br = {(r["dist"], r["dim"]): r["r@10"]
          for r in B["retrieval"] if r["model"] == "bekko-a25m"}
    contested = [(d, k) for (d, k) in jr if (d, k) in br]
    jina_wins = sum(1 for key in contested if jr[key] > br[key])
    check("jina wins the large majority of iso-byte cells vs bekko-a25m",
          jina_wins >= len(contested) - 1, f"{jina_wins}/{len(contested)}")
    check("jina q4 and bekko-a8m are comparable size (size argument is moot)",
          abs(J["model_mb"] - 124.1) < 15, f"jina {J['model_mb']} MB vs bekko-a8m 124.1 MB")

    # ── compute ─────────────────────────────────────────────────────────────
    # The axis the iso-byte table cannot see, and the reason the Part B verdict
    # is a regime choice rather than a dominance.
    L = json.load(open(HERE / "results_latency.json"))
    lat = {(r["model"], r["threads"]): r for r in L}
    a1 = lat[("bekko-a8m", 1)]
    j1 = lat[("jina-v5-nano-q4", 1)]
    check("bekko-a8m is >=10x faster than jina on 1 vCPU (query path)",
          j1["query_ms"] / a1["query_ms"] >= 10,
          f"{j1['query_ms'] / a1['query_ms']:.1f}x "
          f"({a1['query_ms']:.1f} vs {j1['query_ms']:.1f} ms)")
    check("throughput ratio matches the ~12x FLOPs ratio (architectural, not q4)",
          8 <= a1["tokens_per_s"] / j1["tokens_per_s"] <= 16,
          f"{a1['tokens_per_s'] / j1['tokens_per_s']:.1f}x tokens/s")
    check("a25m is the middle rung, not dominated",
          j1["query_ms"] > lat[("bekko-a25m", 1)]["query_ms"] > a1["query_ms"])

    # the ladder: jina still owns the top quality rung, so 'just swap' is wrong
    blog = {(r["model"], r["dim"]): r["r@10"]
            for r in json.load(open(HERE / "results_partb.json"))["retrieval"]
            if r["dist"] == "blog" and "fp32" not in r["model"]}
    blog.update({(r["model"], r["dim"]): r["r@10"]
                 for r in json.load(open(HERE / "results_jina.json"))["retrieval"]
                 if r["dist"] == "blog"})
    check("jina still owns the top quality rung (verdict is a trade, not a flip)",
          max(v for (m, _), v in blog.items() if m == "jina-v5-nano-q4")
          > max(v for (m, _), v in blog.items() if m != "jina-v5-nano-q4"))

    # ── real query path ─────────────────────────────────────────────────────
    P = {r["model"]: r for r in json.load(open(HERE / "results_kbpath.json"))}
    pa, pj = P["bekko-a8m"], P["jina-v5-nano-q4"]
    check("as-shipped end-to-end speedup is FAR below the encoder-only 12.9x",
          2.0 <= pj["search_ms"] / pa["search_ms"] <= 3.0,
          f"{pj['search_ms'] / pa['search_ms']:.1f}x end-to-end vs 12.9x isolated")
    check("the binarizer constant, not encode, dominates bekko's shipped query",
          pa["simhash_ms"] > pa["encode_ms"] * 3,
          f"simhash {pa['simhash_ms']:.1f} ms vs encode {pa['encode_ms']:.1f} ms")
    check("hamming scan is negligible at corpus scale", pa["scan_ms"] < 1.0,
          f"{pa['scan_ms']:.3f} ms")

    F = json.load(open(HERE / "results_kbfix.json"))
    check("cached quantizer is behaviour-preserving (codes AND hits identical)",
          all(r["codes_identical"] and r["hits_identical"] for r in F))
    fa = [r for r in F if r["model"] == "bekko-a8m"]
    fj = [r for r in F if r["model"] != "bekko-a8m"]
    check("caching restores a double-digit end-to-end advantage",
          min(j["fixed_ms"] / a["fixed_ms"] for a, j in zip(fa, fj)) >= 10,
          " ".join(f"{a['query'].split()[0]}={j['fixed_ms'] / a['fixed_ms']:.1f}x"
                   for a, j in zip(fa, fj)))
    check("caching helps the incumbent too, just less",
          all(r["shipped_ms"] > r["fixed_ms"] for r in fj),
          f"jina typical {fj[1]['shipped_ms']:.0f} -> {fj[1]['fixed_ms']:.0f} ms")

    # ── projections (the RHT option) ────────────────────────────────────────
    D = {r["dim"]: r for r in json.load(open(HERE / "results_rht_dims.json"))}
    check("remax's own rht_rotation @rounds=2 reproduces its documented 1.5-1.8x",
          all(1.4 <= D[d]["haar_ms"] / D[d]["remax_rht_r2_ms"] <= 2.2 for d in D),
          " ".join(f"d{d}={D[d]['haar_ms'] / D[d]['remax_rht_r2_ms']:.2f}x" for d in D))
    check("remax_kb's separate srht_matrix is SLOWER than haar at every dim",
          all(D[d]["kb_srht_r3_ms"] > D[d]["haar_ms"] for d in D),
          " ".join(f"d{d}={D[d]['haar_ms'] / D[d]['kb_srht_r3_ms']:.2f}x" for d in D))
    check("rademacher is the cheapest option at v1's dim=256",
          D[256]["kb_rademacher_ms"] < min(D[256][k] for k in
              ("haar_ms", "remax_rht_r2_ms", "remax_rht_r3_ms", "kb_srht_r3_ms")),
          f"{D[256]['kb_rademacher_ms']:.1f} ms")

    R = json.load(open(HERE / "results_rht.json"))
    ret = [r for r in R if r["stage"] == "retrieval"]
    for model in {r["model"] for r in ret}:
        sub = [r for r in ret if r["model"] == model]
        spread = max(r["r@10"] for r in sub) - min(r["r@10"] for r in sub)
        check(f"projection choice is quality-neutral ({model})", spread <= 0.03,
              f"R@10 spread {spread:.3f}")
    # the point of §7: even the cheapest projection dwarfs the encode it serves
    pa = {r["model"]: r for r in json.load(open(HERE / "results_kbpath.json"))}["bekko-a8m"]
    check("even the cheapest projection costs more than bekko-a8m's encode",
          D[256]["kb_rademacher_ms"] > pa["encode_ms"],
          f"rademacher {D[256]['kb_rademacher_ms']:.1f} ms vs encode {pa['encode_ms']:.1f} ms")

    # ── composition ─────────────────────────────────────────────────────────
    C = json.load(open(HERE / "results_compose.json"))
    cells = 0
    for v in ("a8m", "a25m"):
        for dim in (384, 256, 128, 64):
            g = lambda p: [r["r@10"] for r in C if r["variant"] == v and r["dim"] == dim
                           and r["codec"] == "remex" and r["param"] == p][0]
            if g(2) > g(1):
                cells += 1
    check("2-bit beats 1-bit in ALL 8 cells (bekko is Jina-side)", cells == 8, f"{cells}/8")

    def best_at(v, b):
        s = [r for r in C if r["variant"] == v and r["bytes"] == b]
        return max(r["r@10"] for r in s) if s else None

    # ── head-to-head: trimming vs quantization at equal bytes ───────────────
    HH = [r for r in json.load(open(HERE / "results_headtohead.json"))
          if r["variant"] == "a25m"]
    def arm(a, p_):
        return [r for r in HH if r["arm"] == a and r["param"] == p_][0]
    shared = sorted({r["bytes"] for r in HH if r["arm"] == "matryoshka"} &
                    {r["bytes"] for r in HH if r["arm"] == "remex@384"})
    check("quantization beats trimming at EVERY shared byte budget",
          all(max(r["r@10"] for r in HH if r["arm"] == "remex@384" and r["bytes"] == b)
              > max(r["r@10"] for r in HH if r["arm"] == "matryoshka" and r["bytes"] == b)
              for b in shared),
          f"budgets {shared}")
    check("remex 2-bit @96 B beats uncompressed fp32 @1536 B",
          arm("remex@384", 2)["r@10"] > arm("matryoshka", 384)["r@10"],
          f"{arm('remex@384', 2)['r@10']:.3f} @96 B vs "
          f"{arm('matryoshka', 384)['r@10']:.3f} @1536 B")
    check("remex 1-bit @48 B matches Matryoshka d=128 @512 B",
          arm("remex@384", 1)["r@10"] >= arm("matryoshka", 128)["r@10"],
          f"{arm('remex@384', 1)['r@10']:.3f} vs {arm('matryoshka', 128)['r@10']:.3f}")
    # STRAWMAN GUARD: an earlier pass quoted a 2.1x headline against Matryoshka
    # d=12, far below the vendor's documented floor ("Supported truncate
    # dimensions: 256, 128, 64"). The claim must hold at the vendor's own floor.
    m64 = arm("matryoshka", 64)
    check("beats the VENDOR FLOOR d=64 on both bytes and recall (no strawman needed)",
          arm("remex@384", 1)["bytes"] < m64["bytes"]
          and arm("remex@384", 1)["r@10"] > m64["r@10"],
          f"remex 1-bit {arm('remex@384', 1)['bytes']} B/{arm('remex@384', 1)['r@10']:.3f} "
          f"vs d=64 {m64['bytes']} B/{m64['r@10']:.3f}")
    check("headline does not depend on sub-64 (off-spec) tiers",
          arm("remex@384", 2)["r@10"] > arm("matryoshka", 384)["r@10"]
          and arm("remex@384", 1)["r@10"] >= arm("matryoshka", 128)["r@10"])
    check("remex beats remax at equal bytes",
          all(arm("remex@384", b)["r@10"] > arm("remax@384", b)["r@10"] for b in (1, 2, 4, 8)))
    # coarse filter: quantized arm is the better stage-1, and the ceiling is the encoder
    check("remex 2-bit @96 B matches uncompressed R@50 (better coarse filter)",
          arm("remex@384", 2)["r@50"] >= arm("matryoshka", 384)["r@50"] - 1e-9
          and arm("remex@384", 2)["r@50"] > arm("matryoshka", 128)["r@50"],
          f"{arm('remex@384', 2)['r@50']:.3f} @96 B vs Matryoshka d=128 "
          f"{arm('matryoshka', 128)['r@50']:.3f} @512 B")
    check("R@50 ceiling is not a compression limit (fp32 hits it too)",
          arm("matryoshka", 384)["r@50"] < 0.90,
          f"uncompressed fp32 R@50 = {arm('matryoshka', 384)['r@50']:.3f}")

    # ── recovering the R@50 ceiling ─────────────────────────────────────────
    RC = json.load(open(HERE / "results_recover.json"))
    res = {r["method"]: r for r in RC["results"]}
    check("the ceiling is only PARTLY bekko's: jina fails fewer, overlap smaller",
          len(RC["both_fail"]) < len(RC["fail"]),
          f"{len(RC['fail'])} bekko / {len(RC['both_fail'])} shared "
          f"= {len(RC['both_fail']) / RC['n'] * 100:.1f}% true floor")
    check("BM25 recovers a majority of dense's misses despite being worse overall",
          res["BM25 only"]["recovered"] > res["dense only (bekko-a25m)"]["recovered"]
          and res["BM25 only"]["r@10"] < res["dense only (bekko-a25m)"]["r@10"],
          f"{res['BM25 only']['recovered']}/{res['BM25 only']['n_fail']} recovered, "
          f"R@10 {res['BM25 only']['r@10']:.3f}")
    check("RRF fusion is the best overall arm",
          res["dense + BM25, RRF"]["r@10"] > res["dense only (bekko-a25m)"]["r@10"]
          and res["dense + BM25, RRF"]["r@50"] > res["dense only (bekko-a25m)"]["r@50"],
          f"R@10 {res['dense + BM25, RRF']['r@10']:.3f} "
          f"R@50 {res['dense + BM25, RRF']['r@50']:.3f}")
    check("query expansion is the WEAKEST remedy (reproduces the muninn-rm3 negative)",
          res["query expansion (RM3-ish)"]["r@50"]
          < res["dense only (bekko-a25m)"]["r@50"]
          and res["query expansion (RM3-ish)"]["recovered"]
          < res["BM25 only"]["recovered"],
          f"R@50 {res['query expansion (RM3-ish)']['r@50']:.3f} vs "
          f"{res['dense only (bekko-a25m)']['r@50']:.3f}, "
          f"{res['query expansion (RM3-ish)']['recovered']}/26 recovered")

    # ── statistical power ───────────────────────────────────────────────────
    # n=179 is the number that governs every embedding-quality claim here.
    # These guards keep the writeup from re-asserting under-powered results.
    S = json.load(open(HERE / "results_significance.json"))
    by = {c["claim"]: c for c in S["claims"]}
    check("corpus is the 179-chunk blog subset (1 query = 0.56 pp)", S["n"] == 179)
    sig = [c for c in S["claims"] if c["significant"]]
    check("only ONE headline byte-budget claim survives n=179",
          len(sig) == 1 and "d=64" in sig[0]["claim"],
          f"{len(sig)}/{len(S['claims'])} significant: {sig[0]['claim']}")
    for cl in ("remex 2-bit @96B beats UNCOMPRESSED fp32 @1536B",
               "remex 1-bit @48B beats vendor floor d=64 @256B"):
        check(f"UNDER-POWERED, must not be quoted as established: {cl[:44]}",
              not by[cl]["significant"] and by[cl]["ci_lo"] < 0 < by[cl]["ci_hi"],
              f"p={by[cl]['p']:.3f} CI [{by[cl]['ci_lo']:+.3f},{by[cl]['ci_hi']:+.3f}]")
    check("truncation-to-d=64 cost IS established",
          by["Matryoshka d=384 beats d=64"]["significant"],
          f"p={by['Matryoshka d=384 beats d=64']['p']:.3f}")

    # ── honest bytes (the Matryoshka-vs-codec accounting audit) ─────────────
    H = [r for r in json.load(open(HERE / "results_honest_bytes.json"))
         if r["variant"] == "a25m"]
    g = lambda c, d, p: [r for r in H if r["codec"] == c and r["dim"] == d
                         and r["param"] == p][0]

    # payload must come from remex's OWN accounting (which includes the norms),
    # not a hand-computed dim*bits/8 that silently drops them.
    r384_2 = g("remex", 384, 2)
    check("remex payload uses remex's own nbytes (norms included)",
          abs(r384_2["payload_b"] - 100) < 1
          and r384_2["payload_b"] > r384_2["naive_payload_b"],
          f"{r384_2['payload_b']:.0f} B vs naive {r384_2['naive_payload_b']:.0f} B")

    # Matryoshka truncation ships nothing; the codecs do not. Guard the asymmetry.
    check("Matryoshka arms carry zero side data",
          all(r["side_b"] == 0 for r in H if r["codec"] == "fp32"))
    check("codec arms carry non-zero side data IF materialized",
          all(r["side_b"] > 0 for r in H if r["codec"] in ("remex", "remax")))

    f384 = g("fp32", 384, 32)
    check("payload-only: quantization still beats full fp32",
          r384_2["r@10"] >= f384["r@10"] and r384_2["payload_b"] < f384["payload_b"],
          f"{r384_2['payload_b']:.0f} B vs {f384['payload_b']:.0f} B")

    # RETRACTED CLAIM GUARD: an earlier pass charged each codec a materialized
    # dense d x d rotation, concluded the advantage inverts below n~411, and was
    # wrong. remex's codebook is 28 B (analytic, scalar Lloyd-Max), and the
    # rotation is seed-derived in every remax_kb config except v2-haar+int8 --
    # an RHT's whole state in operator form is rounds*d BITS. Assert the real
    # accounting so the over-correction cannot come back.
    from remex.codebook import lloyd_max_codebook as _cb
    bo, ce = _cb(384, 2)
    check("remex codebook is tiny and analytic (not a 'large shared codebook')",
          bo.nbytes + ce.nbytes < 64,
          f"{bo.nbytes + ce.nbytes} B at d=384/2-bit, "
          f"{(384 * 384 * 4) // (bo.nbytes + ce.nbytes)}x smaller than a dense rotation")
    SEED_ONLY, RHT_OPERATOR = 4, 3 * 384 // 8
    n0 = 179
    for label, side in (("seed-only", SEED_ONLY), ("RHT operator form", RHT_OPERATOR)):
        check(f"NO inversion at n=179 with realistic side data ({label})",
              r384_2["payload_b"] + side / n0 < f384["payload_b"],
              f"{r384_2['payload_b'] + side / n0:.1f} B vs {f384['payload_b']:.0f} B")
    check("RHT operator state is O(d) bits, not O(d^2) floats",
          RHT_OPERATOR * 100 < 384 * 384 * 4,
          f"{RHT_OPERATOR} B vs {384 * 384 * 4} B materialized")

    # ── composition ─────────────────────────────────────────────────────────
    C = json.load(open(HERE / "results_compose.json"))
    cells = 0
    for v in ("a8m", "a25m"):
        for dim in (384, 256, 128, 64):
            g = lambda p: [r["r@10"] for r in C if r["variant"] == v and r["dim"] == dim
                           and r["codec"] == "remex" and r["param"] == p][0]
            if g(2) > g(1):
                cells += 1
    check("2-bit beats 1-bit in ALL 8 cells (bekko is Jina-side)", cells == 8, f"{cells}/8")

    def best_at(v, b):
        s = [r for r in C if r["variant"] == v and r["bytes"] == b]
        return max(r["r@10"] for r in s) if s else None

    # ── head-to-head: trimming vs quantization at equal bytes ───────────────
    HH = [r for r in json.load(open(HERE / "results_headtohead.json"))
          if r["variant"] == "a25m"]
    def arm(a, p_):
        return [r for r in HH if r["arm"] == a and r["param"] == p_][0]
    shared = sorted({r["bytes"] for r in HH if r["arm"] == "matryoshka"} &
                    {r["bytes"] for r in HH if r["arm"] == "remex@384"})
    check("quantization beats trimming at EVERY shared byte budget",
          all(max(r["r@10"] for r in HH if r["arm"] == "remex@384" and r["bytes"] == b)
              > max(r["r@10"] for r in HH if r["arm"] == "matryoshka" and r["bytes"] == b)
              for b in shared),
          f"budgets {shared}")
    check("remex 2-bit @96 B beats uncompressed fp32 @1536 B",
          arm("remex@384", 2)["r@10"] > arm("matryoshka", 384)["r@10"],
          f"{arm('remex@384', 2)['r@10']:.3f} @96 B vs "
          f"{arm('matryoshka', 384)['r@10']:.3f} @1536 B")
    check("remex 1-bit @48 B matches Matryoshka d=128 @512 B",
          arm("remex@384", 1)["r@10"] >= arm("matryoshka", 128)["r@10"],
          f"{arm('remex@384', 1)['r@10']:.3f} vs {arm('matryoshka', 128)['r@10']:.3f}")
    check("advantage widens as the budget shrinks",
          (arm("remex@384", 1)["r@10"] / arm("matryoshka", 12)["r@10"])
          > (arm("remex@384", 8)["r@10"] / arm("matryoshka", 96)["r@10"]),
          f"{arm('remex@384', 1)['r@10'] / arm('matryoshka', 12)['r@10']:.2f}x @48 B "
          f"vs {arm('remex@384', 8)['r@10'] / arm('matryoshka', 96)['r@10']:.2f}x @384 B")
    check("remex beats remax at equal bytes",
          all(arm("remex@384", b)["r@10"] > arm("remax@384", b)["r@10"] for b in (1, 2, 4, 8)))
    # coarse filter: quantized arm is the better stage-1, and the ceiling is the encoder
    check("remex 2-bit @96 B matches uncompressed R@50 (better coarse filter)",
          arm("remex@384", 2)["r@50"] >= arm("matryoshka", 384)["r@50"] - 1e-9
          and arm("remex@384", 2)["r@50"] > arm("matryoshka", 128)["r@50"],
          f"{arm('remex@384', 2)['r@50']:.3f} @96 B vs Matryoshka d=128 "
          f"{arm('matryoshka', 128)['r@50']:.3f} @512 B")
    check("R@50 ceiling is an ENCODER limit, not a compression limit",
          arm("matryoshka", 384)["r@50"] < 0.90,
          f"uncompressed fp32 R@50 = {arm('matryoshka', 384)['r@50']:.3f}")

    # ── honest bytes (the Matryoshka-vs-codec accounting audit) ─────────────
    H = [r for r in json.load(open(HERE / "results_honest_bytes.json"))
         if r["variant"] == "a25m"]
    g = lambda c, d, p: [r for r in H if r["codec"] == c and r["dim"] == d
                         and r["param"] == p][0]

    # payload must come from remex's OWN accounting (which includes the norms),
    # not a hand-computed dim*bits/8 that silently drops them.
    r384_2 = g("remex", 384, 2)
    check("remex payload uses remex's own nbytes (norms included)",
          abs(r384_2["payload_b"] - 100) < 1
          and r384_2["payload_b"] > r384_2["naive_payload_b"],
          f"{r384_2['payload_b']:.0f} B vs naive {r384_2['naive_payload_b']:.0f} B")

    # Matryoshka truncation ships nothing; the codecs do not. Guard the asymmetry.
    check("Matryoshka arms carry zero side data",
          all(r["side_b"] == 0 for r in H if r["codec"] == "fp32"))
    check("codec arms carry non-zero side data IF materialized",
          all(r["side_b"] > 0 for r in H if r["codec"] in ("remex", "remax")))

    f384 = g("fp32", 384, 32)
    check("payload-only: quantization still beats full fp32",
          r384_2["r@10"] >= f384["r@10"] and r384_2["payload_b"] < f384["payload_b"],
          f"{r384_2['payload_b']:.0f} B vs {f384['payload_b']:.0f} B")

    # ...but materialized at the benchmarked corpus size the claim INVERTS.
    n0 = 179
    check("at n=179 with side data materialized, truncation wins (claim inverts)",
          r384_2["payload_b"] + r384_2["side_b"] / n0 > f384["payload_b"],
          f"{r384_2['payload_b'] + r384_2['side_b'] / n0:.0f} B vs "
          f"{f384['payload_b']:.0f} B")
    breakeven = r384_2["side_b"] / (f384["payload_b"] - r384_2["payload_b"])
    check("break-even is in the low hundreds of vectors", 300 < breakeven < 600,
          f"n={breakeven:.0f}")

    print()
    if FAIL:
        print(f"{len(FAIL)} FAILED: {FAIL}")
        return 1
    print("all checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
