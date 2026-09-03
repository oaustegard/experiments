"""potion-code-16M-v2 vs bekko-a8m on the n=59 mini-CTXBench file-discovery task,
plus a potion-output quantization ladder (remex/remax) at matched byte budgets.

Mirrors scripts/run_code_quant.py from bekko-embedding-bench but:
  - swaps the encoder under test for potion-code-16M-v2 (Model2Vec, 256-d, fp16,
    encoder-side and output-side quantization both apply)
  - uses bekko-a8m (not a25m) as the dense reference, per BRIEF.md
  - adds encoder-native int8 table quantization (Model2Vec quantize_to="int8")
  - adds a stratum split (identifier-poor vs identifier-rich) and paired
    significance tests on per-instance r@5, following bench_significance.py's
    method (sign test + paired bootstrap CI) but on the continuous r@5 score
    (gold sets here can have >1 file, so recall_at is a fraction, not a 0/1 hit)
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from math import comb
from pathlib import Path

import numpy as np

BEKKO_DIR = Path("/home/user/experiments/bekko-embedding-bench")
sys.path.insert(0, str(BEKKO_DIR / "scripts"))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/usr/local/lib/python3.11/dist-packages"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/oaustegard/remax/src"))

from bekko import BekkoEncoder, matryoshka  # noqa: E402
from eval_search import extract_identifiers, arm_rg, recall_at, rrf  # noqa: E402
from run_code_quant import file_rank, hamming_rank  # noqa: E402

import remex  # noqa: E402
from remax import StackedSignBitQuantizer  # noqa: E402
from model2vec import StaticModel  # noqa: E402

HERE = Path(__file__).resolve().parent
POTION_PATH = "/home/user/experiments/.cache/potion"
RNG = np.random.default_rng(0)

rows: list[dict] = []
timing: dict = {}
failed: list[str] = []


def log(msg: str) -> None:
    print(msg, flush=True)


def add_row(**kw) -> dict:
    row = {"r@5": None, "r@10": None, "rrf_r@10": None, **kw}
    rows.append(row)
    return row


def score(inst, chunks, rg_ranked, ranker):
    r5 = np.zeros(len(inst))
    r10 = np.zeros(len(inst))
    f10 = np.zeros(len(inst))
    for i, it in enumerate(inst):
        ranked = ranker(i)
        r5[i] = recall_at(ranked, it["gold"], 5)
        r10[i] = recall_at(ranked, it["gold"], 10)
        f10[i] = recall_at(rrf(rg_ranked[it["issue"]], ranked), it["gold"], 10)
    return r5, r10, f10


def print_row(label: str, bytes_per_vec, r5, r10, f10) -> None:
    b = f"{bytes_per_vec:5d} B" if bytes_per_vec is not None else "    - B"
    log(f"  {label:34s} {b}   r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}   "
        f"+rg(RRF) r@10 {f10.mean():.3f}")


def sign_test(a: np.ndarray, b: np.ndarray) -> tuple[int, int, float]:
    """Two-sided exact sign test on paired per-instance scores (a vs b),
    ties excluded. Same combinatorial formula as bench_significance.mcnemar_exact."""
    wins = int((a > b).sum())
    losses = int((a < b).sum())
    n = wins + losses
    if n == 0:
        return wins, losses, 1.0
    k = min(wins, losses)
    p = min(1.0, 2.0 * sum(comb(n, i) for i in range(k + 1)) / 2**n)
    return wins, losses, p


def boot_ci(a: np.ndarray, b: np.ndarray, reps: int = 20000) -> tuple[float, float]:
    n = len(a)
    idx = RNG.integers(0, n, size=(reps, n))
    d = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def significance(label: str, name_a: str, a: np.ndarray, name_b: str, b: np.ndarray) -> dict:
    diff = float(a.mean() - b.mean())
    lo, hi = boot_ci(a, b)
    wins, losses, p = sign_test(a, b)
    verdict = "significant" if p < 0.05 else "not significant"
    log(f"  {label:42s} Δr@5 {diff:+.3f}  95%CI [{lo:+.3f},{hi:+.3f}]  "
        f"wins/losses {wins}/{losses}  sign-test p={p:.3f}  ({verdict})")
    return {"claim": label, "a": name_a, "b": name_b, "delta_r5": diff,
            "ci_lo": lo, "ci_hi": hi, "wins": wins, "losses": losses,
            "p": p, "significant": bool(p < 0.05)}


def main() -> None:
    box = subprocess.run(["bash", "-c", "nproc; free -h | sed -n 2p"],
                          capture_output=True, text=True).stdout.strip()
    log(f"box: {box}\n")

    inst = json.load(open(BEKKO_DIR / "instances.json"))
    chunks = json.load(open(BEKKO_DIR / "chunks_ast.json"))
    n_chunks = len(chunks)
    n_files = len({c["file"] for c in chunks})
    log(f"CODE corpus: {n_chunks} chunks, {n_files} files; task = n={len(inst)} file discovery\n")

    queries = [it["title"] + "\n" + it["body"] for it in inst]
    idents_by_inst = [extract_identifiers(q) for q in queries]
    poor_mask = np.array([len(ids) == 0 for ids in idents_by_inst])
    log(f"identifier-poor instances: {int(poor_mask.sum())} / {len(inst)}\n")

    # ---- rg baseline (shared) ------------------------------------------------
    rg_ranked = {}
    t0 = time.time()
    for it, q in zip(inst, queries):
        r, _, _ = arm_rg(extract_identifiers(q))
        rg_ranked[it["issue"]] = r
    log(f"rg baseline done in {time.time() - t0:.1f}s")
    rg5, rg10, rgf10 = score(inst, chunks, rg_ranked, lambda i: rg_ranked[inst[i]["issue"]])
    add_row(encoder="rg", codec="rg", param=None, dim=None, bytes=None,
            **{"r@5": rg5.mean(), "r@10": rg10.mean(), "rrf_r@10": rgf10.mean()})
    print_row("rg baseline", None, rg5, rg10, rgf10)

    # ---- bekko-a8m fp32 d=384 (reference) ------------------------------------
    log("\n=== bekko-a8m fp32 d=384 ===")
    a8m_ok = True
    try:
        mat_a8m = np.asarray(np.memmap(BEKKO_DIR / "vecs_ast_a8m.f32", dtype=np.float32,
                                        mode="r", shape=(n_chunks, 384)))
        enc_a8m = BekkoEncoder("a8m", threads=4)
        t0 = time.time()
        qv_a8m = enc_a8m.encode(queries, batch_size=8)
        timing["bekko_a8m_query_encode_total_s"] = time.time() - t0
        # per-query latency, pinned to 1 thread if possible
        os.environ["OMP_NUM_THREADS"] = "1"
        enc_a8m_1t = BekkoEncoder("a8m", threads=1)
        lat = []
        for q in queries:
            t0 = time.time()
            enc_a8m_1t.encode([q], batch_size=1, sort_by_length=False)
            lat.append(time.time() - t0)
        timing["bekko_a8m_per_query_latency_median_s_1thread"] = float(np.median(lat))
        del os.environ["OMP_NUM_THREADS"]

        r5, r10, f10 = score(inst, chunks, rg_ranked,
                              lambda i: file_rank(mat_a8m @ qv_a8m[i], chunks))
        add_row(encoder="bekko-a8m", codec="fp32", param=32, dim=384, bytes=384 * 4,
                **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
        print_row("bekko-a8m fp32 d=384", 384 * 4, r5, r10, f10)
        bekko_fp32_r5 = r5
        log(f"  reference check: harness's own n=59 number was r@5 0.595; "
            f"{'REPRODUCED' if abs(r5.mean() - 0.595) < 0.01 else 'DID NOT reproduce exactly'} "
            f"(got {r5.mean():.3f})")
    except Exception as e:
        a8m_ok = False
        failed.append(f"bekko-a8m fp32: {e!r}")
        log(f"  FAILED: {e!r}")
        bekko_fp32_r5 = None

    # ---- potion fp16 d=256 as shipped -----------------------------------------
    log("\n=== potion fp16 d=256 (as shipped) ===")
    m_fp16 = StaticModel.from_pretrained(POTION_PATH)
    t0 = time.time()
    dv_fp16 = np.asarray(m_fp16.encode([c["text"] for c in chunks],
                                        max_length=512, batch_size=1024)).astype(np.float32)
    timing["potion_corpus_encode_total_s"] = time.time() - t0
    timing["potion_corpus_chunks_per_s"] = n_chunks / timing["potion_corpus_encode_total_s"]

    t0 = time.time()
    qv_fp16 = np.asarray(m_fp16.encode(queries, max_length=512, batch_size=1024)).astype(np.float32)
    timing["potion_query_encode_total_s"] = time.time() - t0

    os.environ["OMP_NUM_THREADS"] = "1"
    lat = []
    for q in queries:
        t0 = time.time()
        m_fp16.encode([q], max_length=512, batch_size=1)
        lat.append(time.time() - t0)
    timing["potion_per_query_latency_median_s_1thread"] = float(np.median(lat))
    del os.environ["OMP_NUM_THREADS"]

    r5, r10, f10 = score(inst, chunks, rg_ranked,
                          lambda i: file_rank(dv_fp16 @ qv_fp16[i], chunks))
    add_row(encoder="potion", codec="fp16", param=16, dim=256, bytes=256 * 2,
            **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
    print_row("potion fp16 d=256", 256 * 2, r5, r10, f10)
    potion_fp16_r5 = r5
    poor_potion_r5, rich_potion_r5 = r5[poor_mask], r5[~poor_mask]

    # ---- potion dimension truncation: load-time vs post-hoc slice -----------
    log("\n=== potion dimension truncation d=128, 64 ===")
    trunc_agree = {}
    for dim in (128, 64):
        m_d = StaticModel.from_pretrained(POTION_PATH, dimensionality=dim)
        dv_d = np.asarray(m_d.encode([c["text"] for c in chunks],
                                      max_length=512, batch_size=1024)).astype(np.float32)
        qv_d = np.asarray(m_d.encode(queries, max_length=512, batch_size=1024)).astype(np.float32)
        r5, r10, f10 = score(inst, chunks, rg_ranked,
                              lambda i: file_rank(dv_d @ qv_d[i], chunks))
        add_row(encoder="potion", codec="fp16-loadtime-trunc", param=16, dim=dim,
                bytes=dim * 2, **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
        print_row(f"potion d={dim} (load-time)", dim * 2, r5, r10, f10)

        d_sliced = matryoshka(dv_fp16, dim)
        q_sliced = matryoshka(qv_fp16, dim)
        r5s, r10s, f10s = score(inst, chunks, rg_ranked,
                                 lambda i: file_rank(d_sliced @ q_sliced[i], chunks))
        add_row(encoder="potion", codec="fp16-posthoc-slice", param=16, dim=dim,
                bytes=dim * 2, **{"r@5": r5s.mean(), "r@10": r10s.mean(), "rrf_r@10": f10s.mean()})
        print_row(f"potion d={dim} (post-hoc slice)", dim * 2, r5s, r10s, f10s)

        # per-instance ranking equality check (same file order top-10)
        same = 0
        for i in range(len(inst)):
            a = file_rank(dv_d @ qv_d[i], chunks)[:10]
            b = file_rank(d_sliced @ q_sliced[i], chunks)[:10]
            same += int(a == b)
        trunc_agree[dim] = same
        log(f"    top-10 file ranking identical for {same}/{len(inst)} instances at d={dim}")

    # ---- potion native int8 table quantization -------------------------------
    log("\n=== potion int8 table quantization d=256 ===")
    m_i8 = StaticModel.from_pretrained(POTION_PATH, quantize_to="int8")
    dv_i8 = np.asarray(m_i8.encode([c["text"] for c in chunks],
                                    max_length=512, batch_size=1024)).astype(np.float32)
    qv_i8 = np.asarray(m_i8.encode(queries, max_length=512, batch_size=1024)).astype(np.float32)
    cos = np.sum(dv_i8 * dv_fp16, axis=1) / (
        np.linalg.norm(dv_i8, axis=1) * np.linalg.norm(dv_fp16, axis=1) + 1e-12)
    log(f"  cosine(int8-table output, fp16 output) over corpus: "
        f"mean {cos.mean():.4f}  min {cos.min():.4f}")
    r5, r10, f10 = score(inst, chunks, rg_ranked,
                          lambda i: file_rank(dv_i8 @ qv_i8[i], chunks))
    add_row(encoder="potion", codec="int8-table", param=8, dim=256, bytes=256,
            cosine_to_fp16_mean=float(cos.mean()), cosine_to_fp16_min=float(cos.min()),
            **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
    print_row("potion int8-table d=256", 256, r5, r10, f10)

    # ---- potion output remex 1/2/4-bit ---------------------------------------
    log("\n=== potion output remex quantization d=256 ===")
    potion_remex2_r5 = None
    for bits in (1, 2, 4):
        qz = remex.Quantizer(d=256, bits=bits, seed=0)
        xh = qz.decode(qz.encode(dv_fp16))
        xh = xh / np.clip(np.linalg.norm(xh, axis=1, keepdims=True), 1e-9, None)
        r5, r10, f10 = score(inst, chunks, rg_ranked,
                              lambda i, xh=xh: file_rank(xh @ qv_fp16[i], chunks))
        add_row(encoder="potion", codec="remex", param=bits, dim=256, bytes=256 * bits // 8,
                **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
        print_row(f"potion remex {bits}-bit d=256", 256 * bits // 8, r5, r10, f10)
        if bits == 2:
            potion_remex2_r5 = r5

    # ---- potion output remax k=1/2/4/8 ---------------------------------------
    log("\n=== potion output remax quantization d=256 ===")
    potion_remax1_r5 = None
    for k in (1, 2, 4, 8):
        sq = StackedSignBitQuantizer(d=256, k=k, seed=0).fit(dv_fp16)
        dc, qc = sq.encode(dv_fp16), sq.encode(qv_fp16)
        r5, r10, f10 = score(inst, chunks, rg_ranked,
                              lambda i, qc=qc, dc=dc: hamming_rank(qc[i], dc, chunks))
        add_row(encoder="potion", codec="remax", param=k, dim=256, bytes=256 * k // 8,
                **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
        print_row(f"potion remax k={k} d=256", 256 * k // 8, r5, r10, f10)
        if k == 1:
            potion_remax1_r5 = r5

    # ---- bekko-a8m output remex 2-bit d=384 / remax k=1 d=384 (comparators) --
    if a8m_ok:
        log("\n=== bekko-a8m output quantization d=384 (deployed settings) ===")
        qz = remex.Quantizer(d=384, bits=2, seed=0)
        xh = qz.decode(qz.encode(mat_a8m))
        xh = xh / np.clip(np.linalg.norm(xh, axis=1, keepdims=True), 1e-9, None)
        r5, r10, f10 = score(inst, chunks, rg_ranked,
                              lambda i: file_rank(xh @ qv_a8m[i], chunks))
        add_row(encoder="bekko-a8m", codec="remex", param=2, dim=384, bytes=384 * 2 // 8,
                **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
        print_row("bekko-a8m remex 2-bit d=384", 384 * 2 // 8, r5, r10, f10)

        sq = StackedSignBitQuantizer(d=384, k=1, seed=0).fit(mat_a8m)
        dc, qc = sq.encode(mat_a8m), sq.encode(qv_a8m)
        r5, r10, f10 = score(inst, chunks, rg_ranked,
                              lambda i, qc=qc, dc=dc: hamming_rank(qc[i], dc, chunks))
        add_row(encoder="bekko-a8m", codec="remax", param=1, dim=384, bytes=384 * 1 // 8,
                **{"r@5": r5.mean(), "r@10": r10.mean(), "rrf_r@10": f10.mean()})
        print_row("bekko-a8m remax k=1 d=384", 384 * 1 // 8, r5, r10, f10)
    else:
        failed.append("bekko-a8m output remex/remax skipped: fp32 reference failed")

    # ---- timing: corpus encode wall clock from a8m log ------------------------
    log("\n=== timing ===")
    a8m_log = BEKKO_DIR / "encode_a8m.log"
    a8m_wall_s = None
    if a8m_log.exists():
        txt = a8m_log.read_text()
        if "DONE" in txt:
            import re
            m = re.search(r"DONE 11380 in ([\d.]+)m", txt)
            if m:
                a8m_wall_s = float(m.group(1)) * 60
        else:
            failed.append("bekko-a8m corpus encode did not finish before this run wrote results")
    timing["bekko_a8m_corpus_encode_wall_s"] = a8m_wall_s
    for k, v in timing.items():
        log(f"  {k}: {v}")

    # ---- stratum split ---------------------------------------------------------
    log("\n=== stratum split: identifier-poor vs identifier-rich, r@5 ===")
    strata = {}
    for name, arr in (("rg", rg5), ("bekko-a8m fp32", bekko_fp32_r5), ("potion fp16", potion_fp16_r5)):
        if arr is None:
            continue
        poor = arr[poor_mask]
        rich = arr[~poor_mask]
        strata[name] = {
            "n_poor": int(poor_mask.sum()), "r5_poor": float(poor.mean()) if len(poor) else None,
            "n_rich": int((~poor_mask).sum()), "r5_rich": float(rich.mean()) if len(rich) else None,
        }
        log(f"  {name:16s} poor(n={strata[name]['n_poor']}) r@5={strata[name]['r5_poor']:.3f}   "
            f"rich(n={strata[name]['n_rich']}) r@5={strata[name]['r5_rich']:.3f}")

    # ---- significance -----------------------------------------------------------
    log("\n=== paired significance (per-instance r@5, n=59) ===")
    sig = []
    if bekko_fp32_r5 is not None:
        sig.append(significance("potion fp16 vs bekko-a8m fp32", "potion-fp16", potion_fp16_r5,
                                 "bekko-a8m-fp32", bekko_fp32_r5))
    if potion_remex2_r5 is not None:
        sig.append(significance("potion remex 2-bit vs potion fp16", "potion-remex2", potion_remex2_r5,
                                 "potion-fp16", potion_fp16_r5))
    if potion_remax1_r5 is not None:
        sig.append(significance("potion remax k=1 vs potion fp16", "potion-remax1", potion_remax1_r5,
                                 "potion-fp16", potion_fp16_r5))

    # ---- write outputs -----------------------------------------------------------
    out = {
        "box": box,
        "corpus": {"chunks": n_chunks, "files": n_files, "instances": len(inst)},
        "gold_coverage": 1.0,
        "rows": rows,
        "timing": timing,
        "truncation_agreement": trunc_agree,
        "strata": strata,
        "significance": sig,
        "failed": failed,
    }
    json.dump(out, open(HERE / "results.json", "w"), indent=1, default=float)
    log(f"\nwrote {HERE / 'results.json'}")
    if failed:
        log("\nFAILURES:")
        for f in failed:
            log(f"  - {f}")


if __name__ == "__main__":
    main()
