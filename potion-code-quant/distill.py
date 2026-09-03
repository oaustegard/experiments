"""Can a Model2Vec distillation of bekko-a8m produce embeddings compatible with
the teacher's (ONNX transformer) space?

Distills bekko-a8m twice (Zipf/SIF-weighted averaging on, and off — model2vec
0.9.0 exposes this as `sif_coefficient`, not `apply_zipf`; see the signature
check below) with pca_dims=None so the student stays at the teacher's own
384-d, which is the only way "student queries teacher's index" could plausibly
line up. Then measures:
  (a) cosine(student, teacher) over 2000 random corpus chunks and the 59 queries
  (b) three retrieval cells on the n=59 task: teacher/teacher (already have from
      run.py), student/student, and student-query/TEACHER-index (the deployable
      cell: a small distilled table querying an index bekko already built)
  (c) student table size and per-query encode latency
"""
from __future__ import annotations

import inspect
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BEKKO_DIR = Path("/home/user/experiments/bekko-embedding-bench")
sys.path.insert(0, str(BEKKO_DIR / "scripts"))
sys.path.insert(0, os.environ.get("REMEX_ROOT", "/usr/local/lib/python3.11/dist-packages"))
sys.path.insert(0, os.environ.get("REMAX_ROOT", "/home/user/oaustegard/remax/src"))
from bekko import BekkoEncoder  # noqa: E402
from eval_search import extract_identifiers, arm_rg, recall_at, rrf  # noqa: E402
from run_code_quant import file_rank  # noqa: E402

from model2vec import StaticModel  # noqa: E402
from model2vec.distill import distill  # noqa: E402

HERE = Path(__file__).resolve().parent
CACHE = Path("/home/user/experiments/.cache")
RNG = np.random.default_rng(0)

failed: list[str] = []


def log(m):
    print(m, flush=True)


def main() -> None:
    log(f"model2vec.distill.distill signature: {inspect.signature(distill)}\n")

    inst = json.load(open(BEKKO_DIR / "instances.json"))
    chunks = json.load(open(BEKKO_DIR / "chunks_ast.json"))
    n_chunks = len(chunks)
    queries = [it["title"] + "\n" + it["body"] for it in inst]

    # ---- distill (or load cached) --------------------------------------------
    variants = {
        "zipf": {"sif_coefficient": 1e-4, "path": CACHE / "potion-a8m-distill-zipf"},
        "nozipf": {"sif_coefficient": None, "path": CACHE / "potion-a8m-distill-nozipf"},
    }
    students = {}
    for name, cfg in variants.items():
        p = cfg["path"]
        try:
            if p.exists():
                log(f"loading cached distillation '{name}' from {p}")
                m = StaticModel.from_pretrained(str(p))
            else:
                t0 = time.time()
                m = distill("hotchpotch/bekko-embedding-v1-a8m", pca_dims=None,
                            sif_coefficient=cfg["sif_coefficient"])
                log(f"distilled '{name}' in {time.time() - t0:.1f}s")
                m.save_pretrained(str(p))
            students[name] = m
            vocab_n, dim = m.embedding.shape
            table_bytes = vocab_n * dim * np.dtype(m.embedding.dtype).itemsize
            log(f"  student '{name}': vocab={vocab_n} dim={dim} dtype={m.embedding.dtype} "
                f"table={table_bytes / 2**20:.1f} MB")
        except Exception as e:
            failed.append(f"distill '{name}': {e!r}")
            log(f"  FAILED distill '{name}': {e!r}")

    if not students:
        json.dump({"failed": failed}, open(HERE / "results_distill.json", "w"), indent=1)
        log("\nno students produced; stopping")
        return

    # ---- teacher (a8m fp32 ONNX) ----------------------------------------------
    log("\n=== teacher: bekko-a8m fp32 d=384 ===")
    teacher_mat = np.asarray(np.memmap(BEKKO_DIR / "vecs_ast_a8m.f32", dtype=np.float32,
                                        mode="r", shape=(n_chunks, 384)))
    teacher_enc = BekkoEncoder("a8m", threads=4)
    teacher_qv = teacher_enc.encode(queries, batch_size=8)

    rg_ranked = {}
    for it in inst:
        r, _, _ = arm_rg(extract_identifiers(it["title"] + "\n" + it["body"]))
        rg_ranked[it["issue"]] = r

    def score(ranker):
        r5 = np.zeros(len(inst)); r10 = np.zeros(len(inst)); f10 = np.zeros(len(inst))
        for i, it in enumerate(inst):
            ranked = ranker(i)
            r5[i] = recall_at(ranked, it["gold"], 5)
            r10[i] = recall_at(ranked, it["gold"], 10)
            f10[i] = recall_at(rrf(rg_ranked[it["issue"]], ranked), it["gold"], 10)
        return r5, r10, f10

    r5, r10, f10 = score(lambda i: file_rank(teacher_mat @ teacher_qv[i], chunks))
    teacher_teacher = {"cell": "teacher-query/teacher-index", "r@5": float(r5.mean()),
                        "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
    log(f"  teacher/teacher   r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  "
        f"RRF r@10 {f10.mean():.3f}   (matches run.py's bekko-a8m fp32 row)")

    # ---- cosine + retrieval per student ----------------------------------------
    sample_idx = RNG.choice(n_chunks, size=min(2000, n_chunks), replace=False)
    sample_texts = [chunks[i]["text"] for i in sample_idx]
    sample_teacher = teacher_mat[sample_idx]

    cells = {"teacher-query/teacher-index": teacher_teacher}
    cosines = {}
    tables = {}
    latencies = {}

    for name, m in students.items():
        log(f"\n=== student '{name}' ===")
        student_sample = np.asarray(m.encode(sample_texts, max_length=512,
                                              batch_size=1024)).astype(np.float32)
        cos_corpus = np.sum(student_sample * sample_teacher, axis=1) / (
            np.linalg.norm(student_sample, axis=1) * np.linalg.norm(sample_teacher, axis=1) + 1e-12)

        student_qv = np.asarray(m.encode(queries, max_length=512, batch_size=1024)).astype(np.float32)
        cos_query = np.sum(student_qv * teacher_qv, axis=1) / (
            np.linalg.norm(student_qv, axis=1) * np.linalg.norm(teacher_qv, axis=1) + 1e-12)

        log(f"  cosine(student,teacher) over 2000 corpus chunks: mean {cos_corpus.mean():.4f} "
            f"median {np.median(cos_corpus):.4f} min {cos_corpus.min():.4f}")
        log(f"  cosine(student,teacher) over 59 queries:         mean {cos_query.mean():.4f} "
            f"median {np.median(cos_query):.4f} min {cos_query.min():.4f}")
        cosines[name] = {
            "corpus_mean": float(cos_corpus.mean()), "corpus_median": float(np.median(cos_corpus)),
            "corpus_min": float(cos_corpus.min()),
            "query_mean": float(cos_query.mean()), "query_median": float(np.median(cos_query)),
            "query_min": float(cos_query.min()),
        }

        # student-query/student-index
        student_mat = np.asarray(m.encode([c["text"] for c in chunks], max_length=512,
                                           batch_size=1024)).astype(np.float32)
        r5, r10, f10 = score(lambda i: file_rank(student_mat @ student_qv[i], chunks))
        cells[f"student({name})-query/student({name})-index"] = {
            "cell": f"student({name})-query/student({name})-index",
            "r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
        log(f"  student/student   r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")

        # student-query/TEACHER-index (the deployable cell)
        r5, r10, f10 = score(lambda i: file_rank(teacher_mat @ student_qv[i], chunks))
        cells[f"student({name})-query/teacher-index"] = {
            "cell": f"student({name})-query/teacher-index",
            "r@5": float(r5.mean()), "r@10": float(r10.mean()), "rrf_r@10": float(f10.mean())}
        log(f"  student/TEACHER   r@5 {r5.mean():.3f}  r@10 {r10.mean():.3f}  RRF r@10 {f10.mean():.3f}")

        vocab_n, dim = m.embedding.shape
        table_bytes = int(vocab_n * dim * np.dtype(m.embedding.dtype).itemsize)
        tables[name] = {"vocab": int(vocab_n), "dim": int(dim), "dtype": str(m.embedding.dtype),
                         "table_bytes": table_bytes, "table_mb": table_bytes / 2**20}

        lat = []
        for q in queries:
            t0 = time.time()
            m.encode([q], max_length=512, batch_size=1)
            lat.append(time.time() - t0)
        latencies[name] = {"median_s": float(np.median(lat)), "mean_s": float(np.mean(lat))}
        log(f"  table: vocab={vocab_n} dim={dim} {tables[name]['table_mb']:.1f} MB; "
            f"per-query latency median {latencies[name]['median_s'] * 1000:.3f} ms")

    out = {
        "cells": cells,
        "cosines": cosines,
        "tables": tables,
        "latencies": latencies,
        "failed": failed,
    }
    json.dump(out, open(HERE / "results_distill.json", "w"), indent=1, default=float)
    log(f"\nwrote {HERE / 'results_distill.json'}")


if __name__ == "__main__":
    main()
