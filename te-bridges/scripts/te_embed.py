"""Stage 2 — SPECTER2 fetch for both corpus pools.

Fetches S2 precomputed SPECTER2 vectors, stores them for cross-axis
cosine scoring. Keeps empirical and theory pools separate so we never
compute empirical×empirical or theory×theory pairs.

Output files:
  data/empirical_meta.json  — [{arxiv_id, title, paperId, ...}, ...]
  data/theory_meta.json     — [{arxiv_id, title, paperId, ...}, ...]
  data/empirical_vecs.npy   — float32 (N_emp, 768)
  data/theory_vecs.npy      — float32 (N_th, 768)
  data/embed_missing.json   — {empirical: [...], theory: [...]}

Resumable: intermediate checkpoints in empirical_meta_ckpt.json /
theory_meta_ckpt.json allow re-runs to skip fetched batches.

Usage:
  python te_embed.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from te_common import DATA, chunked, load_json, s2_post, save_json  # noqa: E402

BATCH_SIZE = 500
SPECTER_DIM = 768
INTERBATCH_SLEEP = 4.0


def s2_batch_specter(arxiv_ids: list[str]) -> list[dict | None]:
    payload = {"ids": [f"ARXIV:{a}" for a in arxiv_ids]}
    fields = "paperId,title,externalIds,fieldsOfStudy,s2FieldsOfStudy,embedding.specter_v2"
    return s2_post("/paper/batch", payload, params={"fields": fields})


def fetch_pool(
    pool_name: str,
    arxiv_ids: list[str],
) -> tuple[list[dict], np.ndarray, list[str]]:
    """Fetch SPECTER2 for a named pool. Returns (meta, vecs_matrix, missing)."""
    ckpt_name = f"{pool_name}_meta_ckpt.json"
    ckpt = load_json(ckpt_name, default={"last_batch": -1, "meta": [], "missing": []})
    last_batch = int(ckpt.get("last_batch", -1))
    all_meta: list[dict] = list(ckpt.get("meta") or [])
    all_vecs: list[list[float]] = list(ckpt.get("vecs") or [])
    missing: list[str] = list(ckpt.get("missing") or [])

    batches = list(chunked(arxiv_ids, BATCH_SIZE))
    for batch_idx, batch in enumerate(batches):
        if batch_idx <= last_batch:
            print(f"  [{pool_name}] batch {batch_idx}: skip (checkpoint)")
            continue
        print(f"  [{pool_name}] batch {batch_idx}/{len(batches)-1}: {len(batch)} papers…")
        t0 = time.time()
        resp = s2_batch_specter(batch)
        print(f"    fetched in {time.time()-t0:.1f}s, "
              f"got {sum(1 for x in resp if x)} non-null")

        for arxiv_id, paper in zip(batch, resp):
            if not paper:
                missing.append(arxiv_id)
                continue
            emb = (paper.get("embedding") or {}).get("vector")
            if not emb:
                missing.append(arxiv_id)
                continue
            pid = paper.get("paperId") or arxiv_id
            all_meta.append({
                "paperId": pid,
                "arxiv_id": (paper.get("externalIds") or {}).get("ArXiv") or arxiv_id,
                "title": paper.get("title") or "",
                "fos": paper.get("fieldsOfStudy") or [],
                "s2_fos": [f.get("category") for f in (paper.get("s2FieldsOfStudy") or [])],
            })
            all_vecs.append(emb)

        # Checkpoint: store metadata + raw vectors (small enough at 800/1500 scale)
        save_json(ckpt_name, {
            "last_batch": batch_idx,
            "meta": all_meta,
            "vecs": all_vecs,
            "missing": missing,
        })
        if batch_idx < len(batches) - 1:
            time.sleep(INTERBATCH_SLEEP)

    vecs_mat = np.asarray(all_vecs, dtype=np.float32) if all_vecs else np.zeros((0, SPECTER_DIM), dtype=np.float32)
    return all_meta, vecs_mat, missing


def main() -> None:
    emp_corpus = load_json("empirical_corpus.json")
    th_corpus  = load_json("theory_corpus.json")
    if not emp_corpus or not th_corpus:
        print("ERROR: run te_corpus.py first", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching SPECTER2 for empirical pool ({emp_corpus['n']} papers)…")
    emp_meta, emp_vecs, emp_miss = fetch_pool("empirical", emp_corpus["arxiv_ids"])
    save_json("empirical_meta.json", emp_meta)
    np.save(str(DATA / "empirical_vecs.npy"), emp_vecs)
    print(f"  empirical: {len(emp_meta)} vectors, {len(emp_miss)} missing")

    print(f"\nFetching SPECTER2 for theory pool ({th_corpus['n']} papers)…")
    th_meta, th_vecs, th_miss = fetch_pool("theory", th_corpus["arxiv_ids"])
    save_json("theory_meta.json", th_meta)
    np.save(str(DATA / "theory_vecs.npy"), th_vecs)
    print(f"  theory: {len(th_meta)} vectors, {len(th_miss)} missing")

    save_json("embed_missing.json", {"empirical": emp_miss, "theory": th_miss})
    print(f"\nDone: emp={len(emp_meta)} th={len(th_meta)} total vecs")


if __name__ == "__main__":
    main()
