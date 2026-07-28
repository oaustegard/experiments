"""Stage 2 — Streaming SPECTER2 fetch + remax build.

Fetches S2 precomputed SPECTER2 vectors for the 1000 IDs in batches of 500,
quantizes them with remax (stacked SimHash, k=2), and writes:
  - data/codes.bin       — packed bits (n_actual, 192 bytes/vec)
  - data/metadata.json   — title, arxiv_id, fields_of_study, etc.
  - data/checkpoint.json — resume marker

Resumable: re-runs skip already-processed batches. Papers without SPECTER2
in S2's precomputed cache are dropped from the corpus (logged separately
in data/missing.json).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, chunked, load_json, s2_post, save_json  # noqa: E402
from remax import RemaxBuilder  # noqa: E402

BATCH_SIZE = 500
SPECTER_DIM = 768
REMAX_K = 2
# S2 unauthenticated rate limit: ~1 req/s. Be patient between batches.
INTERBATCH_SLEEP = 4.0

# Path to phase-0 SPECTER2 vectors that S2's index doesn't have yet
# (Sawin, OpenAI companion). Format: {label: {vector: [...], source_id: ..., title: ...}}.
# See oaustegard/claude-workspace#87 for provenance.
EXTRA_VECTORS_PATH = Path("/tmp/sawin_lenstra.json")
# Map from extra-vector label to canonical arXiv ID (Lenstra excluded — no arXiv).
EXTRA_LABEL_TO_ARXIV = {
    "sawin": "2605.20579",
    "openai_companion": "2605.20695",
}


def s2_batch_with_embeddings(arxiv_ids: list[str]) -> list[dict]:
    """POST to /paper/batch with ARXIV:<id> prefixes. Returns parallel list
    of dicts (or None) matching input order."""
    payload = {"ids": [f"ARXIV:{a}" for a in arxiv_ids]}
    fields = "paperId,title,externalIds,fieldsOfStudy,s2FieldsOfStudy,embedding.specter_v2"
    return s2_post("/paper/batch", payload, params={"fields": fields})


def process_batch(
    batch_idx: int, arxiv_ids: list[str], builder: RemaxBuilder
) -> tuple[list[dict], list[str]]:
    print(f"  batch {batch_idx}: requesting {len(arxiv_ids)} papers from S2…")
    t0 = time.time()
    resp = s2_batch_with_embeddings(arxiv_ids)
    print(f"    fetched in {time.time() - t0:.1f}s, got {sum(1 for x in resp if x)} non-null")

    vecs: list[list[float]] = []
    pids: list[str] = []
    meta: list[dict] = []
    missing: list[str] = []

    for arxiv_id, paper in zip(arxiv_ids, resp):
        if not paper:
            missing.append(arxiv_id)
            continue
        emb = (paper.get("embedding") or {}).get("vector")
        if not emb:
            missing.append(arxiv_id)
            continue
        pid = paper.get("paperId") or arxiv_id
        vecs.append(emb)
        pids.append(pid)
        meta.append({
            "paperId": pid,
            "arxiv_id": (paper.get("externalIds") or {}).get("ArXiv") or arxiv_id,
            "title": paper.get("title") or "",
            "fos": paper.get("fieldsOfStudy") or [],
            "s2_fos": [f.get("category") for f in (paper.get("s2FieldsOfStudy") or [])],
        })

    if vecs:
        v = np.asarray(vecs, dtype=np.float32)
        packed = builder.project_and_pack(v)
        builder.append(pids, packed)
        print(f"    embedded {len(vecs)}, missing {len(missing)}")

    return meta, missing


def main() -> None:
    include = load_json("include_ids.json")
    if not include:
        print("ERROR: data/include_ids.json missing; run stage1 first.", file=sys.stderr)
        sys.exit(1)
    ids: list[str] = include["all"]
    print(f"Phase A stage 2: SPECTER2 + remax build over {len(ids)} IDs")

    # Resume support.
    checkpoint = load_json("checkpoint.json", default={"last_batch": -1, "all_meta": [], "missing": []})
    last_batch = int(checkpoint.get("last_batch", -1))
    all_meta: list[dict] = list(checkpoint.get("all_meta") or [])
    missing: list[str] = list(checkpoint.get("missing") or [])

    builder = RemaxBuilder(d=SPECTER_DIM, k=REMAX_K, n_estimated=len(ids))

    # Replay processed batches into the builder so the packed codes are
    # contiguous on disk. Easiest: re-fetch from cache file if present;
    # otherwise rebuild from scratch.
    # For phase A scale (2 batches) just re-process everything if resuming
    # would skip codes — we keep this honest by always rebuilding codes.bin
    # in one pass when last_batch is at the end.
    if last_batch >= 0 and last_batch < (len(ids) - 1) // BATCH_SIZE:
        print(f"  resuming from batch {last_batch + 1}")
        # Re-fetch all completed batches' packed codes by re-running them.
        # At phase-A scale this is fine; phase B will need a real persistent
        # column store.
        for i, batch in enumerate(chunked(ids, BATCH_SIZE)):
            if i > last_batch:
                break
            print(f"  re-replaying batch {i} for code continuity…")
            process_batch(i, batch, builder)
            time.sleep(INTERBATCH_SLEEP)
        # all_meta / missing already loaded from checkpoint
        next_start = last_batch + 1
    else:
        next_start = 0

    for i, batch in enumerate(chunked(ids, BATCH_SIZE)):
        if i < next_start:
            continue
        batch_meta, batch_missing = process_batch(i, batch, builder)
        all_meta.extend(batch_meta)
        missing.extend(batch_missing)
        # Persist checkpoint after each batch.
        save_json("checkpoint.json", {
            "last_batch": i,
            "all_meta": all_meta,
            "missing": missing,
        })
        # Persist current packed codes too — overwrites OK, idempotent.
        codes = builder.partial_codes()
        (DATA / "codes.bin").write_bytes(codes.tobytes())
        print(f"  checkpoint: batch {i} done, codes.bin = {codes.shape}")
        time.sleep(INTERBATCH_SLEEP)

    # Inject extra vectors (Sawin + OpenAI companion — S2 doesn't have them).
    if EXTRA_VECTORS_PATH.exists():
        import json
        extra = json.loads(EXTRA_VECTORS_PATH.read_text())
        injected: list[str] = []
        vecs: list[list[float]] = []
        for label, arxiv_id in EXTRA_LABEL_TO_ARXIV.items():
            entry = extra.get(label)
            if not entry:
                continue
            if arxiv_id not in ids:
                continue  # not in the corpus
            # Was it already filled from S2? Compare against missing list.
            if arxiv_id not in missing:
                continue
            vecs.append(entry["vector"])
            injected.append(arxiv_id)
            all_meta.append({
                "paperId": f"ext:{arxiv_id}",
                "arxiv_id": arxiv_id,
                "title": entry.get("title") or "",
                "fos": ["Mathematics"],  # both Sawin + companion are math
                "s2_fos": [],
                "source": "phase0_external_specter2",
            })
        if vecs:
            v = np.asarray(vecs, dtype=np.float32)
            packed = builder.project_and_pack(v)
            builder.append([f"ext:{a}" for a in injected], packed)
            print(f"  injected {len(injected)} external SPECTER2 vectors: {injected}")
            # Drop them from missing list.
            missing = [m for m in missing if m not in injected]

    # Final outputs.
    codes = builder.partial_codes()
    (DATA / "codes.bin").write_bytes(codes.tobytes())
    save_json("metadata.json", all_meta)
    save_json("missing.json", {"n": len(missing), "ids": missing})
    save_json("codes_meta.json", {
        "n": codes.shape[0],
        "bytes_per_vec": codes.shape[1],
        "bits_per_vec": codes.shape[1] * 8,
        "d": SPECTER_DIM,
        "k": REMAX_K,
        "seed": builder.seed,
    })

    print(f"\nStage 2 complete:")
    print(f"  embedded:  {codes.shape[0]} papers ({codes.shape[1]} bytes/vec, {codes.shape[1] * 8} bits)")
    print(f"  missing:   {len(missing)} (no SPECTER2 in S2 cache)")
    print(f"  codes.bin: {(DATA / 'codes.bin').stat().st_size} bytes")


if __name__ == "__main__":
    main()
