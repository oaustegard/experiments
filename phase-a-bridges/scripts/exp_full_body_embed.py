"""Experiment 2 — Embed full 997-paper corpus in Gemini-body space.

Phase A only embedded the 133 papers in the candidate set. To answer
"does Gemini-body do what SPECTER2 couldn't?" at corpus scale, fetch and
embed the remaining ~864 papers, then check where the known anchor pairs
rank in pure Gemini-body cosine space against the full 996² candidate set.

The data dir is selectable via PHASE_A_DATA_DIR for run-2 reuse.
Outputs (in DATA):
  full_body_texts.json       — {arxiv_id: body_text}
  full_body_embeddings.json  — {arxiv_id: gemini vector}  (or null on failure)
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA, gemini_embed, load_json, save_json  # noqa: E402
from stage4_rerank import body_window, fetch_body  # noqa: E402


CHECKPOINT_EVERY = 25
FETCH_SLEEP = 0.4


def main() -> None:
    meta = load_json("metadata.json")
    if not meta:
        print("ERROR: metadata.json missing — run stage2 first.", file=sys.stderr)
        sys.exit(1)
    all_arxiv = [m["arxiv_id"] for m in meta if m.get("arxiv_id")]
    print(f"Full-corpus body embed: {len(all_arxiv)} papers")

    # Seed from prior partial caches if they exist.
    full_texts = load_json("full_body_texts.json", default={}) or {}
    full_embs = load_json("full_body_embeddings.json", default={}) or {}
    # Also seed from the candidate-set caches (free reuse).
    partial_texts = load_json("body_texts.json", default={}) or {}
    partial_embs = load_json("body_embeddings.json", default={}) or {}
    for k, v in partial_texts.items():
        full_texts.setdefault(k, v)
    for k, v in partial_embs.items():
        full_embs.setdefault(k, v)

    pending = [a for a in all_arxiv if a not in full_embs]
    print(f"  cached: {len(all_arxiv) - len(pending)}  to process: {len(pending)}")

    for n, arxiv_id in enumerate(pending, 1):
        t0 = time.time()
        # Fetch body if not cached.
        if arxiv_id in full_texts and full_texts[arxiv_id]:
            body_text = full_texts[arxiv_id]
        else:
            body = fetch_body(arxiv_id)
            body_text = body_window(body)
            full_texts[arxiv_id] = body_text

        if not body_text or len(body_text) < 200:
            full_embs[arxiv_id] = None
            print(f"  [{n}/{len(pending)}] {arxiv_id}: body too short ({len(body_text)}), null embedding")
        else:
            try:
                v = gemini_embed(body_text)
                full_embs[arxiv_id] = v
                print(f"  [{n}/{len(pending)}] {arxiv_id}: ok len={len(body_text)} ({time.time() - t0:.1f}s)")
            except Exception as e:
                full_embs[arxiv_id] = None
                print(f"  [{n}/{len(pending)}] {arxiv_id}: embed err {type(e).__name__}: {e}")

        if n % CHECKPOINT_EVERY == 0:
            save_json("full_body_texts.json", full_texts)
            save_json("full_body_embeddings.json", full_embs)
            print(f"  checkpoint @ {n}")
        time.sleep(FETCH_SLEEP)

    save_json("full_body_texts.json", full_texts)
    save_json("full_body_embeddings.json", full_embs)

    ok = sum(1 for v in full_embs.values() if v)
    print(f"\nDone: {ok} / {len(full_embs)} papers have valid Gemini-body embeddings")


if __name__ == "__main__":
    main()
