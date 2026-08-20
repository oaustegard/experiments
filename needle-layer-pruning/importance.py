#!/usr/bin/env python3
"""Can a cheap redundancy score find the prunable block without 24 evaluations?

The standard depth-pruning heuristic (Gromov et al. 2024; ShortGPT) scores a
block of `n` layers by how little it changes the residual stream: the angular
distance between the representation entering layer `s` and the one leaving layer
`s+n-1`. Low distance means the block is close to the identity and should be the
one to delete.

This computes that score for every candidate block on the same 62 queries the
sweep evaluates, and reports its rank correlation with the accuracy the sweep
actually measured. One forward pass, no engine, no `.cact`.

    python3 importance.py --count 4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

CHECKPOINT = "checkpoints/needle2.pkl"


def spearman(a, b) -> float:
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--count", type=int, default=4)
    ap.add_argument("--checkpoint", default=CHECKPOINT)
    ap.add_argument("--max-queries", type=int, default=62)
    a = ap.parse_args()

    import jax.numpy as jnp
    from needle.model.run import load_checkpoint
    from needle.model.architecture import SimpleAttentionNetwork
    from needle.model.tokenizer import get_tokenizer

    params, cfg = load_checkpoint(a.checkpoint)
    model = SimpleAttentionNetwork(cfg)
    tok = get_tokenizer(cfg.vocab_size)

    items = [json.loads(l) for l in
             (experiment("needle-bsky") / "evalset.jsonl").read_text().splitlines() if l.strip()]
    queries = [it["query"] for it in items][: a.max_queries]

    # hidden[i] is the representation leaving layer i; the input to layer 0 is
    # the embedding, so a block starting at 0 is scored against that instead.
    sims = np.zeros((cfg.num_layers - a.count + 1,), np.float64)
    n_tok = 0
    for q in queries:
        ids = jnp.asarray([tok.encode(q)], jnp.int32)
        h = np.asarray(model.apply({"params": params}, ids, method=SimpleAttentionNetwork.hidden_states))
        emb = np.asarray(model.apply({"params": params}, ids,
                                     method=lambda m, t: m.embedding(t) * m.embed_scale), np.float32)
        h = np.concatenate([emb[None], h], axis=0)  # h[i] = input to layer i
        t = h.shape[2]
        for s in range(sims.shape[0]):
            x, y = h[s, 0], h[s + a.count, 0]
            cos = (x * y).sum(-1) / (np.linalg.norm(x, axis=-1) * np.linalg.norm(y, axis=-1) + 1e-9)
            sims[s] += cos.sum()
        n_tok += t
    sims /= n_tok
    angular = np.arccos(np.clip(sims, -1, 1)) / np.pi

    measured = {}
    for p in HERE.glob(f"results_cut{a.count}_at*.json"):
        tail = p.stem.split("_at")[1]
        if not tail.isdigit():  # the re-timed duplicates carry a suffix
            continue
        measured[int(tail)] = json.loads(p.read_text())["summary"]["tool_acc_routable"]

    starts = sorted(measured)
    print(f"{'block':<12} {'angular dist':>13} {'heuristic rank':>15} {'measured':>10} {'actual rank':>12}")
    heur_rank = {s: r for r, s in enumerate(sorted(starts, key=lambda s: angular[s]), 1)}
    act_rank = {s: r for r, s in enumerate(sorted(starts, key=lambda s: -measured[s]), 1)}
    for s in starts:
        print(f"[{s:2},{s + a.count:2})      {angular[s]:13.4f} {heur_rank[s]:15} "
              f"{measured[s]:10.3f} {act_rank[s]:12}")

    rho = spearman([-angular[s] for s in starts], [measured[s] for s in starts])
    best_h = min(starts, key=lambda s: angular[s])
    best_m = max(starts, key=lambda s: measured[s])
    print(f"\nSpearman(heuristic, measured accuracy) = {rho:+.3f} over {len(starts)} blocks")
    print(f"heuristic picks [{best_h},{best_h + a.count}) -> measured {measured[best_h]:.3f}")
    print(f"sweep's best is [{best_m},{best_m + a.count}) -> measured {measured[best_m]:.3f}")

    (HERE / f"importance_{a.count}.json").write_text(json.dumps({
        "count": a.count, "angular": {str(s): float(angular[s]) for s in starts},
        "measured": {str(s): measured[s] for s in starts},
        "spearman": rho, "heuristic_pick": best_h, "measured_best": best_m}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
