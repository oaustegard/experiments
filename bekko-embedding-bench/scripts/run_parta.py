"""Run the mini-CTXBench arms and write results_parta.json."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bekko import BekkoEncoder  # noqa: E402
from eval_search import (  # noqa: E402
    approx_tokens,
    arm_dense,
    arm_rg,
    extract_identifiers,
    recall_at,
    rrf,
)

HERE = Path(__file__).resolve().parents[1]
DIM = 384


def load(mode: str, variant: str, n: int) -> np.ndarray:
    p = HERE / f"vecs_{mode}_{variant}.f32"
    return np.memmap(p, dtype=np.float32, mode="r", shape=(n, DIM))


def main() -> None:
    inst = json.load(open(HERE / "instances.json"))
    results = []

    # ── grep arm (chunking-independent) ─────────────────────────────────────
    rg_cache = {}
    for it in inst:
        q = it["title"] + "\n" + it["body"]
        idents = extract_identifiers(q)
        ranked, wall, chars = arm_rg(idents)
        rg_cache[it["issue"]] = (ranked, wall, chars, idents)
        print(f"rg #{it['issue']}: {len(idents)} idents, {len(ranked)} files, {wall:.2f}s",
              flush=True)

    for mode in ("ast", "flat"):
        chunks = json.load(open(HERE / f"chunks_{mode}.json"))
        n = len(chunks)
        for variant in ("a8m", "a25m"):
            done = HERE / f"vecs_{mode}_{variant}.done"
            if not done.exists() or int(done.read_text()) < n:
                print(f"SKIP {mode}/{variant}: incomplete", flush=True)
                continue
            mat = np.asarray(load(mode, variant, n))
            enc = BekkoEncoder(variant, threads=4)
            for it in inst:
                q = it["title"] + "\n" + it["body"]
                t0 = time.time()
                qv = enc.encode([q])[0]
                dranked, backing = arm_dense(qv, mat, chunks)
                dwall = time.time() - t0
                rranked, rwall, rchars, idents = rg_cache[it["issue"]]
                fused = rrf(rranked, dranked)
                # honest token accounting: what an agent ingests for top-10
                dchars = sum(len(chunks[i]["text"]) for i in backing[:10])
                gold = it["gold"]
                row = {
                    "issue": it["issue"], "mode": mode, "variant": variant,
                    "n_idents": len(idents), "n_gold": len(gold),
                    "rg_r5": recall_at(rranked, gold, 5),
                    "rg_r10": recall_at(rranked, gold, 10),
                    "dense_r5": recall_at(dranked, gold, 5),
                    "dense_r10": recall_at(dranked, gold, 10),
                    "rrf_r5": recall_at(fused, gold, 5),
                    "rrf_r10": recall_at(fused, gold, 10),
                    "rg_wall": rwall, "dense_wall": dwall,
                    "rg_tokens": approx_tokens(rchars),
                    "dense_tokens": approx_tokens(dchars),
                }
                results.append(row)
                print(
                    f"{mode}/{variant} #{it['issue']}: rg {row['rg_r5']:.2f}/{row['rg_r10']:.2f} "
                    f"dense {row['dense_r5']:.2f}/{row['dense_r10']:.2f} "
                    f"rrf {row['rrf_r5']:.2f}/{row['rrf_r10']:.2f}",
                    flush=True,
                )
            del mat
    json.dump(results, open(HERE / "results_parta.json", "w"), indent=1)
    print("wrote results_parta.json", flush=True)


if __name__ == "__main__":
    main()
