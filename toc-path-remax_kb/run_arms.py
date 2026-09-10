"""Build a .kb per arm with the real remax_kb writer and score every arm on the
identical query set.

Uses remax_kb's own KBWriter/KB rather than a reimplementation, so the fusion
being measured is the one that ships. Per-modality readings come from the
retrieval legs directly (`_dense_search`, `_bm25_search`) so a single-modality
number is that modality's true ranking, not a weighted fusion approximating it.
"""
from __future__ import annotations

import json
import math
import shutil
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/user/remax_kb")

from remax_kb.embedders import JinaQ4ONNXEmbedder
from remax_kb.pack import Chunk
from remax_kb.pack_v2 import KBWriter
from remax_kb.read_v2 import KB

HERE = Path(__file__).parent
DATA = HERE / "data"
KBS = HERE / "kbs"
ARMS = ["A", "B0", "B1", "B2"]
KS = (1, 3, 5, 10)
OVER_FETCH = 200
BATCH = 16


class BatchedJinaQ4(JinaQ4ONNXEmbedder):
    """KBWriter.commit() encodes every pending chunk in one call, which builds an
    attention mask over the whole corpus (23 GB at 1,871 chunks). Batch inside
    encode() instead. Subclassed rather than wrapped so fingerprint(), prompts
    and full_dim stay identical to the embedder the reader validates against.
    """

    def encode(self, texts: list[str], *, prompt: str) -> np.ndarray:
        out = [
            super(BatchedJinaQ4, self).encode(texts[i:i + BATCH], prompt=prompt)
            for i in range(0, len(texts), BATCH)
        ]
        return np.vstack(out) if len(out) > 1 else out[0]


def load_chunks(arm: str) -> list[dict]:
    return [json.loads(l) for l in (DATA / f"chunks_{arm}.jsonl").read_text().splitlines()]


def build_kb(arm: str, embedder) -> Path:
    out = KBS / arm
    kbi = out / f"{arm}.kbi"
    if kbi.exists():
        return kbi
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    recs = load_chunks(arm)
    w = KBWriter.create(name=arm, output_dir=out, embedder=embedder)
    w.add_chunks([
        Chunk(id=r["id"], text=r["text"],
              meta={"source_path": r["source_path"], "heading_path": r["heading_path"]})
        for r in recs
    ])
    w.commit()
    print(f"  built {arm}: {len(recs)} chunks -> {kbi.name}")
    return kbi


def b2_gold_map(gold: list[dict], b2: list[dict], min_overlap: int = 60) -> dict[str, set[str]]:
    """Map each arm-A gold chunk to the arm-B2 chunk(s) covering its text.

    B2 re-chunks at heading boundaries, so an A chunk can map to more than one
    B2 chunk. That asymmetry favours B2 and is reported alongside its numbers.
    """
    by_file: dict[str, list[dict]] = {}
    for c in b2:
        by_file.setdefault(c["source_path"], []).append(c)

    out: dict[str, set[str]] = {}
    for g in gold:
        ga = " ".join(g["text"].split())
        hits: set[str] = set()
        for c in by_file.get(g["source_path"], []):
            body = c["text"].split("\n\n", 1)[1] if c["heading_path"] else c["text"]
            cb = " ".join(body.split())
            m = SequenceMatcher(None, ga, cb, autojunk=False).find_longest_match(
                0, len(ga), 0, len(cb))
            if m.size >= min_overlap:
                hits.add(c["id"])
        out[g["id"]] = hits
    return out


def rank_of(ranked_ids: list[str], gold_ids: set[str]) -> int | None:
    for i, cid in enumerate(ranked_ids):
        if cid in gold_ids:
            return i + 1
    return None


def score(ranks: list[int | None]) -> dict:
    n = len(ranks)
    out = {f"R@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in KS}
    out["nDCG@3"] = sum(
        (1.0 / math.log2(r + 1)) for r in ranks if r is not None and r <= 3
    ) / n
    out["MRR"] = sum((1.0 / r) for r in ranks if r is not None) / n
    return out


def run_queries(kb: KB, embedder, queries: list[str], gold_sets: list[set[str]]) -> dict:
    modes = {"dense": [], "bm25": [], "fused": []}
    for q, gold in zip(queries, gold_sets):
        d = kb._dense_search(q, embedder)
        modes["dense"].append(rank_of([h.chunk_id for h in d[:50]], gold))
        b = kb._bm25_search(q)
        modes["bm25"].append(rank_of([h.chunk_id for h in b[:50]], gold))
        f = kb.search(q, embedder=embedder, k=10, over_fetch=OVER_FETCH)
        modes["fused"].append(rank_of([h.chunk_id for h in f], gold))
    return modes


def main() -> None:
    gold = [json.loads(l) for l in (DATA / "gold_chunks.jsonl").read_text().splitlines()]
    qrecs = [json.loads(l) for l in (DATA / "queries.jsonl").read_text().splitlines()]
    print(f"{len(gold)} gold chunks, {len(qrecs)} queries")

    b2 = load_chunks("B2")
    b2map = b2_gold_map(gold, b2)
    sizes = [len(v) for v in b2map.values()]
    unmapped = sum(1 for v in b2map.values() if not v)
    print(f"B2 gold sets: mean {sum(sizes)/len(sizes):.2f} chunks, {unmapped} unmapped")

    embedder = BatchedJinaQ4()
    KBS.mkdir(exist_ok=True)

    queries = [r["query"] for r in qrecs]
    gold_a = [{r["gold_id"]} for r in qrecs]
    gold_b2 = [b2map.get(r["gold_id"], set()) for r in qrecs]

    # Positive control: the heading path itself as the query.
    gold_by_id = {g["id"]: g for g in gold}
    ctrl_pairs = [(gold_by_id[g]["heading_path"], {g}) for g in
                  {r["gold_id"] for r in qrecs} if gold_by_id[g]["heading_path"]]
    ctrl_q = [p[0] for p in ctrl_pairs]
    ctrl_gold = [p[1] for p in ctrl_pairs]
    ctrl_gold_b2 = [b2map.get(next(iter(p[1])), set()) for p in ctrl_pairs]
    print(f"positive control: {len(ctrl_q)} heading-path queries")

    results = {"n_queries": len(queries), "n_control": len(ctrl_q),
               "b2_gold_mean": sum(sizes) / len(sizes), "arms": {}, "control": {},
               "ranks": {}, "control_ranks": {}}

    # Checkpoint per arm: a reaped or timed-out run resumes at the next arm
    # instead of re-embedding everything.
    for arm in ARMS:
        ckpt = HERE / f"ckpt_{arm}.json"
        if ckpt.exists():
            print(f"arm {arm}: checkpoint present, skipping", flush=True)
            continue
        print(f"arm {arm}:", flush=True)
        kbi = build_kb(arm, embedder)
        kb = KB.open(kbi)
        g = gold_b2 if arm == "B2" else gold_a
        modes = run_queries(kb, embedder, queries, g)
        cg = ctrl_gold_b2 if arm == "B2" else ctrl_gold
        cmodes = run_queries(kb, embedder, ctrl_q, cg)
        payload = {
            "arm": arm,
            "scores": {m: score(r) for m, r in modes.items()},
            "control_scores": {m: score(r) for m, r in cmodes.items()},
            "ranks": modes,
            "control_ranks": cmodes,
        }
        ckpt.write_text(json.dumps(payload, indent=1))
        f, c = payload["scores"]["fused"], payload["control_scores"]["fused"]
        print(f"  fused R@1 {f['R@1']:.3f}  R@10 {f['R@10']:.3f} | "
              f"control R@1 {c['R@1']:.3f}  -> {ckpt.name}", flush=True)

    for arm in ARMS:
        p = json.loads((HERE / f"ckpt_{arm}.json").read_text())
        results["arms"][arm] = p["scores"]
        results["control"][arm] = p["control_scores"]
        results["ranks"][arm] = p["ranks"]
        results["control_ranks"][arm] = p["control_ranks"]

    (HERE / "results.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {HERE / 'results.json'}", flush=True)


if __name__ == "__main__":
    main()
