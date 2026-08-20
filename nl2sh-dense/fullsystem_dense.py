#!/usr/bin/env python3
"""End to end: does the retrieval lift reach the generated command?

`gemma_fullsystem.py` measured the fine-tuned Gemma 3 270M under three source
conditions — oracle 0.706, none 0.000, real BM25 retrieval 0.206 — and the gap
between the first and the third is what issue #48 exists to close. A retrieval
metric moving is not the deliverable; the routing number is. This runs the same
three conditions with the retriever swapped, so the 0.206 has a successor
measured the same way.

Everything except the retriever is copied from `gemma_fullsystem.py` unchanged:
the same prompt builder, the same k=3 distinct utilities each contributing their
first tldr example, the same greedy decode at 64 new tokens, the same
`utility_ok` scorer, the same seed. `oracle` and `none` are re-run rather than
quoted, because a different container and a different fine-tune of the same
recipe will not reproduce the old numbers to the third decimal, and comparing a
new `retrieval` against an old `oracle` would attribute that drift to retrieval.

    python3 fullsystem_dense.py --model ../nl2sh-retrieval/ft_gemma \\
        --tldr <tldr>/pages --retriever wsum0.7:bm25+minilm-l6-int8
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
from _lib.paths import experiment  # noqa: E402

RETRIEVAL = experiment("nl2sh-retrieval")
SELFHIST = experiment("nl2sh-selfhist")
sys.path.insert(0, str(RETRIEVAL))
import pleias_gate as G  # noqa: E402
import retrieve as R  # noqa: E402
from gemma_arm import build_user, _extract_command  # noqa: E402
import dense_index as D  # noqa: E402
import queries as Q  # noqa: E402


def parse_retriever(spec: str):
    """`bm25` | `dense:<model>` | `rrf:bm25+<model>` | `wsum<alpha>:bm25+<model>`."""
    if spec == "bm25":
        return "bm25", None, None
    kind, _, rest = spec.partition(":")
    model = rest.split("+")[-1]
    if kind == "dense":
        return "dense", model, None
    if kind == "rrf":
        return "rrf", model, None
    if kind.startswith("wsum"):
        return "wsum", model, float(kind[4:])
    raise SystemExit(f"unknown retriever spec: {spec}")


def make_ranker(spec: str, index: R.Index, dense, utilities, pool: int):
    kind, _, alpha = parse_retriever(spec)

    def rank(nl: str) -> list[str]:
        bs = index.scores(nl)
        bu = D.rank_utilities(bs, utilities, pool, positive_only=True)
        if kind == "bm25":
            return [u for u, _ in bu]
        du = D.rank_utilities(dense.scores(nl), utilities, pool)
        if kind == "dense":
            return [u for u, _ in du]
        if kind == "rrf":
            return [u for u, _ in D.rrf(bu, du)]
        return [u for u, _ in D.wsum(bu, du, alpha)]

    return rank


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, default=RETRIEVAL / "ft_gemma")
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--chunks", type=Path, default=D.DEFAULT_CHUNKS)
    ap.add_argument("--retriever", default="bm25")
    ap.add_argument("--adapter", type=Path, default=None,
                    help="query-side linear adapter from adapter.py")
    ap.add_argument("--granularity", default="chunk", choices=["chunk", "page"])
    ap.add_argument("--source-form", default="example", choices=["example", "page"],
                    help="what each retrieved utility contributes to the prompt: its "
                         "first tldr example (the shipped form) or its whole page")
    ap.add_argument("--nl", type=Path, nargs="+", default=list(Q.CYBER_NL))
    ap.add_argument("--modes", nargs="+", default=["oracle", "none", "retrieval"])
    ap.add_argument("-k", type=int, default=3)
    ap.add_argument("--pool", type=int, default=400)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--out", type=Path, default=HERE / "results_fullsystem_dense.json")
    a = ap.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tldr = G.load_tldr(a.tldr)
    data = Q.cyber(tldr, a.nl)
    chunks = D.GRANULARITIES[a.granularity](R.load_chunks(a.chunks))
    index = R.Index(chunks)
    utilities = np.array([c.utility for c in chunks], dtype=object)
    dense = None
    if a.retriever != "bm25":
        _, model_name, _ = parse_retriever(a.retriever)
        _, _, dense = D.load(model_name, a.chunks, granularity=a.granularity,
                             adapter=a.adapter)
    rank = make_ranker(a.retriever, index, dense, utilities, a.pool)

    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32).eval()
    rng = random.Random(a.seed)

    def render(u: str) -> str:
        """One source block for utility `u`.

        `example` is the shipped form: the first tldr example, one line. `page`
        is issue #48 item 4 — Gemma 3 270M has a 32k window where Pleias had 4k,
        so a whole page fits and the model sees every documented flag rather than
        one example's. The prompt still names the utility first either way, so
        `utility_ok` is scored against the same thing.
        """
        if a.source_form == "example":
            d, c = tldr[u][0]
            return f"{u} — {d}: {c}"
        body = "; ".join(f"{d}: {c}" for d, c in tldr[u])
        return f"{u} — {body}"

    def sources_for(nl: str) -> list[str]:
        out = []
        for u in rank(nl):
            if u not in tldr:
                continue
            out.append(render(u))
            if len(out) >= a.k:
                break
        return out

    def run(mode: str) -> dict:
        rows = []
        for r in data:
            gu = r["utility"]
            if mode == "oracle":
                others = [u for u in tldr if len(tldr[u]) >= 1 and u != gu]
                picks = [gu] + rng.sample(others, a.distractors)
                rng.shuffle(picks)
                srcs = [render(u) for u in picks]
            elif mode == "none":
                srcs = []
            else:
                srcs = sources_for(r["nl"])
            prompt = tok.apply_chat_template(
                [{"role": "user", "content": build_user(r["nl"], srcs)}],
                tokenize=False, add_generation_prompt=True)
            ids = tok(prompt, return_tensors="pt", add_special_tokens=False)
            with torch.no_grad():
                out = model.generate(**ids, max_new_tokens=64, do_sample=False,
                                     pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)
            cmd = _extract_command(gen)
            rows.append({"utility": gu, "nl": r["nl"], "command": cmd,
                         "names_utility": bool(r.get("names_utility")),
                         "utility_ok": bool(cmd) and G.gold_utility(cmd) == gu,
                         "gold_in_sources": any(s.split(" ")[0] == gu for s in srcs)})
        clean = [r for r in rows if not r["names_utility"]]
        return {"mode": mode, "n": len(rows), "n_leak_free": len(clean),
                "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / len(clean), 3),
                "command_rate": round(sum(bool(r["command"]) for r in rows) / len(rows), 3),
                "gold_in_sources_rate": round(sum(r["gold_in_sources"] for r in rows) / len(rows), 3),
                "rows": rows}

    out = {"retriever": a.retriever, "granularity": a.granularity,
           "source_form": a.source_form,
           "adapter": str(a.adapter) if a.adapter else None,
           "model": str(a.model), "n": len(data),
           "modes": {m: run(m) for m in a.modes}}
    a.out.write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nretriever: {a.retriever}   sources: {a.source_form}   n={len(data)}")
    print(f"{'mode':<12}{'routing (leak-free)':>22}{'gold in sources':>18}")
    print("-" * 52)
    for m in a.modes:
        s = out["modes"][m]
        print(f"{m:<12}{s['utility_acc_leak_free']:>22.3f}{s['gold_in_sources_rate']:>18.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
