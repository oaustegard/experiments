#!/usr/bin/env python3
"""llama.cpp sibling of run_gen.py — same eval, same prompts, a GGUF model.

Stage 1 (`run_gen.py`) runs `transformers` on fp32/bf16 weights. Issue #52
tabulates candidates by *quantised* size and wants a laptop-realistic number,
so this script holds every experimental variable fixed at stage 1's settings
— oracle sources, seed, greedy decode, max_new_tokens, the fenced-or-bare
command parser — and swaps only the runtime and the weight format: llama.cpp
over a Q4_K_M (or similar) GGUF instead of `transformers` over the full-size
checkpoint.

Reuses `run_gen.py`'s `extract_command`, `load_eval`, `make_sources` verbatim
(imported, not re-typed) and `pleias_gate.load_tldr` / `pleias_gate.gold_utility`
exactly as `run_gen.py` does. The only new code is the model load/download and
the generation call, which goes through `Llama.create_chat_completion` so the
GGUF's own embedded chat template is what renders the prompt (see the
`chat_format` / `chat_template_embedded` fields in the summary — those record
which path llama.cpp actually took, since a GGUF without an embedded template
falls back to a hardcoded "llama-2" format that would silently misrender a
Gemma or Nemotron prompt).

    python3 run_gen_gguf.py --condition generate \
        --model unsloth/gemma-3-270m-it-GGUF --gguf-file gemma-3-270m-it-Q4_K_M.gguf \
        --tldr /path/to/tldr/pages --out results_gguf_270m_generate.json --limit 20
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# Low-priority tenant: pin OpenMP/BLAS threading before llama_cpp (or anything
# it links against) reads the env, regardless of what the caller's shell set.
os.environ.setdefault("OMP_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
RETRIEVAL = HERE.parent / "nl2sh-retrieval"
sys.path.insert(0, str(RETRIEVAL))
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402
import pleias_gate as G  # noqa: E402 - reused exactly as run_gen.py uses it
from run_gen import extract_command, load_eval, make_sources, DEFAULT_EVALS  # noqa: E402

DEFAULT_CACHE = Path.home() / ".cache" / "nl2sh-gguf"


def download_gguf(repo_id: str, filename: str, cache_dir: Path) -> Path:
    """hf_hub_download if available, else a plain curl -L. No auth header —
    every model this script targets is ungated (verified by the recon worker
    with ranged 1-byte fetches returning 206)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / filename
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id=repo_id, filename=filename, local_dir=cache_dir)
        return Path(path)
    except ImportError:
        url = f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
        print(f"huggingface_hub unavailable, curl fallback: {url}", file=sys.stderr)
        subprocess.run(["curl", "-L", "-o", str(dest), url], check=True)
        return dest


THINK_CLOSE = "</think>"


def strip_reasoning(gen: str) -> str:
    """Drop a reasoning trace so the shared parser reads the answer, not the thinking.

    `run_gen.py`'s `extract_command` takes the first line that does not open with
    a hedge, which on a reasoning model is the first line of its scratchpad. Text
    after a closing `</think>` is the answer; an unterminated trace means the
    budget ran out before the model answered at all, and the empty string is the
    honest reading of that — not the last line of its reasoning.
    """
    if THINK_CLOSE in gen:
        return gen.split(THINK_CLOSE)[-1]
    if "<think>" in gen:
        return ""
    return gen


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="unsloth/gemma-3-270m-it-GGUF",
                    help="HF repo id holding the GGUF file")
    ap.add_argument("--gguf-file", required=True, help="filename within --model")
    ap.add_argument("--condition", required=True, choices=sorted(prompts.BUILDERS))
    ap.add_argument("--tldr", type=Path, required=True)
    ap.add_argument("--nl", type=Path, nargs="*", default=DEFAULT_EVALS)
    ap.add_argument("--distractors", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260819)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n-threads", type=int, default=1,
                    help="low-priority tenant default: 1 thread")
    ap.add_argument("--n-ctx", type=int, default=4096)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE,
                    help="where downloaded GGUFs land (never inside the repo)")
    ap.add_argument("--reasoning", default="auto", choices=["auto", "off", "on"],
                    help="Nemotron 3 Nano and its kin emit a <think> trace before the "
                         "answer. Under stage 1's 64-token budget that trace IS the whole "
                         "budget: the 20-row probe scored 0.000 routing with every row "
                         "truncated mid-reasoning. 'off' prepends the /no_think system turn "
                         "these models document, which is what makes them comparable to the "
                         "non-reasoning bases. 'auto' means off for a model whose name says "
                         "nemotron, untouched otherwise.")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import random
    from llama_cpp import Llama

    gguf_path = download_gguf(a.model, a.gguf_file, a.cache_dir)
    gguf_size = gguf_path.stat().st_size

    tldr = G.load_tldr(a.tldr)
    data = load_eval(list(a.nl), tldr)[: a.limit]

    llm = Llama(
        model_path=str(gguf_path),
        n_ctx=a.n_ctx,
        n_threads=a.n_threads,
        n_threads_batch=a.n_threads,
        seed=a.seed,
        verbose=False,
    )
    chat_template_embedded = "tokenizer.chat_template" in llm.metadata
    chat_format = llm.chat_format
    print(f"chat_format={chat_format} embedded_template={chat_template_embedded}",
          file=sys.stderr)

    build = prompts.BUILDERS[a.condition]
    rng = random.Random(a.seed)

    reasoning = a.reasoning
    if reasoning == "auto":
        reasoning = "off" if "nemotron" in a.model.lower() else "on"
    msgs = [{"role": "system", "content": "/no_think"}] if reasoning == "off" else []

    rows, t_start = [], time.perf_counter()
    for r in data:
        srcs = make_sources(r["utility"], tldr, rng, a.distractors)
        user = build(r["nl"], srcs)
        t0 = time.perf_counter()
        resp = llm.create_chat_completion(
            messages=msgs + [{"role": "user", "content": user}],
            max_tokens=a.max_new_tokens,
            temperature=0.0,      # greedy: llama.cpp's argmax path at temp==0
            repeat_penalty=1.0,   # no repetition penalty, matching stage 1's default
            seed=a.seed,
        )
        dt = time.perf_counter() - t0
        usage = resp.get("usage", {})
        n_prompt = usage.get("prompt_tokens", 0)
        n_new = usage.get("completion_tokens", 0)
        gen = (resp["choices"][0]["message"]["content"] or "")
        cmd = extract_command(strip_reasoning(gen))
        rows.append({**r, "gold_cmd": r["cmd"], "command": cmd, "raw": gen.strip()[:400],
                     "utility_ok": bool(cmd) and G.gold_utility(cmd) == r["utility"],
                     "sources": srcs, "literals": prompts.literals(r["nl"]),
                     "prompt_tokens": n_prompt, "new_tokens": n_new,
                     "seconds": round(dt, 2)})
        print(f"{'OK ' if rows[-1]['utility_ok'] else '   '}"
              f"{'[leak]' if r.get('names_utility') else '      '} "
              f"{r['utility']:<12} {cmd[:56]}", flush=True)

    clean = [r for r in rows if not r.get("names_utility")]
    tot_new = sum(r["new_tokens"] for r in rows)
    summary = {
        "model": str(a.model), "condition": a.condition, "dtype": "gguf", "sources": "oracle",
        "distractors": a.distractors, "seed": a.seed, "n": len(rows),
        "n_leak_free": len(clean),
        "utility_acc_all": round(sum(r["utility_ok"] for r in rows) / len(rows), 3),
        "utility_acc_leak_free": round(sum(r["utility_ok"] for r in clean) / len(clean), 3),
        "command_rate": round(sum(bool(r["command"]) for r in rows) / len(rows), 3),
        "mean_prompt_tokens": round(sum(r["prompt_tokens"] for r in rows) / len(rows), 1),
        "mean_new_tokens": round(tot_new / len(rows), 1),
        "mean_seconds": round(sum(r["seconds"] for r in rows) / len(rows), 2),
        "decode_tok_per_s": round(tot_new / sum(r["seconds"] for r in rows), 1),
        "wall_minutes": round((time.perf_counter() - t_start) / 60, 1),
        "runner": "llama.cpp", "quant": a.gguf_file, "gguf_file": a.gguf_file,
        "gguf_bytes": gguf_size, "n_threads": a.n_threads, "n_ctx": a.n_ctx,
        "reasoning": reasoning,
        "chat_format": chat_format, "chat_template_embedded": chat_template_embedded,
    }
    a.out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=1) + "\n")
    print("\n" + json.dumps(summary, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
