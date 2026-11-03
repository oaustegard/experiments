"""Dry runs for a Monad chat page: which prompt framing and decoding settings make Monad behave
as a chat assistant? CPU, PyTorch, batched with left padding.

    python3 dryrun.py --model A8p --variants S0,S1 --decode greedy --out runs/r1.jsonl
    python3 dryrun.py --model monad --variants S1 --decode t0.7 --seeds 0,1 --out runs/r1.jsonl

`variants.json` holds the framings: an optional system turn, an optional text prefixed to the user
message, optional earlier turns. `probes.json` holds the questions; a probe may carry its own
earlier turns (multi-turn). `decodes` below: greedy, or temperature + top-p + a repetition penalty
on recently generated tokens, applied in that order. The page's sampler (monad-chat.js) does the
same arithmetic, so settings tuned here carry over.

Appends one JSON line per generation, so a crashed run resumes where it stopped.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

MWE = experiment("monad-weight-edit")
sys.path.insert(0, str(MWE))

import torch  # noqa: E402
from transformers import LogitsProcessor, LogitsProcessorList  # noqa: E402

from common import END, cut, load, loop_period, split_think  # noqa: E402

DECODES = {
    "greedy": None,
    "t0.7": {"temperature": 0.7, "top_p": 0.9, "penalty": 1.0, "window": 0},
    "t0.7r": {"temperature": 0.7, "top_p": 0.9, "penalty": 1.15, "window": 64},
    "t0.5r": {"temperature": 0.5, "top_p": 0.9, "penalty": 1.15, "window": 64},
    "t0.3r": {"temperature": 0.3, "top_p": 0.9, "penalty": 1.15, "window": 64},
    "gr": {"temperature": 0, "top_p": 1.0, "penalty": 1.15, "window": 64},   # greedy after the penalty (GPU Bridge only)
    "t1.0": {"temperature": 1.0, "top_p": 0.95, "penalty": 1.0, "window": 0},
}


class RecentPenalty(LogitsProcessor):
    """HF-style repetition penalty (positive logits / p, negative * p), but only over the last
    `window` generated tokens, never the prompt."""

    def __init__(self, L, penalty, window):
        self.L, self.p, self.w = L, penalty, window

    def __call__(self, input_ids, scores):
        gen = input_ids[:, self.L:]
        if self.p == 1.0 or gen.shape[1] == 0:
            return scores
        recent = gen[:, -self.w:]
        s = torch.gather(scores, 1, recent)
        s = torch.where(s > 0, s / self.p, s * self.p)
        return scores.scatter(1, recent, s)


def render(messages):
    s = "".join(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n" for m in messages)
    return s + "<|im_start|>assistant\n<think>\n"


def build(variant, probe):
    msgs = []
    if variant.get("system"):
        msgs.append({"role": "system", "content": variant["system"]})
    for u, a in variant.get("shots", []) + probe.get("history", []):
        msgs += [{"role": "user", "content": u}, {"role": "assistant", "content": a}]
    q = probe["q"]
    if variant.get("prefix"):
        q = variant["prefix"] + q
    if variant.get("suffix"):
        q = q + variant["suffix"]
    msgs.append({"role": "user", "content": q})
    return render(msgs)


@torch.no_grad()
def run(tok, model, prompts, dec, seed, max_new, batch):
    out = []
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        enc = tok(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        L = enc["input_ids"].shape[1]
        kw = dict(max_new_tokens=max_new, eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id,
                  stop_strings=[END], tokenizer=tok, cache_implementation="static")
        if dec is None:
            kw["do_sample"] = False
        else:
            torch.manual_seed(seed)
            kw.update(do_sample=True, temperature=dec["temperature"], top_p=dec["top_p"], top_k=0,
                      logits_processor=LogitsProcessorList([RecentPenalty(L, dec["penalty"], dec["window"])]))
        gen = model.generate(**enc, **kw)
        for row in gen[:, L:].tolist():
            out.append(cut(tok, row))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="A8p")
    ap.add_argument("--variants", required=True)
    ap.add_argument("--probes", default="")
    ap.add_argument("--probe-file", default="probes.json")
    ap.add_argument("--decode", default="greedy")
    ap.add_argument("--seeds", default="0")
    ap.add_argument("--max-new", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.set_num_threads(4)
    variants = json.loads((HERE / "variants.json").read_text())
    probes = json.loads((HERE / a.probe_file).read_text())
    if a.probes:
        keep = set(a.probes.split(","))
        probes = [p for p in probes if p["id"] in keep]
    out = HERE / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if out.exists():
        for l in out.open():
            r = json.loads(l)
            done.add((r["model"], r["variant"], r["decode"], r["seed"], r["probe"]))
    tok, model = load(MWE / "models" / a.model)
    for seed in [int(s) for s in a.seeds.split(",")]:
        jobs = [(v, p) for v in a.variants.split(",") for p in probes
                if (a.model, v, a.decode, seed, p["id"]) not in done]
        if not jobs:
            continue
        t0 = time.time()
        res = run(tok, model, [build(variants[v], p) for v, p in jobs], DECODES[a.decode], seed, a.max_new, a.batch)
        with out.open("a") as f:
            for (v, p), r in zip(jobs, res):
                text = tok.decode(r["ids"])
                think, ans, closed = split_think(text)
                f.write(json.dumps({"model": a.model, "variant": v, "decode": a.decode, "seed": seed, "probe": p["id"],
                                    "stopped": r["stopped"], "closed": closed, "loop": loop_period(r["ids"]) is not None
                                    and not r["stopped"], "tokens": len(r["ids"]), "answer": ans,
                                    "think": think}) + "\n")
        print(a.model, a.decode, seed, len(jobs), f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
