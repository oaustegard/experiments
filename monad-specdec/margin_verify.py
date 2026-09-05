"""Margin-based lossy verification on the Baguettotron EAGLE drafter.

Implements the AdaptiveSpec verification rule (arXiv 2609.02897, Urban et al.,
Samsung AI Cambridge) on top of the existing end-to-end harness in
eagle_e2e2.py. At each candidate position the target's own distribution decides
whether a mismatched draft token survives:

    margin(k) = p_target(draft_k) / p_target(top1_k)
              = exp(logit[k, draft_k] - logit[k, top1_k])

Accept when margin(k) >= kappa. An exact match has margin exactly 1.0, so the
one condition covers both regimes and kappa=1.0 IS the lossless control -- the
same code path produces the baseline, which is why the lossless row here can be
compared against the lossy rows without a second implementation.

The paper's tree-shaping axis is NOT implemented. It presupposes tree attention,
a verify-token budget and one pre-captured CUDA graph per (n_steps, top-k, ndt)
triplet; this harness drafts a chain on CPU and has none of that machinery.

Two promotion modes, because the paper's wording ("the first mismatch position")
does not say whether verification continues after a promotion:
  first -- at most one promoted token per verify round
  all   -- the margin test runs at every mismatch in the block

Quality is reported as drift from the greedy path, not as task accuracy.
Baguettotron is a 321M reasoning model; GSM8K on it would measure the model,
not the verifier. The two axes recorded instead are the length of the common
prefix with pure greedy decoding, and the target's own mean log-probability of
what was emitted.

Usage: margin_verify.py [head_checkpoint]
"""
import json
import os
import sys
import time

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/root/.claude/skills/flowing/scripts")

from flowing import Flow, task

from eagle_train import EagleHead

torch.set_num_threads(4)

C_MEASURED = 0.049  # draft/target step cost ratio, measured in eagle_cost.py
MAX_NEW = 48
GAMMAS = (1, 2, 3)
KAPPAS = (1.0, 0.5, 0.3, 0.2, 0.1, 0.05)
MODES = ("first", "all")
CKPT = sys.argv[1] if len(sys.argv) > 1 else "eagle_head_s8.pt"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "margin_verify.json")

QUERIES = [
    "Why does bread rise when yeast is added?",
    "How did Roman engineers keep aqueducts flowing across valleys?",
    "What does a mitochondrion actually do in a cell?",
    "Write a Python function that returns the nth Fibonacci number.",
]


@task
def env_check():
    """Fail before a 20-minute sweep rather than during it."""
    if not os.path.exists(CKPT):
        raise FileNotFoundError(f"head checkpoint missing: {CKPT}")
    return {"torch": torch.__version__, "threads": torch.get_num_threads(),
            "ckpt": CKPT, "ckpt_bytes": os.path.getsize(CKPT)}


_M = {}


def _ensure_model():
    """Idempotent model load.

    Called by every task that touches weights rather than relying on a
    side-effect of load(): the journal replays a task's RETURN VALUE, so on a
    resumed run load() never executes and anything it stashed in a global would
    be missing.
    """
    if _M:
        return _M
    cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
    tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
    target = AutoModelForCausalLM.from_pretrained(
        "PleIAs/Baguettotron", dtype=torch.float32).eval()
    head = EagleHead(cfg)
    head.load_state_dict(torch.load(CKPT))
    head.eval()
    _M.update({"tok": tok, "target": target, "head": head,
               "embed": target.model.embed_tokens, "lm_head": target.lm_head})
    return _M


@task(depends_on=[env_check], timeout_s=900)
def load(env_check):
    m = _ensure_model()
    return {"n_prompts": len(QUERIES), "prompt_lens": [len(p) for p in _prompts()],
            "vocab": m["target"].config.vocab_size}


def _prompts():
    tok = _ensure_model()["tok"]
    return [tok(tok.apply_chat_template([{"role": "user", "content": q}],
                                        tokenize=False, add_generation_prompt=True),
                add_special_tokens=False).input_ids for q in QUERIES]


def _mean_logprob(ids, n0):
    """Target's own mean log-prob of the generated continuation, teacher-forced."""
    target = _ensure_model()["target"]
    with torch.no_grad():
        logits = target(torch.tensor([ids])).logits[0]
    lp = torch.log_softmax(logits[n0 - 1:-1].float(), dim=-1)
    tgt = torch.tensor(ids[n0:])
    return float(lp.gather(1, tgt.unsqueeze(1)).mean())


def baseline(ids, n=None):
    n = MAX_NEW if n is None else n
    target = _ensure_model()["target"]
    ids = list(ids)
    n0 = len(ids)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target(torch.tensor([ids]), use_cache=True)
        past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
        while len(ids) - n0 < n:
            ids.append(nxt)
            out = target(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
            past, nxt = out.past_key_values, int(out.logits[0, -1].argmax())
    return ids, time.perf_counter() - t0


def spec(ids0, gamma, kappa, mode, n=None):
    """Speculative decode with margin-based acceptance.

    kappa=1.0 reproduces strict exact-match verification, because an exact
    match scores margin exactly 1.0.
    """
    n = MAX_NEW if n is None else n
    m = _ensure_model()
    target, head, embed, lm_head = m["target"], m["head"], m["embed"], m["lm_head"]
    ids = list(ids0)
    n0 = len(ids)
    proposed = accepted = promoted = 0
    margins = []
    mismatch_margins = []
    t0 = time.perf_counter()
    with torch.no_grad():
        out = target(torch.tensor([ids]), use_cache=True, output_hidden_states=True)
        past = out.past_key_values
        feats = out.hidden_states[-1][0]
        nxt = int(out.logits[0, -1].argmax())
        htoks = ids[1:] + [nxt]
        while len(ids) - n0 < n:
            f_seq, t_seq, draft = feats, list(htoks), []
            for _ in range(gamma):
                a = head(f_seq.unsqueeze(0), embed(torch.tensor([t_seq])))
                d = int(lm_head(a[:, -1:])[0, -1].argmax())
                draft.append(d)
                f_seq = torch.cat([f_seq, a[0, -1:]], dim=0)
                t_seq = t_seq + [d]
            proposed += len(draft)
            block = [nxt] + draft
            out = target(torch.tensor([block]), past_key_values=past,
                         use_cache=True, output_hidden_states=True)
            past = out.past_key_values
            logits = out.logits[0]
            am = logits.argmax(-1)
            hs = out.hidden_states[-1][0]
            ids.append(nxt)

            budget = 1 if mode == "first" else gamma
            k = 0
            while k < len(draft) and len(ids) - n0 < n:
                top1 = int(am[k])
                if top1 == draft[k]:
                    ids.append(draft[k])
                    k += 1
                    continue
                margin = float(torch.exp(logits[k, draft[k]] - logits[k, top1]))
                mismatch_margins.append(margin)
                if budget > 0 and margin >= kappa:
                    margins.append(margin)
                    promoted += 1
                    budget -= 1
                    ids.append(draft[k])
                    k += 1
                    continue
                break
            accepted += k
            if len(ids) - n0 >= n:
                break
            new_nxt = int(am[k])
            feats = torch.cat([feats, hs[:k + 1]], dim=0)
            htoks = htoks + block[1:k + 1] + [new_nxt]
            nxt = new_nxt
            past.crop(len(ids))
    return {"ids": ids, "seconds": time.perf_counter() - t0,
            "acceptance": accepted / proposed if proposed else 0.0,
            "promoted": promoted,
            "mean_promoted_margin": sum(margins) / len(margins) if margins else None,
            "mismatch_margins": mismatch_margins}


def projected(a, g, c=C_MEASURED):
    """Speedup once the draft head keeps its own KV cache (it currently
    re-runs its whole prefix each round, so wall-clock understates)."""
    return (1 - a ** (g + 1)) / ((1 - a) * (1 + g * c))


def _prefix_len(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@task(depends_on=[load], timeout_s=1800)
def baselines(load):
    """Greedy reference per prompt: the path every lossy run is measured against."""
    rows = []
    for pid in _prompts():
        ids, sec = baseline(pid)
        rows.append({"n0": len(pid), "ids": ids, "seconds": sec,
                     "mean_logprob": _mean_logprob(ids, len(pid))})
    return rows


def _has_baselines(baselines, **_):
    """validate= is handed every gathered dep, not just the one it names."""
    if not baselines or any(len(r["ids"]) <= r["n0"] for r in baselines):
        raise ValueError("baseline produced no continuation")


@task(depends_on=[baselines], validate=_has_baselines, timeout_s=1800)
def lossless_control(baselines):
    """kappa=1.0 must reproduce plain greedy token-for-token.

    This is the gate, not a formality: the margin condition replaced the exact
    match test, so if kappa=1.0 diverges then every lossy row below is measuring
    a decoder bug rather than the rule. A failure here stops the sweep.
    """
    bad = []
    for pid, base in zip(_prompts(), baselines):
        for g in GAMMAS:
            r = spec(pid, g, 1.0, "first")
            if r["ids"] != base["ids"][:len(r["ids"])] or r["promoted"]:
                bad.append({"gamma": g, "promoted": r["promoted"],
                            "prefix": _prefix_len(r["ids"][base["n0"]:],
                                                  base["ids"][base["n0"]:])})
    if bad:
        raise AssertionError(f"kappa=1.0 is not lossless: {bad}")
    return {"lossless_verified": True, "gammas": list(GAMMAS)}


@task(depends_on=[baselines, lossless_control], validate=_has_baselines, timeout_s=7200)
def sweep(baselines, lossless_control):
    """kappa x gamma x mode grid. kappa=1.0 is mode-invariant, so run it once."""
    prompts = _prompts()
    rows = []
    configs = [(k, m) for k in KAPPAS for m in MODES if not (k == 1.0 and m == "all")]
    for kappa, mode in configs:
        for g in GAMMAS:
            for pid, base in zip(prompts, baselines):
                r = spec(pid, g, kappa, mode)
                n0 = base["n0"]
                gen = r["ids"][n0:]
                base_gen = base["ids"][n0:]
                mm = sorted(r["mismatch_margins"])
                rows.append({
                    "kappa": kappa, "mode": mode, "gamma": g, "n0": n0,
                    "acceptance": round(r["acceptance"], 4),
                    "promoted": r["promoted"],
                    "n_mismatch": len(mm),
                    "margin_p50": round(mm[len(mm) // 2], 6) if mm else None,
                    "margin_p90": round(mm[int(len(mm) * 0.9)], 6) if mm else None,
                    "margin_max": round(mm[-1], 6) if mm else None,
                    "frac_ge": {str(kv): round(sum(1 for x in mm if x >= kv) / len(mm), 4)
                                for kv in KAPPAS} if mm else None,
                    "mean_promoted_margin": (round(r["mean_promoted_margin"], 4)
                                             if r["mean_promoted_margin"] else None),
                    "speedup_uncached_draft": round(base["seconds"] / r["seconds"], 4),
                    "projected_cached_draft": round(projected(r["acceptance"], g), 4),
                    "greedy_prefix": _prefix_len(gen, base_gen),
                    "n_generated": len(gen),
                    "mean_logprob": round(_mean_logprob(r["ids"], n0), 4),
                    "baseline_mean_logprob": round(base["mean_logprob"], 4),
                    "identical": gen == base_gen,
                })
            print(f"done kappa={kappa} mode={mode} gamma={g}", flush=True)
    return rows


@task(depends_on=[env_check, load, baselines, lossless_control, sweep])
def report(env_check, load, baselines, lossless_control, sweep):
    agg = {}
    for row in sweep:
        key = f"kappa={row['kappa']}|mode={row['mode']}|gamma={row['gamma']}"
        agg.setdefault(key, []).append(row)

    def mean(sel, k):
        vals = [x[k] for x in sel if x[k] is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary = {}
    for key, sel in agg.items():
        summary[key] = {
            "acceptance": mean(sel, "acceptance"),
            "speedup_uncached_draft": mean(sel, "speedup_uncached_draft"),
            "projected_cached_draft": mean(sel, "projected_cached_draft"),
            "promoted_total": sum(x["promoted"] for x in sel),
            "mean_promoted_margin": mean(sel, "mean_promoted_margin"),
            "n_mismatch_total": sum(x["n_mismatch"] for x in sel),
            "margin_p50": mean(sel, "margin_p50"),
            "margin_p90": mean(sel, "margin_p90"),
            "margin_max": mean(sel, "margin_max"),
            "frac_ge": {str(kv): round(sum(x["frac_ge"][str(kv)] for x in sel
                                           if x["frac_ge"]) / max(1, sum(1 for x in sel
                                           if x["frac_ge"])), 4) for kv in KAPPAS},
            "greedy_prefix": mean(sel, "greedy_prefix"),
            "mean_logprob": mean(sel, "mean_logprob"),
            "baseline_mean_logprob": mean(sel, "baseline_mean_logprob"),
            "all_identical": all(x["identical"] for x in sel),
        }
    out = {
        "paper": "arXiv 2609.02897 AdaptiveSpec, margin rule only (no tree shaping)",
        "env": env_check, "model": load, "lossless_control": lossless_control,
        "config": {"max_new": MAX_NEW, "gammas": list(GAMMAS),
                   "kappas": list(KAPPAS), "modes": list(MODES),
                   "c_measured": C_MEASURED, "prompt_style": "chat_in_distribution"},
        "baseline_seconds": [round(b["seconds"], 3) for b in baselines],
        "summary": summary,
        "rows": sweep,
    }
    with open(OUT, "w") as fh:
        json.dump(out, fh, indent=2)
    print(json.dumps(summary, indent=2), flush=True)
    return OUT


if __name__ == "__main__":
    journal = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           ".margin_verify_journal.jsonl")
    Flow(report, journal_path=journal).run()
