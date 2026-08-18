"""Monad-drafts-Baguettotron vs plain greedy Baguettotron.

Baseline and speculative runs are interleaved per (prompt, gamma) pair because
the container's 4 shared cores drift by 20-30% over a long run; comparing a
speculative number against a baseline measured ten minutes earlier measures the
machine, not the method.
"""
import json, torch
from specdec import (CachedRunner, DRAFT_REPO, TARGET_REPO,
                     speculative_generate, baseline_generate)

torch.set_num_threads(4)

PROMPTS = [
    "The Roman aqueducts were built to carry water into cities. Their construction relied on",
    "Question: Why does bread rise when yeast is added?\nAnswer:",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. The mission",
    "def fibonacci(n):\n    \"\"\"Return the nth Fibonacci number.\"\"\"\n",
    "The mitochondrion is often described as the powerhouse of the cell because it",
]
MAX_NEW = 48
GAMMAS = [1, 2, 3, 4, 8]

drafter = CachedRunner(DRAFT_REPO)
target = CachedRunner(TARGET_REPO)
results = {"max_new_tokens": MAX_NEW, "threads": torch.get_num_threads(),
           "prompts": PROMPTS, "speculative": {}}

for g in GAMMAS:
    runs = []
    for p in PROMPTS:
        target.cache, target.ids = None, []
        base = baseline_generate(target, p, MAX_NEW)

        drafter.cache, drafter.ids = None, []
        target.cache, target.ids = None, []
        drafter.forward_tokens = target.forward_tokens = 0
        spec = speculative_generate(drafter, target, p, MAX_NEW, gamma=g)

        r = {k: v for k, v in spec.items() if k not in ("text", "ids")}
        r["prompt"] = p[:50]
        r["identical_to_greedy"] = spec["ids"] == base["ids"]
        r["base_ms_per_token"] = round(base["seconds"] / base["new_tokens"] * 1000, 2)
        r["spec_ms_per_token"] = round(spec["seconds"] / spec["new_tokens"] * 1000, 2)
        r["speedup"] = round(r["base_ms_per_token"] / r["spec_ms_per_token"], 3)
        r["mean_accepted_per_round"] = round(spec["accepted"] / spec["rounds"], 2)
        print(f"g={g} accept={r['acceptance_rate']:.3f} "
              f"acc/round={r['mean_accepted_per_round']:.2f} "
              f"speedup={r['speedup']:.3f} identical={r['identical_to_greedy']}", flush=True)
        runs.append(r)
    n = len(runs)
    results["speculative"][str(g)] = {
        "runs": runs,
        "mean_acceptance": round(sum(r["acceptance_rate"] for r in runs) / n, 3),
        "mean_accepted_per_round": round(sum(r["mean_accepted_per_round"] for r in runs) / n, 2),
        "mean_speedup": round(sum(r["speedup"] for r in runs) / n, 3),
        "all_identical": all(r["identical_to_greedy"] for r in runs),
    }

json.dump(results, open("results.json", "w"), indent=2)
print(json.dumps({g: {k: v for k, v in d.items() if k != "runs"}
                  for g, d in results["speculative"].items()}, indent=2))
