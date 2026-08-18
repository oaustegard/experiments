"""Alternative arm: Baguettotron's own first-N layers drafting for the full stack.

Same tokenizer, so no granularity penalty and no detokenize/retokenize round
trip. The truncated stack has no trained early-exit head, so this measures what
layer-skip drafting gives for free, not what it could give after training.
"""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
REPO = "PleIAs/Baguettotron"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
full = list(model.model.layers)

PROMPTS = [
    "The Roman aqueducts were built to carry water into cities. Their construction relied on",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. The mission",
    "The mitochondrion is often described as the powerhouse of the cell because it",
]
N_NEW = 32


def greedy_ids(depth, ids, n):
    model.model.layers = torch.nn.ModuleList(full[:depth])
    out = list(ids)
    with torch.no_grad():
        for _ in range(n):
            lg = model(torch.tensor([out])).logits[0, -1]
            out.append(int(lg.argmax()))
    return out


results = []
for depth in [20, 40, 60]:
    agree = total = 0
    for p in PROMPTS:
        ids = tok(p, add_special_tokens=False).input_ids
        ref = greedy_ids(80, ids, N_NEW)
        # Teacher-forced agreement: at each position on the target's own path,
        # does the truncated stack pick the same next token?
        model.model.layers = torch.nn.ModuleList(full[:depth])
        with torch.no_grad():
            for i in range(len(ids), len(ref)):
                lg = model(torch.tensor([ref[:i]])).logits[0, -1]
                agree += int(int(lg.argmax()) == ref[i])
                total += 1
    # Cost of a truncated Baguettotron stack, from the fit in depth_scaling_monad.py.
    cost_ms = 1.264 * depth + 4.711
    results.append({"draft_depth": depth,
                    "acceptance_teacher_forced": round(agree / total, 3),
                    "cost_ratio_c": round(cost_ms / 105.8, 3)})
    print(results[-1], flush=True)

model.model.layers = torch.nn.ModuleList(full)


def theory(a, g, c):
    return (1 - a ** (g + 1)) / ((1 - a) * (g * c + 1))


for r in results:
    r["best_gamma_speedup"] = max(
        (round(theory(r["acceptance_teacher_forced"], g, r["cost_ratio_c"]), 3), g)
        for g in range(1, 9))
print(json.dumps(results, indent=2))
json.dump(results, open("self_spec.json", "w"), indent=2)
