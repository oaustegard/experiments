"""Locate the first token where speculative output leaves the greedy path."""
import json, torch
from specdec import CachedRunner, speculative_generate, baseline_generate

torch.set_num_threads(4)
drafter = CachedRunner("PleIAs/Monad")
target = CachedRunner("PleIAs/Baguettotron")
tt = target.tok

PROMPTS = [
    "The Roman aqueducts were built to carry water into cities. Their construction relied on",
    "In 1969, the Apollo 11 mission landed the first humans on the Moon. The mission",
    "The mitochondrion is often described as the powerhouse of the cell because it",
]
report = []
for p in PROMPTS:
    target.cache, target.ids = None, []
    base = baseline_generate(target, p, 48)
    base_ids = tt(base["text"], add_special_tokens=False).input_ids

    drafter.cache, drafter.ids = None, []
    target.cache, target.ids = None, []
    spec = speculative_generate(drafter, target, p, 48, gamma=4)
    spec_ids = tt(spec["text"], add_special_tokens=False).input_ids

    d = next((i for i, (x, y) in enumerate(zip(base_ids, spec_ids)) if x != y), None)
    entry = {"prompt": p[:45], "identical": base["text"] == spec["text"],
             "first_diff_tok": d}
    if d is not None:
        target.cache, target.ids = None, []
        lg, off = target.logits_for(base_ids[:d], need_from=0)
        row = lg[-1]
        top = torch.topk(row, 3)
        entry["gap_top1_top2"] = round(float(top.values[0] - top.values[1]), 6)
        entry["top3_ids"] = [int(i) for i in top.indices]
        entry["base_tok"] = int(base_ids[d])
        entry["spec_tok"] = int(spec_ids[d])
        entry["base_str"] = tt.decode([base_ids[d]])
        entry["spec_str"] = tt.decode([spec_ids[d]])
        entry["context_tail"] = tt.decode(base_ids[max(0, d - 12):d])
    report.append(entry)
    print(json.dumps(entry, indent=2), flush=True)

json.dump(report, open("divergence.json", "w"), indent=2)
