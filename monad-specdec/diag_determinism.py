"""Is the non-identical output a logic bug or float non-determinism?

Greedy decoding one token at a time and greedy decoding the same prefix in a
single batched forward pass use different GEMM shapes, so the logits differ in
the last bits. Where two candidates are near-tied, argmax flips. This checks
whether the target model alone diverges from itself under the two schedules.
"""
import json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
model = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron", dtype=torch.float32).eval()

PROMPT = "The Roman aqueducts were built to carry water into cities. Their construction relied on"
N = 48
ids = tok(PROMPT, add_special_tokens=False).input_ids

# Schedule A: incremental, one token per forward, KV cache.
with torch.no_grad():
    out = model(torch.tensor([ids]), use_cache=True)
    past = out.past_key_values
    a = list(ids)
    nxt = int(out.logits[0, -1].argmax())
    gaps = []
    for _ in range(N):
        a.append(nxt)
        out = model(torch.tensor([[nxt]]), past_key_values=past, use_cache=True)
        past = out.past_key_values
        lg = out.logits[0, -1]
        top2 = torch.topk(lg, 2).values
        gaps.append(float(top2[0] - top2[1]))
        nxt = int(lg.argmax())

# Schedule B: re-run the whole sequence in one batched pass, take argmax at each step.
with torch.no_grad():
    b = list(ids)
    for _ in range(N):
        lg = model(torch.tensor([b])).logits[0, -1]
        b.append(int(lg.argmax()))

first_diff = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)
res = {
    "identical": a == b,
    "first_divergence_index": first_diff,
    "tokens_before_divergence": None if first_diff is None else first_diff - len(ids),
    "median_top1_top2_gap": round(sorted(gaps)[len(gaps) // 2], 4),
    "n_gaps_below_0.01": sum(1 for g in gaps if g < 0.01),
    "n_gaps_below_0.1": sum(1 for g in gaps if g < 0.1),
    "n_steps": N,
}
print(json.dumps(res, indent=2))
json.dump(res, open("determinism.json", "w"), indent=2)
