"""Throughput for harvesting EAGLE training data on this container.

Training an EAGLE head needs the target's hidden states over a training corpus.
That is a batched forward pass, not autoregressive decoding, so the relevant
number is prefill throughput at a useful batch size, not the 9 tok/s decode
rate.
"""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoConfig

torch.set_num_threads(4)
model = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron", dtype=torch.float32).eval()
V = model.config.vocab_size

rows = []
for bs, sl in [(1, 512), (4, 512), (8, 512), (8, 1024)]:
    ids = torch.randint(0, V, (bs, sl))
    with torch.no_grad():
        model(ids[:1, :64], output_hidden_states=True)  # warm
        xs = []
        for _ in range(3):
            t = time.perf_counter()
            model(ids, output_hidden_states=True)
            xs.append(time.perf_counter() - t)
    sec = statistics.median(xs)
    tps = bs * sl / sec
    rows.append({"batch": bs, "seqlen": sl, "seconds": round(sec, 2),
                 "tokens_per_s": round(tps, 1),
                 "hours_per_100M_tokens": round(100e6 / tps / 3600, 1)})
    print(rows[-1], flush=True)

best = max(r["tokens_per_s"] for r in rows)
out = {"rows": rows, "best_tokens_per_s": best,
       "hours_for_eagle_sharegpt_68k_approx_50M_tokens": round(50e6 / best / 3600, 1),
       "note": "Forward-pass harvest only; excludes the draft-head optimizer steps, "
               "which are negligible at 4-14M params."}
print(json.dumps(out, indent=2))
json.dump(out, open("train_feasibility.json", "w"), indent=2)
