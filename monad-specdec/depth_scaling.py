"""Does decode latency track layer count or parameter count?

The whole result rests on this: Monad has 1/5.7 the parameters of Baguettotron
but 64/80 of its layers, and is only 2.1x faster. Truncating Baguettotron's own
layer stack varies depth while holding width fixed, which separates the two.
"""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
REPO = "PleIAs/Baguettotron"
tok = AutoTokenizer.from_pretrained(REPO)
model = AutoModelForCausalLM.from_pretrained(REPO, dtype=torch.float32).eval()
full_layers = list(model.model.layers)
ids = tok("The Roman aqueducts were built to carry water into cities and",
          add_special_tokens=False, return_tensors="pt").input_ids

rows = []
for n in [10, 20, 40, 60, 80]:
    model.model.layers = torch.nn.ModuleList(full_layers[:n])
    model.config.num_hidden_layers = n
    params = sum(p.numel() for p in model.parameters())
    with torch.no_grad():
        for rep in range(3):
            out = model(ids, use_cache=True)
            past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
            steps = []
            for _ in range(16):
                s = time.perf_counter()
                out = model(nxt, past_key_values=past, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                steps.append((time.perf_counter() - s) * 1000)
            if rep == 0:
                samples = []
            else:
                samples.extend(steps)
    rows.append({"layers": n, "params_m": round(params / 1e6, 1),
                 "decode_ms": round(statistics.median(samples), 2)})
    print(rows[-1], flush=True)

base = rows[-1]
for r in rows:
    r["latency_frac_of_full"] = round(r["decode_ms"] / base["decode_ms"], 3)
    r["param_frac_of_full"] = round(r["params_m"] / base["params_m"], 3)
    r["layer_frac_of_full"] = round(r["layers"] / base["layers"], 3)

json.dump(rows, open("depth_scaling.json", "w"), indent=2)
print(json.dumps(rows, indent=2))
