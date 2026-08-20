"""How much of per-layer decode time is weight-bandwidth?

Halving the weight bytes (fp32 -> bf16) leaves FLOPs and per-layer overhead
unchanged, so the speedup isolates the bandwidth-proportional share. That share
is also the ceiling on what 4-bit quantization can buy on this CPU.
"""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
out = {}
for name, repo in [("monad", "PleIAs/Monad"), ("baguettotron", "PleIAs/Baguettotron")]:
    tok = AutoTokenizer.from_pretrained(repo)
    row = {}
    for dt in [torch.float32, torch.bfloat16]:
        model = AutoModelForCausalLM.from_pretrained(repo, dtype=dt).eval()
        ids = tok("The Roman aqueducts were built to carry water into cities and",
                  add_special_tokens=False, return_tensors="pt").input_ids
        bytes_per_param = 4 if dt == torch.float32 else 2
        samples = []
        with torch.no_grad():
            for rep in range(3):
                o = model(ids, use_cache=True)
                past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                steps = []
                for _ in range(16):
                    s = time.perf_counter()
                    o = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = o.past_key_values, o.logits[:, -1:].argmax(-1)
                    steps.append((time.perf_counter() - s) * 1000)
                if rep:
                    samples.extend(steps)
        row[str(dt).replace("torch.", "")] = {
            "decode_ms": round(statistics.median(samples), 3),
            "weight_mb": round(sum(p.numel() for p in model.parameters())
                               * bytes_per_param / 1e6, 1),
        }
        print(name, dt, row[str(dt).replace("torch.", "")], flush=True)
        del model
    f32, bf16 = row["float32"]["decode_ms"], row["bfloat16"]["decode_ms"]
    # t = fixed + bandwidth; halving bytes halves only the bandwidth term:
    #   bf16 = fixed + bw/2,  f32 = fixed + bw  ->  bw = 2*(f32 - bf16)
    bw = 2 * (f32 - bf16)
    row["speedup_bf16_over_fp32"] = round(f32 / bf16, 3)
    row["bandwidth_share_of_fp32_time"] = round(max(0.0, bw) / f32, 3)
    row["fixed_ms_implied"] = round(f32 - max(0.0, bw), 3)
    out[name] = row

print(json.dumps(out, indent=2))
json.dump(out, open("dtype_scaling.json", "w"), indent=2)
