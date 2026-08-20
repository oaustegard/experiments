"""Decode latency for the official Baguettotron GGUF ladder under llama.cpp.

torch's CPU path is not how anyone serves a quantized model, so the question
"what does 4-bit buy" has to be asked of llama.cpp's int4 kernels, not of
torch. Same machine, same thread count, same prompt as the fp32 runs.
"""
import json, os, statistics, sys, time
os.environ.setdefault("LLAMA_LOG_LEVEL", "0")
from llama_cpp import Llama

PROMPT = "The Roman aqueducts were built to carry water into cities and"
N = 48
rows = []
for q in ["Q8_0", "Q4_K_M", "Q4_0"]:
    path = f"/tmp/specdec/Bag-{q}.gguf"
    llm = Llama(model_path=path, n_ctx=512, n_threads=4, verbose=False, logits_all=False)
    runs = []
    for rep in range(3):
        llm.reset()
        t0 = time.perf_counter()
        out = llm(PROMPT, max_tokens=N, temperature=0.0, echo=False)
        dt = time.perf_counter() - t0
        n = out["usage"]["completion_tokens"]
        if rep:
            runs.append(dt / n * 1000)
    rows.append({"quant": q, "file_mb": round(os.path.getsize(path) / 1e6, 1),
                 "ms_per_token": round(statistics.median(runs), 3),
                 "tok_per_s": round(1000 / statistics.median(runs), 1)})
    print(rows[-1], flush=True)
    del llm

f16 = next(r for r in rows if r["quant"] == "Q8_0")["ms_per_token"]
for r in rows:
    r["speedup_vs_q8"] = round(f16 / r["ms_per_token"], 3)
json.dump(rows, open("gguf_latency.json", "w"), indent=2)
print(json.dumps(rows, indent=2))
