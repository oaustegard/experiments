"""Per-token decode latency for Monad (56M) vs Baguettotron (321M), CPU, batch 1.

Speculative decoding pays only when the draft model's forward pass is a small
fraction of the target's. Both models are deep-and-narrow, so the parameter
ratio (5.7x) is not a safe proxy for the latency ratio.
"""
import json, time, statistics, sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
MODELS = {"monad": "PleIAs/Monad", "baguettotron": "PleIAs/Baguettotron"}
PROMPT = ("The Roman aqueducts were built to carry water into cities. "
          "Their construction relied on a steady gradient, and")


def bench(name, repo, n_decode=32, n_repeat=3):
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32)
    model.eval()
    ids = tok(PROMPT, return_tensors="pt").input_ids

    prefill_ms, decode_ms = [], []
    with torch.no_grad():
        for r in range(n_repeat + 1):
            t0 = time.perf_counter()
            out = model(ids, use_cache=True)
            past = out.past_key_values
            t1 = time.perf_counter()
            nxt = out.logits[:, -1:].argmax(-1)
            steps = []
            for _ in range(n_decode):
                s = time.perf_counter()
                out = model(nxt, past_key_values=past, use_cache=True)
                past = out.past_key_values
                nxt = out.logits[:, -1:].argmax(-1)
                steps.append((time.perf_counter() - s) * 1000)
            if r == 0:
                continue  # warmup
            prefill_ms.append((t1 - t0) * 1000)
            decode_ms.extend(steps)

    n_params = sum(p.numel() for p in model.parameters())
    res = {
        "model": name, "repo": repo, "params_m": round(n_params / 1e6, 1),
        "layers": model.config.num_hidden_layers,
        "hidden": model.config.hidden_size,
        "vocab": model.config.vocab_size,
        "prompt_tokens": int(ids.shape[1]),
        "prefill_ms_median": round(statistics.median(prefill_ms), 2),
        "decode_ms_median": round(statistics.median(decode_ms), 3),
        "decode_ms_mean": round(statistics.mean(decode_ms), 3),
        "decode_ms_p10": round(sorted(decode_ms)[len(decode_ms) // 10], 3),
        "n_decode_samples": len(decode_ms),
    }
    del model
    return res


if __name__ == "__main__":
    out = [bench(n, r) for n, r in MODELS.items()]
    ratio = out[1]["decode_ms_median"] / out[0]["decode_ms_median"]
    summary = {"runs": out, "target_over_draft_latency_ratio": round(ratio, 2),
               "torch_threads": torch.get_num_threads()}
    print(json.dumps(summary, indent=2))
    with open("latency.json", "w") as f:
        json.dump(summary, f, indent=2)
