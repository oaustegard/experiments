"""Does the EAGLE cost ratio c survive quantization?

The earlier c = 0.059 was measured in torch fp32 for both the target and a
randomly-initialized draft module. Quantizing the target speeds up its 80
layers, but an EAGLE draft step is dominated by the vocabulary projection, so c
only stays put if both shrink together. This quantizes target and draft in the
same runtime at the same precision, which is the fair comparison.

Separately: llama.cpp does NOT quantize them together. In the official GGUFs
`token_embd.weight` — which is the LM head, since Baguettotron ties embeddings —
stays Q8_0 in Q4_0 and Q4_K_M alike, while every layer drops to 4-bit.
"""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

torch.set_num_threads(4)
cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
H, V, CTX = cfg.hidden_size, cfg.vocab_size, 64


def timeit(fn, n=30, warmup=8):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        xs = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs)


def q8(mod):
    return torch.ao.quantization.quantize_dynamic(mod, {torch.nn.Linear},
                                                  dtype=torch.qint8)


results = {}
for label, quantize in [("fp32", False), ("int8_dynamic", True)]:
    target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                                  dtype=torch.float32).eval()
    layer = LlamaDecoderLayer(cfg, layer_idx=0).eval()
    fc = torch.nn.Linear(2 * H, H).eval()
    head = torch.nn.Linear(H, V, bias=False).eval()
    head.weight.data = target.lm_head.weight.data.clone()
    if quantize:
        target = q8(target)
        layer, fc, head = q8(layer), q8(fc), q8(head)

    h1, fc_in = torch.randn(1, 1, H), torch.randn(1, 1, 2 * H)
    pos = torch.arange(CTX, CTX + 1).unsqueeze(0)
    import transformers.models.llama.modeling_llama as ll
    rot = ll.LlamaRotaryEmbedding(cfg)(h1, pos)

    parts = {
        "fc": timeit(lambda: fc(fc_in)),
        "decoder_layer": timeit(lambda: layer(h1, position_embeddings=rot,
                                              position_ids=pos)),
        "lm_head_65536": timeit(lambda: head(h1)),
    }
    draft = sum(parts.values())

    tok1, ctx = torch.tensor([[7]]), torch.randint(0, V, (1, CTX))
    xs = []
    for _ in range(10):
        with torch.no_grad():
            p = target(ctx, use_cache=True).past_key_values
            t = time.perf_counter()
            target(tok1, past_key_values=p, use_cache=True)
            xs.append((time.perf_counter() - t) * 1000)
    tgt = statistics.median(xs)

    results[label] = {
        "components_ms": {k: round(v, 4) for k, v in parts.items()},
        "draft_step_ms": round(draft, 4),
        "target_step_ms": round(tgt, 4),
        "cost_ratio_c": round(draft / tgt, 4),
        "lm_head_share_of_draft": round(parts["lm_head_65536"] / draft, 3),
    }
    print(label, json.dumps(results[label]), flush=True)
    del target, layer, fc, head


def speedup(a, g, c):
    return round((1 - a ** (g + 1)) / ((1 - a) * (1 + g * c)), 2)


for label, r in results.items():
    r["best_speedup_alpha0.7"] = max(speedup(0.7, g, r["cost_ratio_c"])
                                     for g in range(1, 9))
results["c_ratio_int8_over_fp32"] = round(
    results["int8_dynamic"]["cost_ratio_c"] / results["fp32"]["cost_ratio_c"], 3)
print(json.dumps(results, indent=2))
json.dump(results, open("eagle_quant.json", "w"), indent=2)
