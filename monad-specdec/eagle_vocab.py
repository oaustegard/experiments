"""Draft cost against draft-vocabulary size.

78% of an EAGLE draft step on Baguettotron is the 65,536-wide output
projection, because a 576-wide hidden state makes everything else tiny. NVIDIA's
NeMo recipe cuts the draft head to a 32,000-token vocabulary for exactly this
reason: the draft only has to cover the tokens it will actually propose, and
anything it cannot express is simply a rejection the target fixes.
"""
import json, statistics, time, torch
from transformers import AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

torch.set_num_threads(4)
cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
H = cfg.hidden_size
TARGET_STEP = json.load(open("eagle_cost.json"))["target_decode_step_ms"]

layer = LlamaDecoderLayer(cfg, layer_idx=0).eval()
fc = torch.nn.Linear(2 * H, H).eval()
h1, fc_in = torch.randn(1, 1, H), torch.randn(1, 1, 2 * H)
pos = torch.arange(64, 65).unsqueeze(0)
import transformers.models.llama.modeling_llama as ll
rot_src = ll.LlamaRotaryEmbedding(cfg)
rot = rot_src(h1, pos)


def timeit(fn, n=40, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        xs = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t) * 1000)
    return statistics.median(xs)


core = timeit(lambda: fc(fc_in)) + timeit(lambda: layer(h1, position_embeddings=rot,
                                                        position_ids=pos))


def speedup(a, g, c):
    return round((1 - a ** (g + 1)) / ((1 - a) * (1 + g * c)), 2)


rows = []
for V in [65536, 32000, 16384, 8192, 4096]:
    head = torch.nn.Linear(H, V, bias=False).eval()
    ms = timeit(lambda: head(h1))
    step = core + ms
    c = step / TARGET_STEP
    rows.append({
        "draft_vocab": V,
        "lm_head_ms": round(ms, 4),
        "draft_step_ms": round(step, 4),
        "cost_ratio_c": round(c, 4),
        "best_speedup_alpha0.6": max(speedup(0.6, g, c) for g in range(1, 9)),
        "best_speedup_alpha0.7": max(speedup(0.7, g, c) for g in range(1, 9)),
        "best_speedup_alpha0.8": max(speedup(0.8, g, c) for g in range(1, 9)),
    })
    print(rows[-1], flush=True)
    del head

out = {"core_fc_plus_layer_ms": round(core, 4), "target_step_ms": TARGET_STEP, "rows": rows}
json.dump(out, open("eagle_vocab.json", "w"), indent=2)
