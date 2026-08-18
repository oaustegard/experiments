"""What would one EAGLE-style draft step cost on Baguettotron?

EAGLE trains an FC layer (2h -> h) plus one decoder layer, and reuses the
target's embedding and LM head. Baguettotron ties its embedding to the LM head,
so that reuse is free in memory but the 65,536 x 576 output projection still
runs on every draft token. With a 576-wide hidden state that projection is
37.7M params against the draft layer's 3.5M, so it, not the decoder layer,
is what sets the draft cost.

This builds the real modules and times them rather than reading the cost off
the depth-scaling intercept.
"""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

torch.set_num_threads(4)
cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron", dtype=torch.float32).eval()
H, V = cfg.hidden_size, cfg.vocab_size


def timeit(fn, n=40, warmup=10):
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        xs = []
        for _ in range(n):
            t = time.perf_counter()
            fn()
            xs.append((time.perf_counter() - t) * 1000)
    return round(statistics.median(xs), 4)


CTX = 64
h1 = torch.randn(1, 1, H)
pos = torch.arange(CTX, CTX + 1).unsqueeze(0)
rot = target.model.rotary_emb(h1, pos)

layer = LlamaDecoderLayer(cfg, layer_idx=0).eval()
fc = torch.nn.Linear(2 * H, H).eval()
fc_in = torch.randn(1, 1, 2 * H)
lm_head = target.lm_head
embed = target.model.embed_tokens
tok1 = torch.tensor([[7]])

parts = {
    "embed_lookup": timeit(lambda: embed(tok1)),
    "eagle_fc_2h_to_h": timeit(lambda: fc(fc_in)),
    "one_decoder_layer": timeit(lambda: layer(h1, position_embeddings=rot, position_ids=pos)),
    "lm_head_65536": timeit(lambda: lm_head(h1)),
}
parts["eagle_draft_step_total"] = round(
    parts["embed_lookup"] + parts["eagle_fc_2h_to_h"]
    + parts["one_decoder_layer"] + parts["lm_head_65536"], 4)

# Full target decode step, same machine, same moment. Each iteration rebuilds
# the cache, because reusing one across steps lets the context grow and drift.
tok = torch.tensor([[7]])
ctx = torch.randint(0, V, (1, CTX))
xs = []
for _ in range(12):
    with torch.no_grad():
        p = target(ctx, use_cache=True).past_key_values
        t = time.perf_counter()
        target(tok, past_key_values=p, use_cache=True)
        xs.append((time.perf_counter() - t) * 1000)
target_step = round(statistics.median(xs), 4)

params = {
    "eagle_trainable_m": round((sum(p.numel() for p in layer.parameters())
                                + sum(p.numel() for p in fc.parameters())) / 1e6, 2),
    "lm_head_m": round(V * H / 1e6, 2),
    "target_total_m": round(sum(p.numel() for p in target.parameters()) / 1e6, 1),
}


def speedup(a, g, c):
    return round((1 - a ** (g + 1)) / ((1 - a) * (1 + g * c)), 3)


c = parts["eagle_draft_step_total"] / target_step
out = {
    "components_ms": parts,
    "target_decode_step_ms": target_step,
    "cost_ratio_c": round(c, 4),
    "params_m": params,
    "lm_head_share_of_draft_step": round(parts["lm_head_65536"]
                                         / parts["eagle_draft_step_total"], 3),
    "projected_speedup": {
        f"alpha={a},gamma={g}": speedup(a, g, c)
        for a in (0.5, 0.6, 0.7, 0.8) for g in (2, 4, 6)
    },
}
print(json.dumps(out, indent=2))
json.dump(out, open("eagle_cost.json", "w"), indent=2)
