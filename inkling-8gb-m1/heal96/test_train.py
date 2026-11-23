import sys, torch, re, types
sys.path.insert(0, ".")
import heal
heal.patch_transformers()
import transformers.models.inkling2_loop.modeling_inkling2_loop as M
from transformers.models.inkling2_loop.configuration_inkling2_loop import Inkling2LoopTextConfig
torch.manual_seed(0)
cfg = Inkling2LoopTextConfig(hidden_size=64, num_hidden_layers=6, vocab_size=512, unpadded_vocab_size=500,
    num_attention_heads=2, num_key_value_heads=1, head_dim=32, swa_head_dim=32, swa_num_attention_heads=2,
    swa_num_key_value_heads=1, n_routed_experts=8, num_experts_per_tok=2, moe_intermediate_size=32,
    local_layer_ids=[0, 1, 3, 4], n_layers_per_attn_res_block=2, sliding_window_size=16,
    loop_config={"core_start": 2, "core_end": 4, "num_passes": 3, "gate_rank": 8, "is_varloop": False})
model = M.Inkling2LoopForCausalLM(cfg).float()
E, H, I = 8, 64, 32
experts = {}
for name, mod in model.named_modules():
    m = re.search(r"layers\.(\d+)\.mlp\.experts$", name)
    if m: experts[int(m.group(1))] = mod
print("layers with experts", sorted(experts))
keep = {li: [0, 2, 3, 5, 6, 7] for li in experts}
for li, mod in experts.items():
    p13 = {"weight": torch.randint(0, 256, (E, 2 * I, H // 2), dtype=torch.uint8)[keep[li]],
           "scale": (torch.rand(E, 2 * I, H // 16) * 0.05).to(torch.float8_e4m3fn)[keep[li]],
           "amax": torch.rand(E)[keep[li]] + 0.5, "den": torch.tensor([3.0])}
    p2 = {"weight": torch.randint(0, 256, (E, H, I // 2), dtype=torch.uint8)[keep[li]],
          "scale": (torch.rand(E, H, I // 16) * 0.05).to(torch.float8_e4m3fn)[keep[li]],
          "amax": torch.rand(E)[keep[li]] + 0.5, "den": torch.tensor([3.0])}
    mod.set_packed({"w13": p13, "w2": p2}, keep[li])
for p in model.parameters(): p.requires_grad_(False)
heal.STATE.student = True
for li, r in heal.routers(model).items():
    km = torch.zeros(E, dtype=torch.bool); km[torch.tensor(keep[li])] = True
    r.keep_mask = km; r.delta = torch.nn.Parameter(torch.zeros_like(r.weight, dtype=torch.float32))
for m in experts.values(): m.add_lora(4, 8)
model.config.use_cache = False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.train()
lora, rd = heal.trainable(model)
print("trainable", len(lora), len(rd), sum(p.numel() for p in lora + rd))
opt = torch.optim.AdamW(lora + rd, lr=1e-2)
ids = torch.randint(0, 500, (2, 32))
# teacher targets: random top-k from a fixed random distribution
heal.TOPK = 8
with torch.no_grad():
    tl = torch.randn(2 * 31, 500).log_softmax(-1)
    lp_t, idx_t = tl.topk(8, -1)
for step in range(30):
    opt.zero_grad()
    h = model.model(input_ids=ids).last_hidden_state[:, :-1].reshape(-1, 64)
    tot = 0
    for c0 in range(0, h.shape[0], 16):
        tot = tot + torch.utils.checkpoint.checkpoint(heal.kd_loss_chunk, model, h[c0:c0+16], idx_t[c0:c0+16], lp_t[c0:c0+16], use_reentrant=False)
    loss = tot / h.shape[0]
    loss.backward()
    if step == 0:
        print("grad norms: lora", sum(float(p.grad.norm()) for p in lora if p.grad is not None), "router", sum(float(p.grad.norm()) for p in rd if p.grad is not None))
    opt.step()
    if step % 10 == 0 or step == 29: print(step, float(loss))
# masked experts never routed
seen = set()
def hook(mod, inp):
    seen.update(inp[1].unique().tolist())
for m in experts.values(): m.register_forward_pre_hook(hook)
with torch.no_grad(): model.model(input_ids=ids)
print("experts used", sorted(seen), "dropped ok:", not ({1, 4} & seen))
sd = heal.adapter_state(model); print("adapter keys", len(sd), list(sd)[:2], [k for k in sd if 'delta' in k][:1])
