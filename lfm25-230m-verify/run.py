"""Actually run LFM2.5-230M to verify Sung Kim's claims on real hardware."""
import time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "LiquidAI/LFM2.5-230M"
torch.set_num_threads(4)

t0 = time.time()
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
load_s = time.time() - t0

# Report ground-truth param count + architecture from the loaded weights
n_params = sum(p.numel() for p in model.parameters())
print(f"=== GROUND TRUTH FROM LOADED MODEL ===")
print(f"architecture class : {model.__class__.__name__}")
print(f"config model_type  : {model.config.model_type}")
print(f"param count        : {n_params:,} ({n_params/1e6:.1f}M)")
print(f"max_position_emb   : {getattr(model.config, 'max_position_embeddings', 'n/a')}")
print(f"load time          : {load_s:.1f}s")
print(f"dtype              : {next(model.parameters()).dtype}")

# Memory footprint
mem_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"weights in RAM     : {mem_bytes/1e6:.0f} MB (fp32)")

# Real generation + tok/s timing on CPU
prompt = "Explain what a hash map is in two sentences."
msgs = [{"role": "user", "content": prompt}]
enc = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
in_len = enc["input_ids"].shape[1]

# warmup
_ = model.generate(**enc, max_new_tokens=8, do_sample=False)

N = 100
t0 = time.time()
out = model.generate(**enc, max_new_tokens=N, do_sample=False)
dt = time.time() - t0
gen_tokens = out.shape[1] - in_len
text = tok.decode(out[0][in_len:], skip_special_tokens=True)

print(f"\n=== CPU INFERENCE ({torch.get_num_threads()} threads) ===")
print(f"generated {gen_tokens} tokens in {dt:.2f}s = {gen_tokens/dt:.1f} tok/s")
print(f"\n=== MODEL OUTPUT ===\n{text}")
