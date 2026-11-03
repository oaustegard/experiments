"""Random-weight Llama shapes for a latency-only bench: Monad's width (256), layers of 8 heads and a
1,536-wide MLP (a parallel pair folded into one layer), at a chosen depth. Weights are bf16 random."""
import sys, json, shutil, torch
from transformers import LlamaConfig, LlamaForCausalLM
from common import HERE, MODEL_DIR
depth = int(sys.argv[1]); narrow = len(sys.argv) > 2 and sys.argv[2] == "narrow"
out = HERE / f"models/{'narrow' if narrow else 'wide'}{depth}"
cfg = json.loads((MODEL_DIR / "config.json").read_text())
cfg.update(num_hidden_layers=depth, num_attention_heads=4 if narrow else 8, num_key_value_heads=4 if narrow else 8, head_dim=64,
           intermediate_size=768 if narrow else 1536)
torch.manual_seed(0)
m = LlamaForCausalLM(LlamaConfig(**cfg)).to(torch.bfloat16)
shutil.rmtree(out, ignore_errors=True); m.save_pretrained(out)
for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "generation_config.json", "chat_template.json"):
    shutil.copy(MODEL_DIR / f, out / f)
print(out)
