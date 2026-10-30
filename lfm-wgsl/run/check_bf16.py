"""bf16 release (release RoPE layout) against the dequantized q4 ONNX (onnxruntime layout) on one prompt:
they should differ only by 4-bit error. Also checks that the export's q_norm is the release's,
permuted to onnxruntime's RoPE pairing."""
import sys
import numpy as np
import torch
from tokenizers import Tokenizer
sys.path.insert(0, __import__("os").path.dirname(__file__))
from lfm3 import M3, Lfm3, bf16, config, onnx_q4  # noqa: E402

torch.set_num_threads(4)
cfg = config()
tok = Tokenizer.from_file(str(M3 / "tokenizer.json"))
ids = [cfg["bos_token_id"]] + tok.encode(open(__file__).read(), add_special_tokens=False).ids[:199]
Wq, Wb = onnx_q4(), bf16()
a = Wb["model.layers.2.self_attn.q_norm.weight"].float()
perm = torch.cat([torch.arange(0, 32), torch.arange(64, 96), torch.arange(32, 64), torch.arange(96, 128)])
b = Wq["model.layers.2.self_attn.q_norm.weight"]
print("q_norm equal as stored:", torch.allclose(a, b, atol=1e-2), "| equal after permutation:", torch.allclose(a[perm], b, atol=1e-2))
lq = Lfm3(Wq, ort=True).logits(ids).numpy()
lb = Lfm3(Wb, ort=False).logits(ids).numpy()
cos = (lq * lb).sum(-1) / np.linalg.norm(lq, axis=-1) / np.linalg.norm(lb, axis=-1)
print(f"bf16 vs q4: cos min {cos.min():.4f} mean {cos.mean():.4f}, top-1 agree {(lq.argmax(-1) == lb.argmax(-1)).mean():.3f}")
lb2 = Lfm3(Wb, ort=True).logits(ids).numpy()
print(f"bf16 in the wrong (onnxruntime) layout vs q4: top-1 agree {(lq.argmax(-1) == lb2.argmax(-1)).mean():.3f}")
