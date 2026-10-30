"""Check run/lfm3.py against onnxruntime on onnx-community's LFM3-1.7B model_q4.onnx, with lfm3.py
running the same q4 weights dequantized. A prompt longer than the 256-position sliding window
exercises the window. Prints logit cosine and top-1 agreement per position; exits 1 below 0.999.

  python3 lfm-wgsl/run/parity3.py [N_TOKENS]
"""
import sys
import time

import numpy as np
import onnxruntime as ort
import torch
from tokenizers import Tokenizer

sys.path.insert(0, __import__("os").path.dirname(__file__))
from lfm3 import M3, Lfm3, config, onnx_q4  # noqa: E402

N = int(sys.argv[1]) if len(sys.argv) > 1 else 320
cfg = config()
tok = Tokenizer.from_file(str(M3 / "tokenizer.json"))
text = (open(__file__).read() + "\n" + open(M3.parent / "README.md").read())
ids = [cfg["bos_token_id"]] + tok.encode(text, add_special_tokens=False).ids[:N - 1]
torch.set_num_threads(4)

t0 = time.time()
model = Lfm3(onnx_q4(), ort=True)
ours = model.logits(ids).numpy()
print(f"torch forward on dequantized q4: {time.time() - t0:.0f} s")

# The export sets accuracy_level 4 on every MatMulNBits, which makes onnxruntime quantize activations
# to int8 (cosine 0.9975, top-1 0.984 against exact math). A copy with accuracy_level 0 isolates the
# forward pass; it reads the same external data file.
acc0 = M3 / "model_q4_acc0.onnx"
if not acc0.exists():
    import onnx
    g = onnx.load(str(M3 / "model_q4.onnx"), load_external_data=False)
    for n in g.graph.node:
        for a in n.attribute:
            if a.name == "accuracy_level":
                a.i = 0
    onnx.save(g, str(acc0))
so = ort.SessionOptions(); so.intra_op_num_threads = 4
sess = ort.InferenceSession(str(acc0), so, providers=["CPUExecutionProvider"])
feed = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, len(ids)), np.int64),
        "num_logits_to_keep": np.array(len(ids), np.int64)}
for i in sess.get_inputs():
    if i.name.startswith("past_conv"):
        feed[i.name] = np.zeros((1, cfg["conv_L_cache"] - 1, cfg["hidden_size"]), np.float32)
    elif i.name.startswith("past_key_values"):
        feed[i.name] = np.zeros((1, cfg["num_key_value_heads"], 0, cfg["head_dim"]), np.float32)
t0 = time.time()
ref = sess.run(["logits"], feed)[0][0]
print(f"onnxruntime: {time.time() - t0:.0f} s")

cos = (ours * ref).sum(-1) / np.linalg.norm(ours, axis=-1) / np.linalg.norm(ref, axis=-1)
agree = ours.argmax(-1) == ref.argmax(-1)
for lo, hi in ((0, 256), (256, len(ids))):
    if hi > lo:
        print(f"positions {lo}-{hi - 1}: cos min {cos[lo:hi].min():.6f} mean {cos[lo:hi].mean():.6f}, "
              f"top-1 agree {agree[lo:hi].mean():.3f}, max |diff| {np.abs(ours[lo:hi] - ref[lo:hi]).max():.4f}")
sys.exit(0 if cos.min() > 0.999 else 1)
