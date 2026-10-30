"""onnxruntime CPU reference for the WGSL engine: prefill a chat prompt, then greedy-decode.

  python3 lfm-wgsl/run/ref.py [STEPS]

Writes test/fixture.json (token ids, greedy tokens, top-5 per step) and test/logits.f32 (the
last-position logits of the prefill and of each decode step, 65536 f32 each), which
test/check.mjs compares against the engine.
"""
import json, sys
from pathlib import Path

import jinja2
import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
MD = ROOT / "model"
STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
cfg = json.loads((MD / "config.json").read_text())
tc = json.loads((MD / "tokenizer_config.json").read_text())
tok = Tokenizer.from_file(str(MD / "tokenizer.json"))

messages = [{"role": "system", "content": "You are a voice assistant. Answer in one to three short spoken sentences."},
            {"role": "user", "content": "Why is the sky blue?"}]
env = jinja2.Environment(); env.globals["raise_exception"] = lambda m: (_ for _ in ()).throw(Exception(m))
prompt = env.from_string(tc["chat_template"]).render(messages=messages, add_generation_prompt=True,
                                                      bos_token=tc.get("bos_token", "<|startoftext|>"))
ids = tok.encode(prompt, add_special_tokens=False).ids

sess = ort.InferenceSession(str(MD / "model_q4.onnx"), providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
state = {}
for i in sess.get_inputs():
    if i.name.startswith("past_conv"):
        state[i.name] = np.zeros((1, cfg["conv_dim"], cfg["conv_L_cache"]), np.float32)
    elif i.name.startswith("past_key_values"):
        state[i.name] = np.zeros((1, 8, 0, 64), np.float32)

def run(new_ids, total):
    feed = {"input_ids": np.array([new_ids], np.int64),
            "attention_mask": np.ones((1, total), np.int64),
            **state}
    if "num_logits_to_keep" in names: feed["num_logits_to_keep"] = np.array(1, np.int64)
    out = sess.run(None, feed)
    res = dict(zip([o.name for o in sess.get_outputs()], out))
    for k in list(state):
        state[k] = res[k.replace("past_conv", "present_conv").replace("past_key_values", "present")]
    return res["logits"][0, -1]

logits, greedy, top5 = [], [], []
lg = run(ids, len(ids))
for s in range(STEPS + 1):
    logits.append(lg.astype(np.float32))
    nxt = int(np.argmax(lg)); greedy.append(nxt)
    top5.append([[int(j), float(lg[j])] for j in np.argsort(-lg)[:5]])
    if s == STEPS: break
    lg = run([nxt], len(ids) + s + 1)

(ROOT / "test").mkdir(exist_ok=True)
np.stack(logits).tofile(ROOT / "test" / "logits.f32")
(ROOT / "test" / "fixture.json").write_text(json.dumps({"prompt": prompt, "ids": ids, "greedy": greedy, "top5": top5}, indent=1))
print(len(ids), "prompt tokens; greedy:", repr(tok.decode(greedy)))
