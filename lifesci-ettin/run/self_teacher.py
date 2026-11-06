"""e17c as its own teacher: encode every training row with e17c (stock table, up to 256 tokens) so a new-vocab
student can be trained to reproduce e17c instead of MedCPT. Isolates the token table from the teacher change.

    python3 -I run/self_teacher.py   # .work/teacher_e17c/{c*.npy, shards/t0..t9.bin, mu_all.npy} -> R2 v1/teacher_e17c/
"""
import json, sys, time
from pathlib import Path
import numpy as np, torch
from tokenizers import Tokenizer
HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE / "run"))
from model import QP, Embedder, from_flat  # noqa: E402
from eval import encode  # noqa: E402
W = HERE / ".work"; O = W / "teacher_e17c"; (O / "shards").mkdir(parents=True, exist_ok=True)
torch.set_num_threads(4)
rows = [json.loads(l) for l in open(W / "data/train.jsonl")]
texts = [(QP if r["role"] == "q" else "") + r["t"] for r in rows]
flat = np.fromfile(W / "params/stock.f32", np.float32)
m = Embedder(from_flat(flat, 50368)).eval()
tok = Tokenizer.from_file(str(W / "e17c/epoch1/tokenizer.json"))
CH = 20000
for c in range(0, len(texts), CH):
    f = O / f"c{c // CH:03d}.npy"
    if f.exists(): continue
    t0 = time.time()
    np.save(f, encode(m, tok, texts[c:c + CH], 256, bs=128).astype(np.float16))
    print(c, round(time.time() - t0, 1), flush=True)
T = np.concatenate([np.load(O / f"c{i:03d}.npy") for i in range((len(texts) + CH - 1) // CH)])
assert T.shape == (len(texts), 768) and np.isfinite(T.astype(np.float32)).all()
mu = T.astype(np.float32).mean(0); np.save(O / "mu_all.npy", mu)
print("mean norm", float(np.linalg.norm(mu)), "avg len", float(np.linalg.norm(T.astype(np.float32)[:20000], axis=1).mean()))
from r2 import put  # noqa: E402
n = len(T) // 10
for i in range(10):
    p = O / f"shards/t{i}.bin"; T[i * n:(i + 1) * n].tofile(p); print(put(p, f"v1/teacher_e17c/t{i}.bin"), flush=True)
