"""Encode the AST corpus with the code-trained encoder, checkpointing as it goes."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from jinacode import JinaCodeEncoder

HERE = Path(__file__).resolve().parents[1]
DIM = 768

def main() -> None:
    chunks = json.load(open(HERE / "chunks_ast.json"))
    texts = [c["text"] for c in chunks]
    n = len(texts)
    out = HERE / "vecs_ast_jinacode.f32"
    done_p = HERE / "vecs_ast_jinacode.done"
    mm = np.memmap(out, dtype=np.float32, mode="r+" if out.exists() else "w+", shape=(n, DIM))
    start = int(done_p.read_text()) if done_p.exists() else 0
    if start >= n:
        print("already complete", flush=True); return
    # length-sort the WHOLE corpus once, then process in order, so padding waste
    # is minimal while checkpoints stay contiguous in the sorted domain.
    order = np.argsort([len(t) for t in texts], kind="stable")
    enc = JinaCodeEncoder(threads=4)
    t0 = time.time(); CK = 256
    for s in range(start, n, CK):
        idx = order[s:s+CK]
        v = enc.encode([texts[i] for i in idx], batch_size=16, sort_by_length=False)
        mm[idx] = v
        mm.flush(); done_p.write_text(str(s + len(idx)))
        el = time.time() - t0; rate = (s + len(idx) - start) / max(el, 1e-9)
        print(f"jina-code: {s+len(idx)}/{n} {rate:.1f} ch/s eta {(n-s-len(idx))/max(rate,1e-9)/60:.1f}m", flush=True)
    print(f"DONE {n} in {(time.time()-t0)/60:.1f}m", flush=True)

if __name__ == "__main__":
    main()
