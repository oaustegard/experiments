"""Shard-based harvest over a mixed in-distribution corpus.

Two corrections over the first harvest. The corpus is now mostly SYNTH, the set
Baguettotron was actually trained on, formatted through the model's own chat
template rather than as raw fields — the model is a reasoning model that expects
<|im_start|>user ... <|im_start|>assistant <think>, and the first harvest fed it
this repository's markdown instead. Wikitext supplies general encyclopedic prose
alongside it.

Shards are written incrementally so training can start on whatever has landed.
"""
import glob, os, sys, time
import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
SEQ = 512
SHARD_TOKENS = 250_000
TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 3_000_000
OUT = "/tmp/specdec/shards"
os.makedirs(OUT, exist_ok=True)

tok = AutoTokenizer.from_pretrained("PleIAs/Baguettotron")
model = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                             dtype=torch.float32).eval()


def synth_docs(limit):
    """SYNTH rows rendered the way the model saw them in training."""
    f = pq.ParquetFile("/tmp/specdec/synth0.parquet")
    n = 0
    for batch in f.iter_batches(batch_size=512):
        d = batch.to_pydict()
        for i in range(len(d["language"])):
            if d["language"][i] != "en":
                continue
            q, r, a = d["query"][i], d["synthetic_reasoning"][i], d["synthetic_answer"][i]
            if not (q and a):
                continue
            think = f"<think>\n{r}</think>\n" if r else ""
            yield (f"<|im_start|>user\n{q}<|im_end|>\n"
                   f"<|im_start|>assistant\n{think}{a}<|im_end|>\n")
            n += 1
            if n >= limit:
                return


def wiki_docs(limit):
    f = pq.ParquetFile("/tmp/specdec/wiki0.parquet")
    buf, n = [], 0
    for batch in f.iter_batches(batch_size=2048):
        for t in batch.column("text").to_pylist():
            if not t or not t.strip():
                continue
            buf.append(t)
            if sum(len(x) for x in buf) > 6000:
                yield "".join(buf); buf = []; n += 1
                if n >= limit:
                    return


def stream_tokens():
    """Interleave 4 SYNTH docs per wikitext doc."""
    s, w = synth_docs(10**9), wiki_docs(10**9)
    while True:
        for _ in range(4):
            try:
                yield tok(next(s), add_special_tokens=False).input_ids
            except StopIteration:
                return
        try:
            yield tok(next(w), add_special_tokens=False).input_ids
        except StopIteration:
            pass


done_shards = len(glob.glob(f"{OUT}/*.npz"))
total = done_shards * SHARD_TOKENS
print(f"resuming at shard {done_shards} ({total} tokens)", flush=True)

buf, H_b, T_b, L_b, shard_tok = [], [], [], [], 0
t0 = time.perf_counter()
gen = stream_tokens()
# Skip what previous runs already consumed.
skipped = 0
while skipped < total:
    try:
        skipped += len(next(gen))
    except StopIteration:
        break

with torch.no_grad():
    while total < TARGET:
        try:
            buf.extend(next(gen))
        except StopIteration:
            print("corpus exhausted", flush=True); break
        while len(buf) >= SEQ:
            chunk, buf = buf[:SEQ], buf[SEQ:]
            out = model(torch.tensor([chunk]), output_hidden_states=True)
            h = out.hidden_states[-1][0]
            am = out.logits[0].argmax(-1)
            H_b.append(h[:-1].to(torch.float16).numpy())
            T_b.append(np.array(chunk[1:], dtype=np.int32))
            L_b.append(am[1:].to(torch.int32).numpy())
            shard_tok += SEQ; total += SEQ
            if shard_tok >= SHARD_TOKENS:
                idx = len(glob.glob(f"{OUT}/*.npz"))
                np.savez(f"{OUT}/shard{idx:03d}.npz",
                         h=np.concatenate(H_b), tok=np.concatenate(T_b),
                         lab=np.concatenate(L_b), seq=SEQ)
                el = time.perf_counter() - t0
                print(f"shard {idx} saved  total={total}  "
                      f"{total/el:.0f} tok/s  {el/60:.1f} min", flush=True)
                H_b, T_b, L_b, shard_tok = [], [], [], 0
print("harvest done", flush=True)
