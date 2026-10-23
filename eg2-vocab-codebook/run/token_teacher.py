"""EG2 q4 vector of every vocabulary token as a one-token query: ids = the query prompt's ids
with token i in place of the trailing "▁" before <eos> (no detokenize/retokenize round trip). 262,144 x 768, fp16, .work/tok_teacher.npy.
Soft-token placeholder ids get a zero vector. Checkpoints every 32 batches; a rerun resumes."""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import QUERY, WORK, Teacher  # noqa: E402

N, B = 262144, 512
SOFT = {258880, 258881, 258884}  # image/audio/video placeholders: the graph routes them to multimodal features


def main():
    T = Teacher()
    pre = T.tok.encode(QUERY).ids
    print("prompt ids", pre, flush=True)
    out_p, done_p = WORK / "tok_teacher.npy", WORK / "tok_teacher.done"
    out = np.lib.format.open_memmap(out_p, mode="r+" if out_p.exists() else "w+", dtype=np.float16, shape=(N, 768))
    start = int(done_p.read_text()) if done_p.exists() else 0
    empty = np.zeros((0, 512), np.float32)
    t0 = time.time()
    for s in range(start, N, B):
        rng_ids = range(s, min(N, s + B))
        ids = np.array([pre[:-2] + [0 if i in SOFT else i, pre[-1]] for i in rng_ids], np.int64)
        e = T.sess.run(["sentence_embedding"], {"input_ids": ids, "attention_mask": np.ones_like(ids),
                                                "image_features": empty, "video_features": empty, "audio_features": empty})[0]
        e = e / np.linalg.norm(e, axis=1, keepdims=True)
        for j, i in enumerate(rng_ids):
            if i in SOFT:
                e[j] = 0
        out[s:s + len(ids)] = e
        if (s // B) % 32 == 31 or s + B >= N:
            out.flush()
            done_p.write_text(str(s + B))
            print(s + B, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
