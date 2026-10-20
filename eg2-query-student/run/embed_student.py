"""Off-the-shelf small-model embeddings for query files (the A1 inputs).

Usage: python3 -I run/embed_student.py MODEL out/queries/X.jsonl [more.jsonl ...]

MODEL is a directory name under out/models/ (sentence-transformers layout). Uses the model's
own "query" prompt when it declares one. Writes out/student/MODEL/X.npy (unnormalized).
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent


def main():
    torch.set_num_threads(4)
    name, files = sys.argv[1], sys.argv[2:]
    m = SentenceTransformer(str(ROOT / "out" / "models" / name), device="cpu")
    prompt = "query" if "query" in (m.prompts or {}) else None
    dst = ROOT / "out" / "student" / name
    dst.mkdir(parents=True, exist_ok=True)
    for p in map(Path, files):
        out = dst / (p.stem + ".npy")
        if out.exists():
            continue
        qs = [json.loads(l)["q"] for l in p.open()]
        E = m.encode(qs, batch_size=128, prompt_name=prompt, normalize_embeddings=False,
                     convert_to_numpy=True, show_progress_bar=False)
        np.save(out, E.astype(np.float32))
        print(name, p.stem, E.shape, flush=True)


if __name__ == "__main__":
    main()
