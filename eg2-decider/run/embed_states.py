"""Embed every Typed Decisions state (train+test) and every unrelated-set input, under each state prompt.

    python3 -I run/embed_states.py [--prompts doc,cls]

Resumable: the encoder cache skips what is already embedded.
"""
import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "browser-decider" / "run"))
from common import WORKFLOWS, load_cases  # noqa: E402
from enc import EG2, encoder, render, results_dir  # noqa: E402

OOD = HERE.parents[1] / "browser-decider" / "data" / "decider_eval.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="doc,cls")
    a = ap.parse_args()
    m = encoder()
    texts = [r["input"] for r in map(json.loads, open(OOD))]
    for w in WORKFLOWS:
        for split in ("test", "train"):
            texts += [render(c["state"]) for c in load_cases(w, split)]
    print(f"{len(texts)} states", flush=True)
    for p in a.prompts.split(","):
        t0 = time.time()
        m.embed(texts, prompt=p, log_every=200)
        print(f"prompt {p}: done in {time.time() - t0:.0f}s", flush=True)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
