"""Teacher-forced agreement of MLX checkpoints with the transformers transcripts.

One forward pass per image over prompt + reference answer; the LM head runs
only at answer positions. Reports top-1 agreement with the reference tokens
and the mean log-probability the checkpoint gives them. Much cheaper than
parity.py's greedy decode on CPU, where quantized matmuls have no fast path.

    python sweep.py REF_DIR MLX_DIR [MLX_DIR ...]
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.utils import load
from PIL import Image


def evaluate(model_dir: str, ref_dir: Path) -> dict:
    ref = json.loads((ref_dir / "reference.json").read_text())
    model, processor = load(model_dir)
    lm = model.language_model
    rows = {}
    for name, r in ref.items():
        image = Image.open(ref_dir / f"{name}.png").convert("RGB")
        inputs = processor(text=[r["prompt"]], images=[image])
        prompt_ids = np.asarray(inputs["input_ids"]).reshape(-1)
        answer = np.load(ref_dir / f"{name}_tokens.npy")
        ids = mx.array(np.concatenate([prompt_ids, answer])[None])
        start = time.perf_counter()
        embeds = model.get_input_embeddings(
            ids, inputs["pixel_values"], tgt_sizes=inputs["tgt_sizes"], image_bound=inputs["image_bound"]
        ).inputs_embeds
        hidden = lm(ids, inputs_embeds=embeds, skip_logits=True, return_hidden=True).hidden_states[-1]
        # position i predicts token i + 1: the answer is predicted from the last prompt token onward
        hidden = hidden[0, len(prompt_ids) - 1 : -1]
        logits = model.language_model.model.embed_tokens.as_linear(hidden).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        target = mx.array(answer)
        top1 = mx.argmax(logits, axis=-1) == target
        chosen = mx.take_along_axis(logprobs, target[:, None], axis=-1)[:, 0]
        mx.eval(top1, chosen)
        rows[name] = {
            "tokens": int(answer.size),
            "top1_agreement": round(float(top1.mean()), 4),
            "disagreements": int((~np.array(top1)).sum()),
            "mean_logprob": round(float(chosen.mean()), 4),
            "min_logprob": round(float(chosen.min()), 3),
            "seconds": round(time.perf_counter() - start, 1),
        }
        print(Path(model_dir).name, name, json.dumps(rows[name]), flush=True)
    return rows


def main() -> None:
    ref_dir = Path(sys.argv[1])
    for model_dir in sys.argv[2:]:
        rows = evaluate(model_dir, ref_dir)
        (Path(model_dir) / "sweep.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
