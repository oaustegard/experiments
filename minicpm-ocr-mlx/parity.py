"""Check an MLX checkpoint of MiniCPM-OCR against the transformers reference.

For each image saved by reference.py: the processor's token ids and slice
sizes must equal transformers', then the first-step logits are compared
(cosine, max abs diff, top-1) and greedy decoding is compared token by token.

    python parity.py MLX_MODEL_DIR REF_DIR
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_vlm.generate import generate
from mlx_vlm.utils import load
from PIL import Image


def main(model_dir: str, ref_dir: str) -> int:
    ref_dir = Path(ref_dir)
    ref = json.loads((ref_dir / "reference.json").read_text())
    model, processor = load(model_dir)
    failures = 0
    report = {}
    for name, r in ref.items():
        image = Image.open(ref_dir / f"{name}.png").convert("RGB")
        inputs = processor(text=[r["prompt"]], images=[image])
        ref_inputs = np.load(ref_dir / f"{name}_inputs.npz")
        ids = np.asarray(inputs["input_ids"]).reshape(-1)
        same_ids = ids.tolist() == ref_inputs["input_ids"][0].tolist()
        same_sizes = np.asarray(inputs["tgt_sizes"][0]).tolist() == ref_inputs["target_sizes"].tolist()

        kwargs = {k: inputs[k] for k in ("pixel_values", "tgt_sizes", "image_bound")}
        logits = model(mx.array(ids[None]), **kwargs).logits[0, -1].astype(mx.float32)
        logits = np.array(logits)
        ref_logits = np.load(ref_dir / f"{name}_logits.npy")
        cos = float(logits @ ref_logits / np.linalg.norm(logits) / np.linalg.norm(ref_logits))

        start = time.perf_counter()
        result = generate(
            model, processor, r["prompt"], image=[image], max_tokens=r["n_tokens"] + 8, temperature=0.0, verbose=False
        )
        seconds = time.perf_counter() - start
        ref_tokens = np.load(ref_dir / f"{name}_tokens.npy").tolist()
        mlx_tokens = processor.tokenizer.encode(result.text, add_special_tokens=False)
        ref_text_tokens = processor.tokenizer.encode(r["text"], add_special_tokens=False)
        agree = next((i for i, (a, b) in enumerate(zip(mlx_tokens, ref_text_tokens)) if a != b), None)
        report[name] = {
            "ids_equal": same_ids,
            "slice_sizes_equal": same_sizes,
            "logit_cosine": round(cos, 6),
            "logit_max_abs_diff": round(float(np.abs(logits - ref_logits).max()), 4),
            "top1_equal": int(logits.argmax()) == int(ref_logits.argmax()),
            "text_equal": result.text.strip() == r["text"].strip(),
            "first_divergent_token": agree,
            "ref_tokens": len(ref_tokens),
            "generation_tps": round(result.generation_tps, 1),
            "seconds": round(seconds, 1),
            "text": result.text,
        }
        ok = same_ids and same_sizes and report[name]["top1_equal"] and report[name]["text_equal"]
        failures += not ok
        print(name, "OK" if ok else "MISMATCH", json.dumps({k: v for k, v in report[name].items() if k != "text"}))
        if not report[name]["text_equal"]:
            print("  ref:", repr(r["text"]))
            print("  mlx:", repr(result.text))
    (Path(model_dir) / "parity.json").write_text(json.dumps(report, indent=1, ensure_ascii=False))
    return failures


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
