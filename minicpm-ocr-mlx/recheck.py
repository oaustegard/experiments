"""Sub-5-minute recheck of this experiment.

1. PROCESSOR   minicpm_ocr.py, through the same prompt path as `mlx_vlm.generate`,
               still produces transformers' token ids and slice sizes for every
               reference image (results/reference.json). Needs the HF tokenizer
               and configs, not the weights.
2. PROSE       the results table in README.md is what results/*.json say.

It does not rerun the model: that is parity.py and sweep.py, which take an
hour on CPU. See README.md.

    python recheck.py [HF_DIR]          # HF_DIR defaults to a hub download
    python recheck.py --table           # print the table built from results/
"""

import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
VARIANTS = [  # name, quantized parts, bf16 checkpoint size in GB (read back from disk)
    ("fp32", "none (fp32 reference port)", None),
    ("q8-all", "8-bit: MLP, attention, linear attention, embeddings", 1.90),
    ("q4-default", "4-bit: MLP, attention (mlx-vlm 4.6 default)", 2.16),
    ("q4-linattn", "4-bit: + linear attention", 1.89),
    ("q4-all", "4-bit: + embeddings / tied LM head", 1.52),
]
CASES = ["paragraph", "page", "small", "table"]


def table() -> str:
    lines = [
        "| Checkpoint | Quantized | bf16 size | Teacher-forced top-1 (tokens) | Mean log p | Greedy text = transformers |",
        "|---|---|---|---|---|---|",
    ]
    for name, parts, size in VARIANTS:
        sweep = json.loads((RESULTS / f"sweep-{name}.json").read_text())
        agree = sum(sweep[c]["tokens"] - sweep[c]["disagreements"] for c in CASES)
        total = sum(sweep[c]["tokens"] for c in CASES)
        logp = sum(sweep[c]["mean_logprob"] * sweep[c]["tokens"] for c in CASES) / total
        parity_file = RESULTS / f"parity-{name}.json"
        if parity_file.exists():
            parity = json.loads(parity_file.read_text())
            greedy = f"{sum(parity[c]['text_equal'] for c in CASES)}/{len(CASES)}"
        else:
            greedy = "not run"
        size_text = f"{size:.2f} GB" if size else "4.9 GB (fp32)"
        lines.append(f"| `{name}` | {parts} | {size_text} | {agree}/{total} | {logp:.4f} | {greedy} |")
    return "\n".join(lines)


def check_processor(hf_dir: Path) -> list[str]:
    import numpy as np
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import get_model_and_args, load_config, load_processor
    from PIL import Image

    sys.path.insert(0, str(HERE))
    from convert import stage

    failures = []
    reference = json.loads((RESULTS / "reference.json").read_text())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)
        stage(hf_dir, path)
        config = load_config(path)
        get_model_and_args(config, path)  # imports minicpm_ocr.py, which registers the processor
        processor = load_processor(path)
        for name, ref in reference.items():
            task = ref["prompt"].split("\n")[2].split("<|im_end|>")[0]
            prompt = apply_chat_template(processor, config, task, num_images=1)
            out = processor(text=[prompt], images=[Image.open(RESULTS / f"{name}.png").convert("RGB")])
            ids = np.asarray(out["input_ids"]).reshape(-1).tolist()
            sizes = np.asarray(out["tgt_sizes"][0]).tolist()
            ok = ids == ref["input_ids"] and sizes == ref["target_sizes"]
            print(f"processor {name}: {'ok' if ok else 'MISMATCH'} ({len(ids)} ids, slices {sizes})")
            if not ok:
                failures.append(f"processor {name}")
    return failures


def main() -> int:
    if "--table" in sys.argv:
        print(table())
        return 0
    failures = []
    if table() not in (HERE / "README.md").read_text():
        failures.append("README table differs from results/*.json (run recheck.py --table)")
    print("prose:", "ok" if not failures else failures[0])
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        hf_dir = Path(args[0])
    else:
        from huggingface_hub import snapshot_download

        hf_dir = Path(snapshot_download("openbmb/MiniCPM-OCR-1.0", allow_patterns=["*.json", "*.jinja"]))
    failures += check_processor(hf_dir)
    print("FAIL:" if failures else "all checks pass", *failures)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
