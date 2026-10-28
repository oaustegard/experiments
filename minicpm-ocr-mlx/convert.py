"""Convert openbmb/MiniCPM-OCR-1.0 to an MLX checkpoint that stock mlx-vlm loads.

The checkpoint's config names minicpm_ocr.py as its `model_file`, so
`mlx_vlm.generate --model OUT` works without patching mlx-vlm.

    python convert.py OUT [--hf openbmb/MiniCPM-OCR-1.0] [--bits 4|8] [--dtype bfloat16]
                          [--quantize mlp,attn,linear_attn,embed]
"""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from huggingface_hub import snapshot_download
from mlx_vlm.convert import convert

HERE = Path(__file__).resolve().parent


def stage(hf_dir: Path, staging: Path) -> None:
    """Link the HF files into `staging`, add minicpm_ocr.py and point the config at it."""
    for f in hf_dir.iterdir():
        if f.name != "config.json" and f.is_file():
            (staging / f.name).symlink_to(f.resolve())
    shutil.copy(HERE / "minicpm_ocr.py", staging / "minicpm_ocr.py")
    config = json.loads((hf_dir / "config.json").read_text())
    config["model_file"] = "minicpm_ocr.py"
    (staging / "config.json").write_text(json.dumps(config, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--hf", default="openbmb/MiniCPM-OCR-1.0", help="HF repo id or local directory")
    ap.add_argument("--bits", type=int, choices=[4, 8], help="quantize the text model to this many bits")
    ap.add_argument("--group-size", type=int, default=64)
    ap.add_argument("--dtype", default="bfloat16", help="dtype of unquantized weights")
    ap.add_argument(
        "--quantize",
        default="mlp,attn",
        help="comma list of text-model parts to quantize: mlp, attn, linear_attn, embed "
        "(embed also covers the tied lm_head); the vision tower always stays unquantized",
    )
    args = ap.parse_args()

    parts = set(args.quantize.split(","))

    def predicate(path: str, module) -> bool:
        if not hasattr(module, "to_quantized") or path.startswith(("vision_tower", "vit_merger", "merger")):
            return False
        if module.weight.shape[-1] % args.group_size:
            return False
        if "embed_tokens" in path:
            return "embed" in parts
        if ".linear_attn." in path:
            return "linear_attn" in parts
        if ".self_attn." in path:
            return "attn" in parts
        return ".mlp." in path and "mlp" in parts

    hf_dir = Path(args.hf) if Path(args.hf).is_dir() else Path(snapshot_download(args.hf))
    with tempfile.TemporaryDirectory() as tmp:
        stage(hf_dir, Path(tmp))
        convert(
            str(tmp),
            mlx_path=args.out,
            quantize=args.bits is not None,
            q_bits=args.bits or 4,
            q_group_size=args.group_size,
            dtype=args.dtype,
            quant_predicate=predicate,
        )
    config = json.loads((Path(args.out) / "config.json").read_text())
    assert config.get("model_file") == "minicpm_ocr.py", "convert dropped model_file"
    assert (Path(args.out) / "minicpm_ocr.py").is_file(), "convert did not copy minicpm_ocr.py"


if __name__ == "__main__":
    main()
