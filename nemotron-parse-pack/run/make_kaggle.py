"""Fill kaggle_eval.py's placeholders and write the script to push.

    python3 run/make_kaggle.py --pages run/pilot8.txt --arms pilot --max-tokens 4096 --batch 4 --out /tmp/job.py
"""
import argparse
import json
from pathlib import Path

ENC = r"radio_model\.model\.blocks\.\d+\.(attn|mlp)\.(qkv|proj|fc1|fc2)\.weight"
SMALL = r"patch_generator\.(pos_embed|embedder|cls_token)|encoder\.conv[12]\.weight|encoder\.sum_proj\.weight"
DEC = r"decoder\.layers\.\d+\.((self_attn|encoder_attn)\.(q|k|v|out)_proj|fc[12])\.weight"
EMB = r"decoder\.embed_tokens\.weight|lm_head\.weight"

SCHEMES = {
    "A": [[ENC, "int4g", 32], [SMALL, "int8"], [DEC, "int8"], [EMB, "int8"]],
    "B": [[ENC, "int4g", 32], [SMALL, "int8"], [DEC, "int4g", 32], [EMB, "int8"]],
    "C": [[ENC, "int4g", 32], [SMALL, "int8"], [DEC, "int4g", 32], [EMB, "int4g", 32]],
    "D": [[ENC, "int8"], [SMALL, "int8"], [DEC, "int8"], [EMB, "int8"]],
}
ARMSETS = {"pilot": ["fp32", "A", "B", "C", "D"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pages", required=True)
    ap.add_argument("--arms", required=True, help="a set name, or comma-separated arm names")
    ap.add_argument("--max-tokens", type=int, default=9000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--teacher", default="fp32")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    names = ARMSETS.get(a.arms) or a.arms.split(",")
    arms = [[n, None if n == "fp32" else SCHEMES[n]] for n in names]
    src = (Path(__file__).parent / "kaggle_eval.py").read_text()
    for k, v in {"__PAGES__": Path(a.pages).read_text().strip(), "__ARMS__": json.dumps(arms),
                 "__MAX_TOKENS__": str(a.max_tokens), "__BATCH__": str(a.batch),
                 "__TEACHER__": a.teacher if "fp32" in names else ""}.items():
        src = src.replace(f'"{k}"', json.dumps(v)) if k in ("__PAGES__", "__ARMS__", "__TEACHER__") else src.replace(k, v)
    Path(a.out).write_text(src)


if __name__ == "__main__":
    main()
