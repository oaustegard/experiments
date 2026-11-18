"""Build one arm: a model folder like datalab-to/indra-ocr with fake-quantized
weights (quant.py) and, optionally, a substituted n-gram table.

    python3 run/make_arm.py --src /tmp/indra-models --out /tmp/indra-eval/arms/s4g32 \
        --scheme s4g32 [--per rec=i8c,det=f16] [--lm lm.npz=PATH]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from quant import fake_model  # noqa: E402

ONNX = ["det", "orient", "script", "rec", "rec_world", "rec_cjk", "rec_rtl", "order",
        "mathcls", "math_enc", "math_dec_init", "math_dec_step"]
OTHER = ["meta.json", "math_meta.json", "lm.npz", "lm_world.npz", "lm_cjk.npz"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scheme", default="f32", help="default scheme for every model")
    ap.add_argument("--per", default="", help="model=scheme,... overrides")
    ap.add_argument("--lm", action="append", default=[], help="NAME.npz=PATH substitute")
    ap.add_argument("--only", default="", help="comma list of models to quantize (rest f32)")
    ap.add_argument("--no-lm", action="store_true", help="leave the n-gram tables out (no beam search)")
    a = ap.parse_args()
    per = dict(kv.split("=") for kv in a.per.split(",") if kv)
    only = set(a.only.split(",")) if a.only else None
    a.out.mkdir(parents=True, exist_ok=True)
    report = {}
    for name in ONNX:
        src = a.src / f"{name}.onnx"
        sch = per.get(name, a.scheme if (only is None or name in only) else "f32")
        if sch == "f32":
            shutil.copy(src, a.out / src.name)
            report[name] = {"scheme": "f32", "bytes": None}
            continue
        r = fake_model(str(src), str(a.out / src.name), sch,
                       other="f32" if sch == "f32" else "f16")
        report[name] = {"scheme": sch, "bytes": round(r["bytes"]),
                        "worst_rel_err": max((v["rel_err"] for v in r["tensors"].values()), default=0)}
    subs = dict(kv.split("=", 1) for kv in a.lm)
    for f in OTHER:
        if a.no_lm and f.startswith("lm"):
            continue
        src = Path(subs.get(f, a.src / f))
        if src.exists():
            shutil.copy(src, a.out / f)
    (a.out / "arm.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v.get("bytes") for k, v in report.items()}))


if __name__ == "__main__":
    main()
