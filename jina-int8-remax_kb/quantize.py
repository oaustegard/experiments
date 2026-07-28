"""Dynamic int8 quantization of the Jina v5-nano retrieval ONNX export.

Weight-only dynamic quantization (QInt8): weights -> int8, activations quantized
on the fly at inference. Standard CPU-transformer recipe; no calibration data
needed. Shrinks the 847 MB fp32 export and speeds CPU MatMul/Gemm.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "model.onnx"
DST = HERE / "model.int8.onnx"
PREP = HERE / "model.prep.onnx"


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def main() -> int:
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.shape_inference import quant_pre_process

    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1

    print(f"source fp32: {mb(SRC):.1f} MB")
    # Pre-process (shape inference + graph opt) — onnxruntime recommends this
    # before dynamic quantization; avoids missing-shape warnings/failures.
    t0 = time.time()
    try:
        quant_pre_process(str(SRC), str(PREP), skip_symbolic_shape=False)
        src = PREP
        print(f"pre-processed in {time.time()-t0:.1f}s")
    except Exception as exc:  # noqa: BLE001
        print(f"pre-process skipped ({type(exc).__name__}: {exc}); quantizing raw")
        src = SRC

    t0 = time.time()
    quantize_dynamic(str(src), str(DST), weight_type=QuantType.QInt8)
    print(f"quantized in {time.time()-t0:.1f}s")
    print(f"int8: {mb(DST):.1f} MB  ({mb(SRC)/mb(DST):.1f}x smaller than fp32 source)")
    if PREP.exists():
        PREP.unlink()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
