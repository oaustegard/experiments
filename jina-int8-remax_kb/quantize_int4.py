"""int4 (block) weight quantization of the Jina v5-nano ONNX export, via
onnxruntime's MatMulNBitsQuantizer. Tests whether — since remax 1-bits the
vectors downstream anyway — embedder weights can go below int8.
"""
from __future__ import annotations

import sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / "model.onnx"
DST = HERE / "model.int4.onnx"


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def main() -> int:
    import onnx
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    if not SRC.exists():
        print(f"missing {SRC}", file=sys.stderr)
        return 1
    print(f"source fp32: {mb(SRC):.1f} MB", flush=True)
    model = onnx.load(str(SRC))
    t0 = time.time()
    quant = MatMulNBitsQuantizer(model, bits=4, block_size=32, is_symmetric=True)
    quant.process()
    print(f"quantized int4 in {time.time()-t0:.1f}s", flush=True)
    quant.model.save_model_to_file(str(DST), use_external_data_format=False)
    print(f"int4: {mb(DST):.1f} MB  ({mb(SRC)/mb(DST):.1f}x smaller than fp32)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
