"""Sub-8-bit Jina ONNX, with the embedding-table workaround.

Problem: MatMulNBits only quantizes MatMul nodes, leaving EuroBERT's large
multilingual token-embedding table (a Gather initializer, ~400 MB fp32) at full
precision — so naive int4 (465 MB) ends up BIGGER than int8 dynamic (212 MB),
which quantizes the whole graph.

Workaround: low-bit the MatMuls with MatMulNBits, THEN run int8 quantize_dynamic
to mop up the embedding Gather + any leftover weights. Result: int{2,3,4}
matmuls + int8 embeddings, monotonically smaller as bits drop.

Each variant is verified by actually running a forward pass on the CPU EP — if
ORT has no kernel for that bit width, encode() raises and we record it (that is
the practical inference floor, distinct from the quantization floor).
"""
from __future__ import annotations

import sys, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
SRC = HERE / "model.onnx"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _lib.paths import spoke
sys.path.insert(0, str(spoke("remax_kb")))
from remax_kb.embedders import JinaONNXEmbedder  # noqa: E402


def mb(p: Path) -> float:
    return p.stat().st_size / 1e6


def verify(model_path: Path) -> str:
    """Run a real encode; return 'ok (norm=..)' or 'RUN-FAIL: ..'."""
    try:
        emb = JinaONNXEmbedder(model_path=model_path, tokenizer_path=HERE / "tokenizer.json")
        v = emb.encode(["the quick brown fox jumps over the lazy dog"], prompt="document")
        return f"ok shape={v.shape} norm={float(np.linalg.norm(v[0])):.3f}"
    except Exception as exc:  # noqa: BLE001
        return f"RUN-FAIL: {type(exc).__name__}: {str(exc)[:160]}"


def main() -> int:
    import onnx
    from onnxruntime.quantization import quantize_dynamic, QuantType
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer

    print(f"source fp32: {mb(SRC):.1f} MB\n", flush=True)
    results = []
    # ORT's MatMulNBitsQuantizer asserts bits in {2,4,8} — 3-bit is unsupported,
    # so the sweep is 4 then 2 (8 is covered by int8 quantize_dynamic elsewhere).
    for bits in (4, 2):
        tmp = HERE / f".q{bits}_matmul.onnx"
        dst = HERE / f"model.q{bits}.onnx"
        print(f"=== {bits}-bit ===", flush=True)
        model = onnx.load(str(SRC))
        t0 = time.time()
        quant = MatMulNBitsQuantizer(model, bits=bits, block_size=32, is_symmetric=True)
        quant.process()
        quant.model.save_model_to_file(str(tmp), use_external_data_format=False)
        mm_mb = mb(tmp)
        print(f"  matmul-only: {mm_mb:.1f} MB ({time.time()-t0:.1f}s)", flush=True)
        # mop up the embedding Gather + leftover weights with int8 dynamic
        try:
            quantize_dynamic(str(tmp), str(dst), weight_type=QuantType.QUInt8)
            final_mb = mb(dst)
            print(f"  + int8 mop-up: {final_mb:.1f} MB", flush=True)
        except Exception as exc:  # noqa: BLE001
            print(f"  mop-up failed ({type(exc).__name__}: {str(exc)[:120]}); keeping matmul-only")
            dst.write_bytes(tmp.read_bytes())
            final_mb = mm_mb
        status = verify(dst)
        print(f"  verify: {status}\n", flush=True)
        tmp.unlink(missing_ok=True)
        results.append((bits, mm_mb, final_mb, status))

    print("=" * 64)
    print(f"{'bits':<6}{'matmul-only MB':>16}{'+int8 mop-up MB':>17}{'runs?':>8}")
    for bits, mm, fin, status in results:
        print(f"{bits:<6}{mm:>16.0f}{fin:>17.0f}{'yes' if status.startswith('ok') else 'NO':>8}")
    print("\nref: fp32 847 · int8-dynamic(all) 212 · int4-matmul-only 465")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
