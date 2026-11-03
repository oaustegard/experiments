"""fp32 export -> 8-bit export, the form the chat page ships (graph shared with base_q8 / A8p_q8).

    python3 quant8.py export/base export/base_q8

onnxruntime's MatMulNBitsQuantizer: every MatMul (lm_head included) to int8, symmetric, block 32,
accuracy_level 4; embeddings, norms and rotary caches stay fp32, so meta.json packs them as bf16.
"""
import json, shutil, sys
from pathlib import Path

import onnx
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, DefaultWeightOnlyQuantConfig

from export import split

HERE = Path(__file__).resolve().parent


def main():
    src, dst = (HERE / p for p in sys.argv[1:3])
    cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=8)
    q = MatMulNBitsQuantizer(onnx.load(str(src / "model.onnx")), bits=8, block_size=32, is_symmetric=True,
                             accuracy_level=4, algo_config=cfg)
    q.process()
    shutil.rmtree(dst, ignore_errors=True)
    dst.mkdir(parents=True)
    q.model.save_model_to_file(str(dst / "model.onnx"), use_external_data_format=True)
    m = onnx.load(str(dst / "model.onnx"), load_external_data=False)
    locs = {e.value for t in m.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
            for e in t.external_data if e.key == "location"}
    assert locs == {"model.onnx.data"}, locs
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(src / f, dst / f)
    print(json.dumps({k: v for k, v in split(dst).items() if k != "regions"}))


if __name__ == "__main__":
    main()
