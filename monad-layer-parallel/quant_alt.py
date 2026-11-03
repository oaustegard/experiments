"""int4 arms that compensate rounding error, after the registered decision rule fired (round-to-nearest
int4 failed prediction 2: loops 5.8% -> 18.6%). Quantizes an existing fp32 export of the harness
(../monad-weight-edit/export/<src>) with onnxruntime's MatMulNBitsQuantizer:

    python3 quant_alt.py gptq A8p A8p_gptq      # GPTQ, block 32, calibration on dev-set answers
    python3 quant_alt.py hqq  A8p A8p_hqq       # half-quadratic quantization, block 32, no calibration

GPTQ (Frantar et al. 2022) rounds each weight column in turn and folds the rounding error into the
not-yet-rounded columns using the inverse Hessian of that layer's calibration inputs: closed form, no
gradient descent. Calibration: 24 sequences of A8p's own dev answers (prompt + answer, <= 256 tokens),
disjoint from the 16 prompts the KL screen uses only in the sense that the screen regenerates; both
come from the dev split, never from the test split.
"""
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MWE = HERE.parent / "monad-weight-edit"
sys.path.insert(0, str(MWE))

import numpy as np  # noqa: E402
import onnx  # noqa: E402
from onnxruntime.quantization import CalibrationDataReader  # noqa: E402
from onnxruntime.quantization.matmul_nbits_quantizer import (  # noqa: E402
    GPTQWeightOnlyQuantConfig, HQQWeightOnlyQuantConfig, MatMulNBitsQuantizer)

from common import load  # noqa: E402
from export import split  # noqa: E402
from fold import calib  # noqa: E402


class Reader(CalibrationDataReader):
    def __init__(self, model_dir, n=24, L=256):
        tok, _ = load(MWE / "models/A8p")
        g = onnx.load(str(model_dir / "model.onnx"), load_external_data=False)
        layers = sum(1 for i in g.graph.input if i.name.endswith(".key"))
        kv = next(i.type.tensor_type.shape.dim[1].dim_value for i in g.graph.input if i.name.endswith(".key"))
        self.feeds = []
        for ids, _ in calib(tok, "A8p__gpu_dev.jsonl", n=n, L=L):
            T = len(ids)
            f = {"input_ids": np.array([ids], np.int64), "attention_mask": np.ones((1, T), np.int64),
                 "logits_index": np.array([T - 1], np.int64)}
            for l in range(layers):
                f[f"past_key_values.{l}.key"] = np.zeros((1, kv, 0, 64), np.float32)
                f[f"past_key_values.{l}.value"] = np.zeros((1, kv, 0, 64), np.float32)
            self.feeds.append(f)
        self.i = 0

    def get_next(self):
        if self.i >= len(self.feeds):
            return None
        self.i += 1
        return self.feeds[self.i - 1]

    def rewind(self):
        self.i = 0


def main():
    algo, src, dst = sys.argv[1:4]
    sdir, ddir = MWE / "export" / src, MWE / "export" / dst
    if algo == "gptq":
        cfg = GPTQWeightOnlyQuantConfig(calibration_data_reader=Reader(sdir), block_size=32, percdamp=0.01,
                                        actorder=False, mse=False, perchannel=True)
        q = MatMulNBitsQuantizer(str(sdir / "model.onnx"), bits=4, block_size=32, is_symmetric=True, algo_config=cfg)
    else:
        q = MatMulNBitsQuantizer(str(sdir / "model.onnx"), bits=4, block_size=32,
                                 algo_config=HQQWeightOnlyQuantConfig(block_size=32, bits=4))
    import os
    os.chdir(sdir)   # INC's GPTQ reloads the model from bytes and resolves external data against the cwd
    q.process()
    os.chdir(HERE)
    shutil.rmtree(ddir, ignore_errors=True)
    ddir.mkdir(parents=True)
    q.model.save_model_to_file(str(ddir / "model.onnx"), use_external_data_format=True)
    # the saver names the data file after the model; the job and ort_gen expect model.onnx.data
    m = onnx.load(str(ddir / "model.onnx"), load_external_data=False)
    locs = {e.value for t in m.graph.initializer if t.data_location == onnx.TensorProto.EXTERNAL
            for e in t.external_data if e.key == "location"}
    assert locs == {"model.onnx.data"}, locs
    for f in ("genai_config.json", "tokenizer.json", "tokenizer_config.json"):
        shutil.copy(sdir / f, ddir / f)
    print(json.dumps({k: v for k, v in split(ddir).items() if k != "regions"}))


if __name__ == "__main__":
    main()
