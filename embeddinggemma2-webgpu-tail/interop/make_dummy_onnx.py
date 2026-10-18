"""Write interop/dummy_eg2.onnx: a stand-in with EmbeddingGemma 2's output
contract, for testing the ORT gpu-buffer -> RemexTail handoff without weights.

Inputs  x [B,S,768] float, mask [B,S] float
Outputs sentence_embedding [B,768] = L2-normalized masked mean of x
        last_hidden_state  [B,S,768] = x
"""
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

OUT = Path(__file__).resolve().parent / "dummy_eg2.onnx"

x = helper.make_tensor_value_info("x", TensorProto.FLOAT, ["B", "S", 768])
m = helper.make_tensor_value_info("mask", TensorProto.FLOAT, ["B", "S"])
se = helper.make_tensor_value_info("sentence_embedding", TensorProto.FLOAT, ["B", 768])
lh = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT, ["B", "S", 768])
init = [
    numpy_helper.from_array(np.array(1.0, np.float32), "one"),
    numpy_helper.from_array(np.array([1], np.int64), "axes1"),
    numpy_helper.from_array(np.array([-1], np.int64), "axesm1"),
    numpy_helper.from_array(np.array(1e-12, np.float32), "eps"),
]
nodes = [
    helper.make_node("Mul", ["x", "one"], ["last_hidden_state"]),
    helper.make_node("Unsqueeze", ["mask", "axesm1"], ["m3"]),
    helper.make_node("Mul", ["x", "m3"], ["xm"]),
    helper.make_node("ReduceSum", ["xm", "axes1"], ["s"], keepdims=0),
    helper.make_node("ReduceSum", ["m3", "axes1"], ["cnt"], keepdims=0),
    helper.make_node("Div", ["s", "cnt"], ["pooled"]),
    helper.make_node("Mul", ["pooled", "pooled"], ["sq"]),
    helper.make_node("ReduceSum", ["sq", "axesm1"], ["ss"], keepdims=1),
    helper.make_node("Max", ["ss", "eps"], ["ssc"]),
    helper.make_node("Sqrt", ["ssc"], ["nrm"]),
    helper.make_node("Div", ["pooled", "nrm"], ["sentence_embedding"]),
]
model = helper.make_model(helper.make_graph(nodes, "dummy_eg2", [x, m], [se, lh], initializer=init),
                          opset_imports=[helper.make_opsetid("", 17)])
model.ir_version = 8
onnx.checker.check_model(model)
onnx.save(model, OUT)
print(OUT)
