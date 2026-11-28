"""Rebuild bench.py's synthetic ANE encoder and count placement by op type, const separate."""
import sys, collections
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.models.compute_plan import MLComputePlan

C, H, S, L = 384, 1536, 256, 6          # same constants as bench.py
rng = np.random.default_rng(0)
w = lambda *s: (rng.standard_normal(s) * 0.02).astype(np.float32)


@mb.program(input_specs=[mb.TensorSpec(shape=(1, C, 1, S))], opset_version=ct.target.macOS14)
def prog(x):
    for _ in range(L):
        h = mb.gelu(x=mb.conv(x=x, weight=w(H, C, 1, 1)))
        x = mb.add(x=x, y=mb.conv(x=h, weight=w(C, H, 1, 1)))
        q = mb.reshape(x=x, shape=[1, C, S])
        s = mb.mul(x=mb.matmul(x=q, y=q, transpose_x=True), y=np.float32(C ** -0.5))
        o = mb.matmul(x=q, y=mb.softmax(x=s, axis=-1), transpose_y=True)
        x = mb.layer_norm(x=mb.add(x=x, y=mb.reshape(x=o, shape=[1, C, 1, S])), axes=[1])
    return x


m = ct.convert(prog, convert_to="mlprogram", compute_precision=ct.precision.FLOAT16,
               minimum_deployment_target=ct.target.macOS14)
m.save(sys.argv[1])
keep = ct.models.MLModel(sys.argv[1], compute_units=ct.ComputeUnit.CPU_AND_NE)
plan = MLComputePlan.load_from_path(path=keep.get_compiled_model_path(),
                                    compute_units=ct.ComputeUnit.CPU_AND_NE)
by = collections.defaultdict(collections.Counter)
for fn in plan.model_structure.program.functions.values():
    for op in fn.block.operations:
        u = plan.get_compute_device_usage_for_mlprogram_operation(op)
        d = type(u.preferred_compute_device).__name__ if u else "none"
        by[op.operator_name][d] += 1
tot = collections.Counter()
for k, v in sorted(by.items()):
    print(f"{k:28s} {dict(v)}")
    tot.update(v)
print("TOTAL", dict(tot))
nc = collections.Counter()
for k, v in by.items():
    if k != "const":
        nc.update(v)
print("NON-CONST", dict(nc))
