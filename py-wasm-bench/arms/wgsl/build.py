"""Translate kernels_gpu.KERNELS to out/<name>.wgsl, plus a manifest and f32 inputs.

out/<name>.json   manifest the GPU runner reads: entry, workgroups, buffers (binding,
                  access, length, input file), uniforms (values), outputs
out/<name>.wgsl   the shader (pykernel output)
out/<name>.<buf>.in.f32   little-endian float32 input buffer

Usage: python3 build.py [--kernel mandelbrot,matmul] [--n N]
"""
import argparse
import array
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kernels_gpu import KERNELS
from pykernel import translate

OUT = os.path.join(HERE, "out")


def build_one(case, n, outdir=OUT):
    os.makedirs(outdir, exist_ok=True)
    spec = translate(case.fn)
    man = spec.manifest()
    wgsl_name = f"{case.name}.wgsl"
    with open(os.path.join(outdir, wgsl_name), "w") as f:
        f.write(spec.wgsl)

    inputs = case.inputs(n)
    grid = n * n
    for b in man["buffers"]:
        b["length"] = grid
        b["file"] = None
        if b["name"] in inputs:
            vals = inputs[b["name"]]
            assert len(vals) == grid, (b["name"], len(vals), grid)
            fname = f"{case.name}.{b['name']}.in.f32"
            arr = array.array("f", vals)
            if sys.byteorder != "little":
                arr.byteswap()
            with open(os.path.join(outdir, fname), "wb") as f:
                f.write(arr.tobytes())
            b["file"] = fname

    scalars = {"n": n}
    for u in man["uniforms"]:
        if u["name"] == "pk_count":
            u["value"] = grid
        elif u["name"] in scalars:
            u["value"] = scalars[u["name"]]
        else:
            raise KeyError(f"no value for uniform {u['name']!r}")

    man.update(
        n=n,
        grid=grid,
        workgroups=math.ceil(grid / man["workgroup_size"]),
        wgsl=wgsl_name,
        outputs=[b["name"] for b in man["buffers"] if b["access"] == "read_write"],
    )
    with open(os.path.join(outdir, f"{case.name}.json"), "w") as f:
        json.dump(man, f, indent=1)
    return os.path.join(outdir, f"{case.name}.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="")
    ap.add_argument("--n", type=int, default=0, help="override the default N (testing only)")
    a = ap.parse_args()
    names = a.kernel.split(",") if a.kernel else list(KERNELS)
    for name in names:
        case = KERNELS[name]
        n = a.n or case.default_n
        path = build_one(case, n)
        print(f"built {name} n={n} -> {path}")


if __name__ == "__main__":
    main()
