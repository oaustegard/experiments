"""Verify the WGSL arm: GPU output against a numpy float32 model of the same algorithm.

WGSL is f32, so the f64 checksums from kernels/*.py cannot match to 1e-9. The
criteria are:
  matmul      GPU checksum within 1e-4 relative of the numpy float32 checksum
  mandelbrot  at least 99.9% of pixels with an identical escape count (numpy float32)
The relative error against the f64 checksum from `python3.11 kernels/<k>.py` is
printed for each kernel but is not a pass criterion.

Usage: python3 verify.py [--kernel mandelbrot,matmul] [--n N] [--runs 5]
Runs build.sh and run_gpu.js itself. Exit status 0 only if every kernel passes.
"""
import argparse
import json
import os
import subprocess
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from kernels_gpu import KERNELS, MAX_ITER

OUT = os.path.join(HERE, "out")
KERNELS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "kernels"))
F32 = np.float32
MATMUL_REL_TOL = 1e-4
MANDEL_MATCH_MIN = 0.999


def np_mandelbrot_f32(n):
    """Escape counts with the same float32 operations, in the same order, as the shader."""
    grid = n * n
    gid = np.arange(grid, dtype=np.int64)
    py = (gid // n).astype(F32)
    px = (gid % n).astype(F32)
    nf = F32(n)
    ci = (F32(2.0) * py) / nf - F32(1.0)
    cr = (F32(2.0) * px) / nf - F32(1.5)
    zr = np.zeros(grid, F32)
    zi = np.zeros(grid, F32)
    k = np.zeros(grid, np.int64)
    active = np.ones(grid, bool)
    for _ in range(MAX_ITER):
        zr2 = zr * zr
        zi2 = zi * zi
        active &= ~((zr2 + zi2) > F32(4.0))
        new_zi = (F32(2.0) * zr) * zi + ci
        new_zr = (zr2 - zi2) + cr
        zi = np.where(active, new_zi, zi)
        zr = np.where(active, new_zr, zr)
        k += active
    return k


def np_matmul_f32(a_flat, b_flat, n):
    a = np.asarray(a_flat, F32).reshape(n, n)
    b = np.asarray(b_flat, F32).reshape(n, n)
    c = np.zeros((n, n), F32)
    for k in range(n):  # k-outer accumulation: same order per element as the shader's loop
        c += a[:, k, None] * b[None, k, :]
    return c.reshape(-1)


def f64_checksum(name):
    out = subprocess.run(["python3.11", os.path.join(KERNELS_DIR, f"{name}.py")],
                         capture_output=True, text=True, check=True).stdout
    rec = json.loads(out.strip().splitlines()[-1])
    return rec["n"], rec["checksum"]


def rel(a, b):
    return abs(a - b) / max(abs(b), 1e-300)


def run_gpu(name, n, runs):
    p = subprocess.run(["bash", os.path.join(HERE, "build.sh"), "--kernel", name, "--n", str(n)],
                       capture_output=True, text=True, check=False)
    if p.returncode != 0:
        raise RuntimeError(f"build.sh failed: {p.stderr[-800:]}")
    p = subprocess.run(["node", os.path.join(HERE, "run_gpu.js"),
                        os.path.join(OUT, f"{name}.json"), "--runs", str(runs)],
                       capture_output=True, text=True, timeout=900, check=False)
    lines = [l for l in p.stdout.strip().splitlines() if l.strip()]
    if not lines:
        raise RuntimeError(f"run_gpu.js printed nothing; stderr: {p.stderr[-800:]}")
    rec = json.loads(lines[-1])
    if not rec.get("ok"):
        raise RuntimeError(json.dumps(rec)[:3000])
    return rec


def verify(name, n, runs):
    case = KERNELS[name]
    rec = run_gpu(name, n, runs)
    gpu = np.fromfile(rec["outputs"]["out" if name == "mandelbrot" else "c"], dtype="<f4")
    grid = n * n
    assert gpu.size == grid, (gpu.size, grid)
    gpu_cs = case.checksum(gpu.tolist(), n)

    result = {"kernel": name, "n": n, "gpu_ms_median": rec["gpu_ms_median"],
              "gpu_ms_all": [round(t, 2) for t in rec["times_ms"]], "adapter": rec.get("adapter")}
    if name == "mandelbrot":
        ref_k = np_mandelbrot_f32(n)
        gpu_k = np.rint(gpu).astype(np.int64)
        match = float(np.mean(gpu_k == ref_k))
        ref_cs = case.checksum(ref_k.astype(float).tolist(), n)
        result.update(match_fraction=match, mismatched_pixels=int(np.sum(gpu_k != ref_k)),
                      gpu_checksum=gpu_cs, numpy_f32_checksum=ref_cs,
                      passed=match >= MANDEL_MATCH_MIN)
        result["criterion"] = f"escape-count match >= {MANDEL_MATCH_MIN}"
    else:
        a_flat = np.fromfile(os.path.join(OUT, f"{name}.a.in.f32"), dtype="<f4")
        b_flat = np.fromfile(os.path.join(OUT, f"{name}.b.in.f32"), dtype="<f4")
        ref_c = np_matmul_f32(a_flat, b_flat, n)
        ref_cs = case.checksum(ref_c.tolist(), n)
        denom = np.maximum(np.abs(ref_c), 1e-6)
        elem_rel = float(np.max(np.abs(gpu.astype(np.float64) - ref_c) / denom))
        result.update(gpu_checksum=gpu_cs, numpy_f32_checksum=ref_cs,
                      checksum_rel_err_vs_numpy=rel(gpu_cs, ref_cs),
                      max_elem_rel_err_vs_numpy=elem_rel,
                      passed=rel(gpu_cs, ref_cs) <= MATMUL_REL_TOL)
        result["criterion"] = f"checksum rel err vs numpy f32 <= {MATMUL_REL_TOL}"

    if n == case.default_n:  # the f64 kernel only runs at its default N
        f64_n, f64_cs = f64_checksum(name)
        if f64_n != n:
            raise RuntimeError(f"kernels/{name}.py default N is {f64_n}, expected {n}")
        result.update(f64_n=f64_n, f64_checksum=f64_cs,
                      gpu_rel_err_vs_f64=rel(gpu_cs, f64_cs),
                      numpy_f32_rel_err_vs_f64=rel(ref_cs, f64_cs))
    else:
        result.update(f64_n=None, f64_checksum=None, gpu_rel_err_vs_f64=float("nan"),
                      numpy_f32_rel_err_vs_f64=float("nan"))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kernel", default="")
    ap.add_argument("--n", type=int, default=0, help="override N (default: each kernel's default)")
    ap.add_argument("--runs", type=int, default=5)
    a = ap.parse_args()
    names = a.kernel.split(",") if a.kernel else list(KERNELS)
    all_ok = True
    for name in names:
        n = a.n or KERNELS[name].default_n
        try:
            r = verify(name, n, a.runs)
        except Exception as e:  # noqa: BLE001 -- report and keep going: one kernel's failure does not hide the other
            all_ok = False
            print(f"{name}: FAILED: {e}")
            continue
        all_ok &= r["passed"]
        print(f"== {name} n={r['n']}  {'PASS' if r['passed'] else 'FAIL'}  ({r['criterion']})")
        print(f"   adapter: {r.get('adapter')}")
        print(f"   gpu median {r['gpu_ms_median']:.1f} ms over {len(r['gpu_ms_all'])} runs {r['gpu_ms_all']}")
        if name == "mandelbrot":
            print(f"   escape-count match {r['match_fraction']*100:.4f}% ({r['mismatched_pixels']} pixels differ)")
        else:
            print(f"   max element rel err vs numpy f32 {r['max_elem_rel_err_vs_numpy']:.3e}")
        print(f"   checksum gpu {r['gpu_checksum']!r}  numpy-f32 {r['numpy_f32_checksum']!r}")
        print(f"   f64 (python3.11 kernels/{name}.py, n={r['f64_n']}) {r['f64_checksum']!r}")
        print(f"   rel err vs f64: gpu {r['gpu_rel_err_vs_f64']:.3e}   numpy-f32 {r['numpy_f32_rel_err_vs_f64']:.3e}")
    print("ALL PASS" if all_ok else "NOT ALL PASS")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
