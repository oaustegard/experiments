"""mandelbrot and matmul from kernels/, rewritten in the pykernel subset.

Each @kernel function is ONE invocation's work. The grid is n*n invocations,
gid = row * n + col, so every list[float] buffer has n*n elements. The host side
(inputs, buffer sizes, checksum) mirrors kernels/<name>.py: the inputs come from the
same make() and the checksum is computed from the output buffer the same way.
"""
import importlib.util
import os
from collections.abc import Callable
from dataclasses import dataclass

from pykernel import kernel

HERE = os.path.dirname(os.path.abspath(__file__))
KERNELS_DIR = os.path.abspath(os.path.join(HERE, "..", "..", "kernels"))
MAX_ITER = 50  # kernels/mandelbrot.py run(): max_iter


@kernel(workgroup_size=64)
def mandelbrot(gid: int, out: list[float], n: int):
    max_iter = 50
    px = gid % n
    py = gid // n
    ci = 2.0 * py / n - 1.0
    cr = 2.0 * px / n - 1.5
    zr = 0.0
    zi = 0.0
    k = 0
    while k < max_iter:
        zr2 = zr * zr
        zi2 = zi * zi
        if zr2 + zi2 > 4.0:
            break
        zi = 2.0 * zr * zi + ci
        zr = zr2 - zi2 + cr
        k += 1
    out[gid] = float(k)


@kernel(workgroup_size=64)
def matmul(gid: int, a: list[float], b: list[float], c: list[float], n: int):
    i = gid // n
    j = gid % n
    s = 0.0
    for k in range(n):
        s += a[i * n + k] * b[k * n + j]
    c[gid] = s


# ---- host side -------------------------------------------------------------

def _load_kernels_module(filename):
    spec = importlib.util.spec_from_file_location(filename[:-3], os.path.join(KERNELS_DIR, filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _matmul_inputs(n):
    """The same two matrices kernels/matmul.py run(n) builds, flattened row-major."""
    make = _load_kernels_module("matmul.py").make
    a, b = make(n, 1), make(n, 2)
    return {"a": [x for row in a for x in row], "b": [x for row in b for x in row]}


def checksum_mandelbrot(out, n):
    """kernels/mandelbrot.py's checksum, computed from the escape counts in the output buffer."""
    inside = 0
    total_iters = 0
    for v in out[: n * n]:
        k = int(v)
        total_iters += k
        if k == MAX_ITER:
            inside += 1
    return float(inside * 1000000 + total_iters % 1000000)


def checksum_matmul(c, n):
    """kernels/matmul.py run()'s checksum, computed from the product in the output buffer."""
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += c[i * n + j] * float((i + j) % 5 + 1)
    return s


@dataclass(frozen=True)
class Case:
    name: str
    fn: Callable
    default_n: int
    inputs: Callable            # n -> {buffer name: flat list of floats}; others start zeroed
    checksum: Callable          # (flat output list, n) -> float


KERNELS = {
    "mandelbrot": Case("mandelbrot", mandelbrot, 500, lambda n: {}, checksum_mandelbrot),
    "matmul": Case("matmul", matmul, 200, _matmul_inputs, checksum_matmul),
}
