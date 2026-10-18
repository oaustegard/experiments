"""Mandelbrot escape counts on an n x n grid. Pure scalar float/int loops."""
import json
import sys
import time


def run(n: int) -> float:
    max_iter = 50
    inside = 0
    total_iters = 0
    for py in range(n):
        ci = 2.0 * py / n - 1.0
        for px in range(n):
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
            total_iters += k
            if k == max_iter:
                inside += 1
    return float(inside * 1000000 + total_iters % 1000000)


DEFAULT_N = 500

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    t0 = time.perf_counter()
    r = run(n)
    t1 = time.perf_counter()
    print(json.dumps({"kernel": "mandelbrot", "n": n, "checksum": r, "seconds": t1 - t0}))
