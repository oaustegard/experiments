"""Dense n x n float matmul, naive i-k-j order, list-of-lists."""
import json
import sys
import time


def make(n: int, seed: int) -> list[list[float]]:
    m: list[list[float]] = []
    for i in range(n):
        row: list[float] = []
        for j in range(n):
            row.append(float((i * 31 + j * 17 + seed) % 97) / 97.0)
        m.append(row)
    return m


def matmul(a: list[list[float]], b: list[list[float]], n: int) -> list[list[float]]:
    c: list[list[float]] = []
    for i in range(n):
        ci = [0.0] * n
        ai = a[i]
        for k in range(n):
            aik = ai[k]
            bk = b[k]
            for j in range(n):
                ci[j] += aik * bk[j]
        c.append(ci)
    return c


def run(n: int) -> float:
    a = make(n, 1)
    b = make(n, 2)
    c = matmul(a, b, n)
    s = 0.0
    for i in range(n):
        for j in range(n):
            s += c[i][j] * float((i + j) % 5 + 1)
    return s


DEFAULT_N = 200

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    t0 = time.perf_counter()
    r = run(n)
    t1 = time.perf_counter()
    print(json.dumps({"kernel": "matmul", "n": n, "checksum": r, "seconds": t1 - t0}))
