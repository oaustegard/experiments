"""Dynamic-Python workload: string building, dict counting, sorting.
Represents glue code that type-specialising compilers help least with."""
import json
import sys
import time

SYLLABLES = ["ka", "ri", "mo", "tu", "le", "sa", "no", "vi", "da", "pe", "zo", "hu"]


def make_word(x: int) -> str:
    parts: list[str] = []
    k = 1 + x % 4
    for _ in range(k):
        parts.append(SYLLABLES[x % 12])
        x = (x * 1103515245 + 12345) % 2147483648
        x = x // 7
    return "".join(parts)


def run(n: int) -> float:
    counts: dict[str, int] = {}
    x = 42
    for _ in range(n):
        x = (x * 1103515245 + 12345) % 2147483648
        w = make_word(x)
        counts[w] = counts.get(w, 0) + 1
    top = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    acc = 0
    for i, (w, c) in enumerate(top):
        acc += (i + 1) * c + len(w)
    return float(acc * 100000 + len(counts))


DEFAULT_N = 500000

if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_N
    t0 = time.perf_counter()
    r = run(n)
    t1 = time.perf_counter()
    print(json.dumps({"kernel": "dictwords", "n": n, "checksum": r, "seconds": t1 - t0}))
