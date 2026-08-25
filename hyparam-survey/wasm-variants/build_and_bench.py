#!/usr/bin/env python3
"""Can hysnappy's decoder be made faster, and does SIMD do it?

The survey claimed hysnappy's byte-loop `memcpy` is a byte loop *because* the
module had to stay under Chrome's old 4 KB synchronous-compile ceiling. That
was invented. This builds the alternatives and measures them.

Four source variants x {no SIMD, -msimd128}:

  base       as shipped
  b_i64      `unaligned_copy64` guards its 64-bit path on `sizeof(void *) == 8`,
             which is FALSE on wasm32, so the shipped build takes the
             two-32-bit-stores branch. WASM has native i64 regardless of
             pointer width.
  c_widecpy  the hand-written `memcpy` copies one byte at a time; this copies
             8 at a time with a byte tail. That memcpy is the literal-copy
             path in `writer_append`.
  e_both     b_i64 + c_widecpy

Each (variant, corpus) is timed in a FRESH PROCESS via one.mjs, because eight
codegen variants measured in one process contaminate each other through V8's
tiering — a first attempt in-process ranked the same binary 1st on one corpus
and 7th on another. The timing loop calls the wasm export only: no slice(), no
allocation, nothing to GC.

Needs clang with the wasm32 target and wasm-ld. Writes results.json.
"""
from __future__ import annotations

import json
import shutil
import statistics
import subprocess
import sys
import urllib.request
from pathlib import Path

HERE = Path(__file__).parent
SRC = HERE / "src"
BUILD = HERE / "build"

# hysnappy's C lives in its GitHub repo, not the npm tarball.
UPSTREAM = "https://raw.githubusercontent.com/hyparam/hysnappy/{commit}/c/uncompress.c"
COMMIT = "fab2cda3910eea3d13dd69375de16a2100131285"  # v1.1.1, 2026-02-21
SHA256 = "72db4af7d724dc896916a450a607d7acfc4bce363951a732963e164b5b7ebb23"

CFLAGS = ["--target=wasm32", "-O3", "-nostdlib",
          "-Wl,--export-all", "-Wl,--no-entry"]

BYTE_MEMCPY = """void *memcpy(void *dest, const void *src, size_t n) {
    char *d = dest;
    const char *s = src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}"""

WIDE_MEMCPY = """void *memcpy(void *dest, const void *src, size_t n) {
    unsigned char *d = dest;
    const unsigned char *s = src;
    while (n >= 8) {
        unsigned long long v;
        __builtin_memcpy(&v, s, 8);
        __builtin_memcpy(d, &v, 8);
        d += 8; s += 8; n -= 8;
    }
    while (n--) { *d++ = *s++; }
    return dest;
}"""

I64_GUARD_FROM = """static inline void unaligned_copy64(const void *src, void *dst) {
	if (sizeof(void *) == 8) {"""
I64_GUARD_TO = """static inline void unaligned_copy64(const void *src, void *dst) {
	if (1) { /* wasm has native i64 regardless of pointer width */"""

CORPORA = ["literal", "match", "json"]
TRIALS = 5


def fetch_source() -> str:
    SRC.mkdir(exist_ok=True)
    path = SRC / "base.c"
    if not path.exists():
        with urllib.request.urlopen(UPSTREAM.format(commit=COMMIT), timeout=60) as r:
            path.write_bytes(r.read())
    import hashlib
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != SHA256:
        sys.exit(f"uncompress.c sha256 {got}, pinned {SHA256}")
    return path.read_text()


def write_variants(base: str) -> None:
    i64 = base.replace(I64_GUARD_FROM, I64_GUARD_TO)
    assert i64 != base, "the sizeof(void*) guard moved; update I64_GUARD_FROM"
    wide = base.replace(BYTE_MEMCPY, WIDE_MEMCPY)
    assert wide != base, "the byte-loop memcpy moved; update BYTE_MEMCPY"
    both = i64.replace(BYTE_MEMCPY, WIDE_MEMCPY)
    assert both != i64
    (SRC / "b_i64.c").write_text(i64)
    (SRC / "c_widecpy.c").write_text(wide)
    (SRC / "e_both.c").write_text(both)


def build() -> dict[str, int]:
    BUILD.mkdir(exist_ok=True)
    sizes = {}
    for name in ("base", "b_i64", "c_widecpy", "e_both"):
        for simd in (False, True):
            out = BUILD / f"{name}{'_simd' if simd else ''}.wasm"
            cmd = ["clang", *CFLAGS, *(["-msimd128"] if simd else []),
                   "-o", str(out), str(SRC / f"{name}.c")]
            r = subprocess.run(cmd, capture_output=True, text=True, check=False)
            if r.returncode != 0:
                sys.exit(f"build failed: {' '.join(cmd)}\n{r.stderr}")
            sizes[out.name] = out.stat().st_size
    return sizes


def bench(wasm: Path, corpus: str) -> list:
    runs = []
    for _ in range(TRIALS):
        r = subprocess.run(["node", str(HERE / "one.mjs"), str(wasm), corpus],
                           capture_output=True, text=True, check=False, cwd=HERE)
        text = r.stdout.strip()
        if text == "WRONG":
            return "WRONG"
        if r.returncode != 0 or not text:
            sys.exit(f"one.mjs failed for {wasm.name}/{corpus}: {r.stderr[:400]}")
        runs.append(int(text))
    return runs


def main() -> int:
    if not shutil.which("clang"):
        sys.exit("clang not found; this variant study needs clang with the wasm32 target")
    base = fetch_source()
    write_variants(base)
    sizes = build()
    print("module sizes")
    for name, n in sorted(sizes.items()):
        print(f"  {name:24} {n} B")

    out = {
        "generated_by": "build_and_bench.py",
        "upstream": {"repo": "hyparam/hysnappy", "commit": COMMIT,
                     "file": "c/uncompress.c", "sha256": SHA256},
        "toolchain": subprocess.run(["clang", "--version"], capture_output=True,
                                    text=True, check=False).stdout.splitlines()[0],
        "cflags": " ".join(CFLAGS),
        "method": ("one fresh node process per (variant, corpus); the timing loop calls "
                   "the wasm export only, with no allocation in it"),
        "trials_per_cell": TRIALS,
        "sizes_bytes": sizes,
        "corpora": {
            "literal": "4 MiB incompressible pseudo-random; every byte a literal copy",
            "match": "4 MiB of a repeated 26-byte alphabet; almost all back-references",
            "json": "4 MiB of tiled realistic API JSON",
        },
        "results": {},
    }
    for corpus in CORPORA:
        out["results"][corpus] = {}
        for name in sorted(sizes):
            runs = bench(BUILD / name, corpus)
            out["results"][corpus][name] = ("WRONG" if runs == "WRONG"
                                            else {"median_mbps": int(statistics.median(runs)), "runs": runs})
        baseline = out["results"][corpus]["base.wasm"]["median_mbps"]
        print(f"\n{corpus}")
        rows = [(n, v) for n, v in out["results"][corpus].items()]
        rows.sort(key=lambda kv: -(kv[1]["median_mbps"] if kv[1] != "WRONG" else 0))
        for name, v in rows:
            if v == "WRONG":
                print(f"  {name:24} WRONG OUTPUT")
                continue
            v["vs_base"] = round(v["median_mbps"] / baseline, 2)
            print(f"  {name:24} {v['median_mbps']:>6} MB/s  {v['vs_base']:.2f}x base   {v['runs']}")

    (HERE / "results.json").write_text(json.dumps(out, indent=2) + "\n")
    print("\nwrote results.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
