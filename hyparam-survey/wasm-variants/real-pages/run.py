#!/usr/bin/env python3
"""Do the wasm-variant deltas survive contact with real Parquet pages?

The sibling study measured 2.7x from a wider copy on a 4 MiB incompressible
blob. That corpus is one literal run of four million bytes. Real Parquet pages
are not shaped like that, so this runs the same variants over pages taken from
a production file.

  1. encode.mjs   fetch NYC TLC yellow_tripdata_2024-01 (48 MB, real, ZSTD),
                  read five columns spanning DOUBLE / INT64 / INT32 / STRING,
                  and re-encode each on its own as SNAPPY. Real values, real
                  distributions, the codec hysnappy actually decodes.
  2. dump.mjs     intercept hyparquet's decompressor to capture every
                  compressed page verbatim.
  3. analyze.mjs  walk each page's snappy tag stream and count bytes emitted
                  as literals against bytes emitted by back-references. This
                  is what decides whether a wider copy can help at all.
  4. drive.mjs    time every variant on every column, nine trials each in a
                  fresh process, with a bootstrap CI on the ratio of medians
                  and a byte-weighted rollup over the file's real column mix.

Writes results.json and mix.json. Needs network for step 1 and clang for the
sibling build (run ../build_and_bench.py first).
"""
import json
import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).parent
STEPS = [
    ("encode.mjs", "fetch + re-encode real columns as snappy"),
    ("dump.mjs", "capture the compressed pages"),
    ("analyze.mjs", "literal vs copy mix per page"),
    ("drive.mjs", "time every variant, bootstrap CIs, weighted rollup"),
]


def main() -> int:
    if not (HERE.parent / "build").exists():
        sys.exit("run ../build_and_bench.py first — it produces build/*.wasm")
    for script, what in STEPS:
        print(f"\n=== {script}: {what}")
        r = subprocess.run(["node", script], cwd=HERE, check=False)
        if r.returncode != 0:
            return r.returncode
    res = json.loads((HERE / "results.json").read_text())
    print(f"\nweighted rollup: {json.dumps(res.get('weighted', {}), indent=2)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
