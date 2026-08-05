"""What does a rebuild actually cost, and what does incremental save?

Two costs, and the second is the one that compounds:

  1. CI wall-clock per push.
  2. Git history. The index is a committed binary blob, so every rebuild stores
     a **complete new copy** — binary files do not delta-compress usefully. At
     N commits the repo carries N copies.

Also verifies the equivalence claim rather than asserting it: an incrementally
built matrix must be bit-identical to a full rebuild.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "repo-index"))
sys.path.insert(0, "/home/user/remex")
import hcindex as H  # noqa: E402
from ask import Encoder  # noqa: E402


def main() -> None:
    cfg = H.load_cfg(REPO)
    enc = Encoder()

    for label, extra in (("md+py (no json)", [".json"]), ("all text", [])):
        c = dict(cfg)
        c["extensions"] = [e for e in cfg["extensions"] if e not in extra]
        chunks = H.build_corpus(REPO, c)
        n = len(chunks)

        # full build, timed on a subset then extrapolated -- encoding the whole
        # 'all text' corpus twice would cost ~30 min for a number we can measure
        sample = chunks[:300]
        t0 = time.time()
        enc([x.text for x in sample])
        rate = len(sample) / (time.time() - t0)
        full_s = n / rate

        # incremental: edit one median-sized file, re-encode only its chunks
        from collections import Counter
        per_file = Counter(x.f for x in chunks)
        typical = sorted(per_file.values())[len(per_file) // 2]
        inc_s = typical / rate

        idx_mb = n * 384 * 2 / 8 / 2**20
        print(f"\n=== {label} ===")
        print(f"  {n:6d} chunks / {len(per_file)} files, {rate:.0f} chunks/s")
        print(f"  full rebuild        {full_s:7.1f} s")
        print(f"  incremental (1 file, {typical} chunks) {inc_s:7.1f} s "
              f"({full_s/max(inc_s,1e-9):.0f}x less encode)")
        print(f"  index blob          {idx_mb:7.2f} MB per commit")
        print(f"  git history @ 200 rebuilds {idx_mb*200:7.0f} MB")

    # ── equivalence check: incremental must be bit-identical to a full build ──
    print("\n=== equivalence ===")
    c = dict(cfg); c["extensions"] = [e for e in cfg["extensions"] if e != ".json"]
    chunks = H.build_corpus(REPO, c)[:400]
    full = enc([x.text for x in chunks])
    half, hashes, n_enc, n_re = H.incremental(chunks[:200], enc)
    combined, _, n_enc2, n_re2 = H.incremental(chunks, enc,
                                               prev_codes=half, prev_hashes=hashes)
    print(f"  seeded with {len(half)} rows, then rebuilt {len(chunks)}: "
          f"encoded {n_enc2}, reused {n_re2}")
    print(f"  bit-identical to full rebuild: {np.array_equal(combined, full)}")
    print(f"  max abs delta: {np.abs(combined - full).max():.3e}")


if __name__ == "__main__":
    main()
