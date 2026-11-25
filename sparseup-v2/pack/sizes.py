"""Pack sizes raw, gzip -9 and brotli q11 (what a static host would send). Writes results/sizes.json."""
import gzip
import json
from pathlib import Path

import brotli

HERE = Path(__file__).resolve().parent


def main():
    out = {}
    for pack in ("int8", "q4"):
        d = HERE / "packs" / pack
        if not d.exists():
            continue
        row = {}
        for f in ("weights.bin", "manifest.json", "tokenizer.json"):
            b = (d / f).read_bytes()
            row[f] = {"raw": len(b), "gzip9": len(gzip.compress(b, 9)), "brotli11": len(brotli.compress(b, quality=11))}
        row["total"] = {k: sum(v[k] for v in row.values()) for k in ("raw", "gzip9", "brotli11")}
        out[pack] = row
        print(pack, {k: f"{v / 1e6:.1f} MB" for k, v in row["total"].items()}, flush=True)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "sizes.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
