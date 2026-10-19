"""Copy the rotation and codebook the demo needs into web/assets/.

Usage: python3 -I run/prep_assets.py

Writes web/assets/params.json (d, bits, seed, rotation, kappa, boundaries,
centroids, and the RHT plan the GPU kernels rotate with) and web/assets/R.f32.txt (raw little-endian float32 bytes of R,
row-major). The .txt extension is deliberate: the GitHub Pages builder drops
.f32 and .bin files. R is remex seed-42 'rht' for d=768 and does not depend
on the data, so the demo never reads fixtures/ at runtime.
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FX = ROOT / "fixtures"
ASSETS = ROOT / "web" / "assets"
D = 768


def main() -> int:
    p = json.loads((FX / "params.json").read_text())
    if p["d"] != D or p["bits"] != 2:
        raise SystemExit(f"fixture params d={p['d']} bits={p['bits']}: demo needs d=768 bits=2")
    r_bytes = (FX / "R.f32").read_bytes()
    if len(r_bytes) != D * D * 4:
        raise SystemExit(f"R.f32 is {len(r_bytes)} bytes, expected {D * D * 4}")

    ASSETS.mkdir(parents=True, exist_ok=True)
    out = {
        "d": p["d"],
        "bits": p["bits"],
        "seed": p["seed"],
        "rotation": p["rotation"],
        "remex_version": p["remex_version"],
        "kappa": p["kappa"],
        "boundaries": p["boundaries"],
        "centroids": p["centroids"],
        "rht": p["rht"],
        "R": {"file": "R.f32.txt", "dtype": "float32-le", "shape": [D, D], "layout": "row-major"},
    }
    (ASSETS / "params.json").write_text(json.dumps(out, indent=1) + "\n")
    (ASSETS / "R.f32.txt").write_bytes(r_bytes)
    print(f"wrote {ASSETS / 'params.json'} and {ASSETS / 'R.f32.txt'} ({len(r_bytes)} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
