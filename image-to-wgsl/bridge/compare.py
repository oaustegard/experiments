"""Compare a fetched bridge result with the torch decode and the original image.

    python3 bridge/compare.py bridge/spec.bin FETCH_DIR result.json -> results/bridge_<jobid>.json
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import codec  # noqa: E402


def main():
    spec_path, fetch_dir, result_path = sys.argv[1:4]
    spec = json.loads(Path(spec_path).read_text())
    res = json.loads(Path(result_path).read_text())
    ret = res.get("outputs") or res.get("result") or res
    rows = {r["id"]: r for r in ret["rows"]}
    out = {"device": res.get("device"), "tabName": res.get("tabName"), "ua": res.get("ua"),
           "adapter": ret.get("adapter"), "shaders": []}
    for s in spec:
        r = dict(rows.get(s["id"], {"id": s["id"], "error": "missing"}))
        f = Path(fetch_dir) / f"img_{s['id']}.npy"
        if f.exists():
            img = np.load(f).reshape(s["h"], s["w"], 4)[..., :3]
            stem = Path(s["run"]).stem.split("_")[0]
            image = HERE / "data" / "kodak" / f"{stem}.png"
            run, resid = codec.load_run(HERE / s["run"], image if image.exists() else None)
            ref = codec.render_image(run["P"], run["bg"], run["H"], run["W"], run["A"], run["layout"], n=s["n"],
                                     residual=resid if s["id"].endswith("_res") else None).numpy()
            ref8 = np.rint(ref * 255).astype(np.uint8)
            d = np.abs(img.astype(int) - ref8.astype(int))
            r.update(max_abs_vs_torch=int(d.max()), frac_px_differ=float((d.max(-1) > 0).mean()),
                     psnr_vs_torch=codec.psnr8(img, ref8))
            if "target" in run:
                tgt8 = np.rint(run["target"].numpy() * 255).astype(np.uint8)
                r.update(psnr_gpu=codec.psnr8(img, tgt8), psnr_torch=codec.psnr8(ref8, tgt8))
        out["shaders"].append(r)
    dest = HERE / "results" / f"bridge_{res.get('id', Path(result_path).stem)}.json"
    dest.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
