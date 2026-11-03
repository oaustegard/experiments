"""Write the inputs of bridge/deepzoom_job.js: the source image, the web modules and a config.

    python3 bridge/make_deepzoom_job.py IMAGE.png OUTDIR [--cfg '{"nBase": 1024, ...}'] [--local]
OUTDIR gets png.bin, mods.json and cfg.json for gpu_bridge.py submit; with --local, one inputs.json for run_local.mjs.
"""
import argparse, base64, json
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "web"
DEFAULT = {"base": 768, "nBase": 1024, "nTile": 256, "levels": [1, 2], "arms": ["residual"], "checkDraw": True}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image"); ap.add_argument("outdir"); ap.add_argument("--cfg", default="{}"); ap.add_argument("--local", action="store_true")
    a = ap.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    cfg = {**DEFAULT, "id": Path(a.image).stem, **json.loads(a.cfg)}
    mods = {n: (WEB / n).read_text() for n in ["trainer.js", "gputrain.js", "instanced.js", "deepzoom.js"]}
    png = Path(a.image).read_bytes()
    if a.local:
        (out / "inputs.json").write_text(json.dumps({"png": {"b64": base64.b64encode(png).decode()}, "cfg": cfg,
                                                       "mods": {"b64": base64.b64encode(json.dumps(mods).encode()).decode()}}))
    else:
        (out / "png.bin").write_bytes(png)
        (out / "mods.json").write_text(json.dumps(mods))
        (out / "cfg.json").write_text(json.dumps(cfg))
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
