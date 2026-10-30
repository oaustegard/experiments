"""Summarize a fetched campaign job: per-level table, a GIF and a contact sheet of its frames.
python summarize.py RESULT.json FETCH_DIR OUT_PREFIX [W H]"""
import json, sys
from pathlib import Path
import numpy as np
from PIL import Image

r = json.load(open(sys.argv[1])); d = Path(sys.argv[2]); prefix = sys.argv[3]
W, H = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (320, 200)
o = r.get("outputs") or {}
print(f"device {r.get('device')} status {r.get('status')} run {r.get('runMs', 0) / 1000:.0f}s planner {o.get('planner')}")
if r.get("error"): print(r["error"][:800])
rows = []
for L in o.get("levels", []):
    s = L["stats"]
    rows.append({"map": L["map"], "result": L["result"], "game_s": L["seconds"], "wall_s": round(L["wallMs"] / 1000, 1),
                 "kills": f'{L["kills"]}/{L["totalKills"]}', "items": f'{L["items"]}/{L["totalItems"]}', "hp": L["health"],
                 "fine": s["finePlans"], "fine_broken": s["fineBroken"], "coarse": s["coarsePlans"], "coarse_empty": s["coarseEmpty"],
                 "stuck": s["stuck"], "done": s["targets"]})
for x in rows: print(x)
json.dump({"job": r.get("id"), "device": r.get("device"), "planner": o.get("planner"), "levels": rows}, open(prefix + ".json", "w"), indent=1)
f = d / "frames.npy"
if f.exists():
    p = np.load(d / "palette.npy").astype(np.uint32)
    pal = np.stack([p & 255, (p >> 8) & 255, (p >> 16) & 255], 1).astype(np.uint8)
    fr = np.load(f).reshape(-1, H, W)
    ims = [Image.fromarray(pal[x]) for x in fr]
    ims[0].save(prefix + ".gif", save_all=True, append_images=ims[1:], duration=150, loop=0)
    k = max(1, len(ims) // 16); sel = ims[::k][:16]
    sheet = Image.new("RGB", (W * 4, H * ((len(sel) + 3) // 4)))
    for i, im in enumerate(sel): sheet.paste(im, ((i % 4) * W, (i // 4) * H))
    sheet.save(prefix + "_sheet.png")
    print(len(ims), "frames ->", prefix + ".gif")
