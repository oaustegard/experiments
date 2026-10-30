"""Top-down picture of a run: walls, the trace (blue -> red over time), targets (green exit, cyan key, magenta other), S = stuck.
python test/tracemap.py RESULT.json LEVEL_INDEX OUT.png"""
import json, subprocess, sys
from pathlib import Path
from PIL import Image, ImageDraw
here = Path(__file__).resolve().parent
r = json.load(open(sys.argv[1])); L = (r.get("outputs", {}).get("levels") if "outputs" in r else None or [r])
L = (r["outputs"]["levels"] if "outputs" in r else [r])[int(sys.argv[2])]
lines = json.loads(subprocess.run(["node", str(here / "maplines.mjs"), L["map"]], capture_output=True, text=True).stdout)
xs = [v for l in lines for v in (l[0], l[2])]; ys = [v for l in lines for v in (l[1], l[3])]
x0, x1, y0, y1 = min(xs) - 50, max(xs) + 50, min(ys) - 50, max(ys) + 50
s = 1100 / max(x1 - x0, y1 - y0); P = lambda x, y: ((x - x0) * s, (y1 - y) * s)
im = Image.new("RGB", (int((x1 - x0) * s), int((y1 - y0) * s)), (17, 17, 17)); d = ImageDraw.Draw(im)
for l in lines: d.line([P(l[0], l[1]), P(l[2], l[3])], fill=(224, 160, 48) if l[5] else (190, 190, 190) if l[4] else (80, 80, 80), width=1)
T = L["trace"]
for k in range(1, len(T)):
    f = k / len(T); d.line([P(T[k-1][1], T[k-1][2]), P(T[k][1], T[k][2])], fill=(int(255 * f), 80, int(255 * (1 - f))), width=3)
for e in L["events"]:
    if e["ev"] == "target":
        c = {"exit": (0, 255, 0), "key": (0, 255, 255)}.get(e["kind"], (255, 0, 255)); x, y = P(e["x"], e["y"]); d.ellipse([x - 4, y - 4, x + 4, y + 4], outline=c)
    if e["ev"] == "stuck": x, y = P(e["x"], e["y"]); d.text((x, y), "S", fill=(255, 255, 0))
d.text((10, 10), f'{L["map"]} {L["result"]} {L["seconds"]}s kills {L["kills"]}/{L["totalKills"]}', fill=(255, 255, 255))
im.save(sys.argv[3])
