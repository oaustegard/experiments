"""Read what DOOM Arena shipped back: python arena_logs.py [SESSION] [--sheet OUT.png]

Pulls the GPU-bridge comms checkout (.spokes/gpu-bridge-comms), lists arena sessions, and for one session
(the latest by default) prints a line per attempt; --sheet writes a contact sheet of its thumbnails.
"""
import json, subprocess, sys
from pathlib import Path

COMMS = Path(__file__).resolve().parents[3] / "gpu-bridge-comms"
subprocess.run(["git", "-C", str(COMMS), "pull", "-q", "--ff-only"], check=False)
logs = COMMS / "doom-logs"
sessions = sorted(p.name for p in logs.iterdir()) if logs.exists() else []
if not sessions: sys.exit("no arena sessions yet")
args = [a for a in sys.argv[1:] if not a.startswith("--")]
name = args[0] if args else sessions[-1]
d = logs / name
meta = json.loads((d / "session.json").read_text()) if (d / "session.json").exists() else {}
print("sessions:", ", ".join(sessions[-5:]))
print(f"== {name} on {meta.get('device')} skill {meta.get('skill')} global {meta.get('global')} opts {meta.get('opts')}")
for f in sorted(d.glob("[0-9][0-9][0-9]-*.json")):
    r = json.loads(f.read_text())
    s = r.get("stats", {})
    print(f"{f.stem:28s} {r['result']:11s} {r['seconds']:7.1f}s kills {r['kills']:7s} hp {r['health']:4d} plans {r.get('plans')} @{r.get('planMs')}ms stuck {s.get('stuck')} done {s.get('targets')}")
if (d / "end.json").exists(): print("ended:", (d / "end.json").read_text())
if "--sheet" in sys.argv:
    from PIL import Image
    out = sys.argv[sys.argv.index("--sheet") + 1]
    pngs = sorted(d.glob("*.png"))[-24:]
    if pngs:
        ims = [Image.open(p) for p in pngs]; w, h = ims[0].size
        sheet = Image.new("RGB", (w * 6, h * ((len(ims) + 5) // 6)))
        for i, im in enumerate(ims): sheet.paste(im, ((i % 6) * w, (i // 6) * h))
        sheet.save(out); print("sheet:", out)
