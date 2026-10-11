"""Lay the narration on a clock: prologue, then for each stop a flight, the figure inking in, the lines.

  python3 timeline.py   -> timeline.json (read by film.js), narration.wav (24 kHz mono)
"""
import json, math
from pathlib import Path
import numpy as np, soundfile as sf

HERE = Path(__file__).resolve().parent
SR = 24000
GAP = 0.6            # between sentences
LEAD = 1.6           # after arriving, before Aun speaks (the figure starts inking)
TAIL = 1.2           # after the last sentence of a stop, before the next flight
TITLE_LEAD = 5.0     # opening, before the prologue

sky = json.load(open(HERE.parent / "work/sky.json"))
epic = json.load(open(HERE.parent / "work/epic.json"))
cons = {c["id"]: c for c in sky["cons"]}
narr = json.load(open(HERE / "narration.json"))

def dist(a, b):
    la1, la2, dl = map(math.radians, (a[1], b[1], b[0] - a[0]))
    return math.acos(max(-1, min(1, math.sin(la1) * math.sin(la2) + math.cos(la1) * math.cos(la2) * math.cos(dl))))

audio = np.zeros(0, np.float32)
def place(t, line):
    global audio
    y, sr = sf.read(HERE / "takes" / line["wav"], dtype="float32"); assert sr == SR
    i = int(t * SR)
    if len(audio) < i + len(y): audio = np.pad(audio, (0, i + len(y) - len(audio)))
    audio[i:i + len(y)] += y
    return dict(t=round(t, 3), dur=round(len(y) / SR, 3), text=line["text"])

by_seg = {}
for l in narr: by_seg.setdefault(l["seg"], []).append(l)

t = TITLE_LEAD; intro = []
for l in by_seg[0]:
    intro.append(place(t, l)); t += l["dur"] + GAP
t += 0.8
start = [cons[epic["stops"][0]["id"]]["center"][0] - 30, 62.0]
prev = [start[0] + 14, start[1]]
stops = []
for j, st in enumerate(epic["stops"]):
    c = cons[st["id"]]["center"]
    fly = min(6.0, max(2.8, 2.4 + 2.2 * dist(prev, c)))
    s = dict(cons=st["id"], t0=round(t, 3), t1=round(t + fly, 3)); t += fly + LEAD
    s["lines"] = []
    for l in by_seg[j + 1]:
        s["lines"].append(place(t, l)); t += l["dur"] + GAP
    t += TAIL - GAP
    s["t2"] = round(t, 3); stops.append(s); prev = c
outro = t + 0.3
cut = outro + 7.6
duration = cut + 10.0
# centre of the whole cord for the closing wide shot
v = np.mean([[math.cos(math.radians(cons[s["cons"]]["center"][1])) * math.cos(math.radians(cons[s["cons"]]["center"][0])),
              math.cos(math.radians(cons[s["cons"]]["center"][1])) * math.sin(math.radians(cons[s["cons"]]["center"][0])),
              math.sin(math.radians(cons[s["cons"]]["center"][1]))] for s in stops], axis=0)
end_center = [math.degrees(math.atan2(v[1], v[0])), math.degrees(math.asin(v[2] / np.linalg.norm(v)))]
tl = dict(title=epic["title"], start=start, anchor=[0, 90], intro=dict(lines=intro), stops=stops,
          outro=round(outro, 3), cut=round(cut, 3), duration=round(duration, 3), endCenter=end_center, fps=24)
(HERE / "timeline.json").write_text(json.dumps(tl, indent=1, ensure_ascii=False))
audio = np.pad(audio, (0, max(0, int(duration * SR) - len(audio))))
sf.write(HERE / "narration.wav", audio, SR)
print(f"duration {duration/60:.2f} min; speech {sum(l['dur'] for l in narr)/60:.2f} min; {len(stops)} stops; end centre {end_center}")
