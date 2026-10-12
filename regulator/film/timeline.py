"""Lay the narration out in time: python3 timeline.py -> timeline.json

Each scene gets HEAD of air, its lines one after another with GAP between them, and TAIL;
title, board and end cards are held for a fixed time. Line times are absolute seconds.
"""
import json
from pathlib import Path
from script import SCENES

HERE = Path(__file__).resolve().parent
HEAD, GAP, TAIL, XF = 0.7, 0.45, 1.0, 0.9
CHIME = 7.2
HOLD = dict(title=6.0, board=20.0, end=9.0)

def main():
    nar = {(n["s"], n["l"]): n for n in json.load(open(HERE / "narration.json"))}
    t, out = 0.0, []
    for i, s in enumerate(SCENES):
        t0 = t; lines = []
        if s["lines"]:
            t += HEAD
            for j, (spk, text, cards) in enumerate(s["lines"]):
                n = nar[(i, j)]
                lines.append(dict(spk=spk, text=text, cards=cards, t0=round(t, 3), t1=round(t + n["dur"], 3), wav=n["wav"]))
                t += n["dur"] + GAP
                if s.get("chime") and s["chime"][0] == j:   # leave room for the repeater to sound
                    lines[-1]["chime"] = dict(t=round(t, 3), level=s["chime"][1]); t += CHIME + 0.6
            t += TAIL - GAP
        else:
            t += HOLD.get(s.get("kind"), s.get("hold", 5.0))
        if s.get("kind") == "cast":
            t = max(t, t0 + 3.8)
        out.append(dict(i=i, part=s["part"], kind=s.get("kind", ""), who=s.get("who", ""), shot=s["shot"], kb=s["kb"],
                        dial=s.get("dial"), t0=round(t0, 3), t1=round(t, 3), lines=lines))
    json.dump(dict(fps=24, xf=XF, duration=round(t + 0.5, 3), scenes=out), open(HERE / "timeline.json", "w"), indent=1, ensure_ascii=False)
    print(f"{len(out)} scenes, {t/60:.2f} min")
    reveal = next(s["t0"] for s in out if s["part"] == "reveal")
    print(f"pause card at {reveal/60:.2f} min")

if __name__ == "__main__":
    main()
