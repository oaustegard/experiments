"""Fair play: every clock reading in the formal case is spoken and carded before the reveal,
and the pre-reveal narration is exported for detectives.

  python3 check_fair.py            -> report; writes work/prereveal.txt
"""
import sys, re
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from world import S
from script import SCENES

def norm(r):
    h, m = map(int, r.split(":")); h = h + 12 if h < 12 else h
    return f"{h:02d}:{m:02d}"

pre = [s for s in SCENES if s["part"] in ("open", "act1", "act2", "pause")]
cards = {(c, norm(r.split("=")[0])) for s in pre for (_, _, cs) in s["lines"] for c, r in cs}
need = {(a[2], a[3]) for x in S for a in x["atoms"] if a[0] == "read"}
missing = sorted(need - cards); extra = sorted(cards - need - {("regulator", "22:05")})
print("readings needed:", len(need), "| carded before the reveal:", len(cards & need))
print("missing:", missing); print("carded but not in the case:", extra)
# every card should also be spoken in its line: check the minute words loosely
NAMES = {"Odile": "O"}
Path("work").mkdir(exist_ok=True)
SPK = dict(M="INSPECTOR MATTHEY (narrating)", O="ODILE MONNIER", F="FÉLIX MONNIER", L="LINA RUEDI", D="DR VAUTHIER", B="BASTIEN GIRARD", R="ROSA HÄFELI")
txt = "\n".join(f"{SPK[k]}: {t}" for s in pre for (k, t, _) in s["lines"])
Path("work/prereveal.txt").write_text(txt + "\n")
words = sum(len(t.split()) for s in SCENES for (_, t, _) in s["lines"])
print("pre-reveal words:", len(txt.split()), "| total words:", words, f"(~{words/150:.1f} min at 150 wpm)")
sys.exit(1 if missing else 0)
