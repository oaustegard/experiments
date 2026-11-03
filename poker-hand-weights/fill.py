#!/usr/bin/env python3
"""Write the numbers into RESULTS.md: each ⟦key⟧ becomes "+x ± y" from results/headtohead.json or
results/reference.json, followed by an evidence ref, and ⟦TABLES⟧ becomes summarize.py's tables. Run once
after the runs finish; it refuses if a key is missing.

  python3 poker-hand-weights/fill.py
"""
import json, re, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
res = {}
for f in ("headtohead.json", "reference.json"):
    for k, v in json.loads((HERE / "results" / f).read_text()).items():
        res[k] = (f, v)
doc = (HERE / "RESULTS.md").read_text()


def sub(m):
    k = m.group(1)
    if k == "TABLES":
        out = subprocess.run([sys.executable, str(HERE / "summarize.py")], capture_output=True, text=True, check=True).stdout
        return out.strip()
    if k not in res:
        sys.exit(f"missing result {k}")
    f, v = res[k]
    num = f"{v['bb100']:+.1f}".replace("-", "−")
    return f"{num} ± {v['ci95']:.1f} <!-- @evidence[[results/{f}#{k}.bb100]] -->"


doc = re.sub(r"⟦([^⟧]+)⟧", sub, doc)
(HERE / "RESULTS.md").write_text(doc)
print("filled")
