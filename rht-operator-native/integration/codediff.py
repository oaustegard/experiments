"""Share of codes differing between variants, computed where the codes are.

python codediff.py <label> <codes_dir>  -> out/<label>__codediff.json
"""
import json, sys
from pathlib import Path
import numpy as np

label, D = sys.argv[1], Path(sys.argv[2])
res = {}
for a in sorted(D.glob(f"{label}__main__*.npy")):
    key = a.name.split("__main__")[1][:-4]
    A = np.load(a)
    row = {}
    for other in ("branch", "main2"):
        b = D / f"{label}__{other}__{key}.npy"
        if b.exists():
            B = np.load(b)
            row[other] = dict(coords=float((A != B).mean()), rows=float((A != B).any(1).mean()))
    res[key] = row
Path("out").mkdir(exist_ok=True)
json.dump(res, open(f"out/{label}__codediff.json", "w"), indent=1)
print(json.dumps(res)[:400])
