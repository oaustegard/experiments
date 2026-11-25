"""Ask Jev again about 300 held-out cases: how far its second answer is from its first (the ceiling for any student).

    python3 edgewalk/run/repeat.py   -> results/jev_repeat.json
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[1]))
from _lib.paths import spoke  # noqa: E402
import sandwich  # noqa: E402
import train  # noqa: E402
sys.path.insert(0, str(spoke("claude-workspace") / "scripts"))
import model_advisor  # noqa: E402

rows = [json.loads(l) for l in train.LABELS.read_text().splitlines()]
test = [r for r in rows if train.held_out(r["case"])][:300]
with ThreadPoolExecutor(8) as ex:
    again = list(ex.map(lambda r: float(model_advisor._jev(sandwich.state(r["case"]), sandwich.QUESTION)["is_sandwich"]["noul"]), test))
a, b = np.array([r["p"] for r in test]), np.array(again)
out = {"n": len(test), "mae": float(np.abs(a - b).mean()), "zone_agreement": float(np.mean([train.zone(u) == train.zone(v) for u, v in zip(a, b)]))}
print(out)
(HERE.parent / "results" / "jev_repeat.json").write_text(json.dumps(out, indent=1))
