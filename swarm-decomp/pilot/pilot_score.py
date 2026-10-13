#!/usr/bin/env python3
"""Score the optimisation-level pilot from the runner, not from the workers' own reports.
    pilot_score.py PILOT_DIR  -> per level: matched/assigned, mean frac; per function table; pilot.json"""
import json, subprocess, sys
from pathlib import Path
P = Path(sys.argv[1])
ASSIGNED = {
    "neatroff-reg": ["num_set", "str_rm", "num_setfmt", "tr_ta"],
    "neateqn-box": ["box_alloc", "box_size", "box_merge", "box_bracketsel"],
    "neatmail-mbox": ["mbox_len", "mbox_pos", "mbox_get", "mbox_ith"],
    "neatroff-sbuf": ["sbuf_init", "sbuf_add", "sbuf_append", "sbuf_printf"],
    "neatcc-tok": ["tok_addr", "tok_jump", "tok_get", "tok_num"],
}
out = {}
for lv in ("O0", "O1", "O2"):
    filt = [x for s, fs in ASSIGNED.items() for f in fs for x in ("--filter", f"{s}/{f}")]
    j = P / f"score-{lv}.json"
    subprocess.run([sys.executable, "tests/run.py", "--suite", ",".join(ASSIGNED), "--json", str(j), "--quiet", *filt],
                   cwd=P / lv, check=True)
    out[lv] = json.loads(j.read_text())
for lv, r in out.items():
    n = sum(v["status"] == "PASS" for v in r.values())
    print(f"{lv}: {n}/{len(r)} match ({100*n/len(r):.0f}%), mean frac {sum(v['frac'] for v in r.values())/len(r):.3f}")
for t in sorted(out["O0"]):
    print(f"  {t:32s} " + "  ".join(f"{lv}:{out[lv][t]['status'][:4]} {out[lv][t]['frac']:.2f}" for lv in out)
          + f"   {out['O2'][t]['reason'][:50]}")
(P / "pilot.json").write_text(json.dumps(out, indent=1))
