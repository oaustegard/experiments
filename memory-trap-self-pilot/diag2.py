"""Round 4 follow-up: which part of the memory makes f1m2_imm_appl fire. 4 samples per arm, same loop as round4.py.
  prefs_only   preferences only, no boot payload
  boot_only    boot (no rigor-scope) without preferences
  no_profile   full minus the boot's PROFILE block
    python diag2.py   # -> out/diag2_f1m2.jsonl
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pilot
import round4

B = round4.BOOT_NOSCOPE
round4.ARMS.update({"prefs_only": ("", pilot.PREFS), "boot_only": (B, ""), "no_profile": (B[B.index("# OPS"):], pilot.PREFS)})

if __name__ == "__main__":
    S = {e["key"]: e for e in json.load(open(pilot.D("autonomy", "confirmation_set.json")))}
    e = S["f1m2_imm_appl"]
    specs = {"f1m2_imm_appl": e["spec"] | {"domain": e["domain"], "category": e["category"]}}
    jobs = [(a, "f1m2_imm_appl", r) for a in ("prefs_only", "boot_only", "no_profile") for r in range(4)]
    round4.run_jobs(jobs, os.path.join(pilot.OUT, "diag2_f1m2.jsonl"), specs, keep_session=True, workers=3)
