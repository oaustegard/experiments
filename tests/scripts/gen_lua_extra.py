#!/usr/bin/env python3
"""Write tests/suites/lua/extra/<nn>_<slug>.lua from lua_extra_scripts.py (committed output).
Run: python3 tests/scripts/gen_lua_extra.py"""
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import lua_extra_scripts as S  # noqa: E402

OUT = os.path.join(os.path.dirname(HERE), "suites", "lua", "extra")
os.makedirs(OUT, exist_ok=True)
for old in glob.glob(os.path.join(OUT, "*.lua")):
    os.remove(old)
for i, (slug, src) in enumerate(S.SCRIPTS, 1):
    with open(os.path.join(OUT, "%02d_%s.lua" % (i, slug)), "w") as f:
        f.write(src.strip("\n") + "\n")
print("wrote %d scripts to %s" % (len(S.SCRIPTS), OUT))
