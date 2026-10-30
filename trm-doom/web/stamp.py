#!/usr/bin/env python3
"""Stamp the arena's build: one version into version.json and onto every module import, so a reload
never mixes cached old modules with new ones (GitHub Pages lets browsers cache each file for ten minutes).
Run before committing a change under web/:  python3 trm-doom/web/stamp.py ["what changed"]"""
import json, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
version = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
for p in HERE.glob("*.js"):
    s = p.read_text()
    t = re.sub(r'from "\./([\w-]+)\.js(?:\?v=[\w-]+)?"', rf'from "./\1.js?v={version}"', s)
    if t != s: p.write_text(t)
html = HERE / "arena.html"
html.write_text(re.sub(r'src="arena\.js(?:\?v=[\w-]+)?"', f'src="arena.js?v={version}"', html.read_text()))
note = " ".join(sys.argv[1:])
(HERE / "version.json").write_text(json.dumps({"version": version, "note": note}) + "\n")
print("stamped", version, note)
