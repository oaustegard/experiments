#!/usr/bin/env python3
"""Clone Oskar's public repos and write code files as JSONL for the healing data mix.

Held-out split: files changed in the last 14 days get the newest dates, so the
data step's newest-5% eval set comes from recent work. Files with anything that
looks like a credential are dropped, not redacted.
"""
import json, os, re, subprocess, sys, time

REPOS = ["experiments", "claude-skills", "muninn-utilities", "m5", "remex"]
EXT = {".py", ".js", ".mjs", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".h", ".cc", ".cpp", ".hpp", ".java",
       ".rb", ".sh", ".mojo", ".jl", ".swift", ".kt", ".sql", ".html", ".css", ".zig", ".lua", ".r", ".toml", ".yaml", ".yml"}
SKIP_DIR = re.compile(r"(^|/)(node_modules|vendor|dist|build|third_party|\.git|prboom|__pycache__|\.venv|site-packages)(/|$)")
SECRET = re.compile(r"(ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-[A-Za-z0-9_-]{20,}|AKIA[0-9A-Z]{16}|"
                    r"xox[abpr]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY|AIza[0-9A-Za-z_-]{30,}|"
                    r"(?i:(api[_-]?key|secret|token|password)\s*[:=]\s*['\"][A-Za-z0-9_\-/+=]{24,}['\"]))")

out = open(sys.argv[1] if len(sys.argv) > 1 else "/content/code.jsonl", "w")
root = "/content/repos"
os.makedirs(root, exist_ok=True)
n_files = n_drop = n_chars = 0
for r in REPOS:
    d = os.path.join(root, r)
    if not os.path.exists(d):
        rc = subprocess.call(["git", "clone", "-q", "--filter=blob:none", f"https://github.com/oaustegard/{r}.git", d])
        if rc:
            print("skip", r)
            continue
    recent = subprocess.run(["git", "-C", d, "log", "--since=14 days ago", "--name-only", "--pretty=format:%ct"],
                            capture_output=True, text=True).stdout
    last = {}
    ts = 0
    for line in recent.splitlines():
        if line.isdigit():
            ts = int(line)
        elif line.strip():
            last.setdefault(line.strip(), ts)
    files = subprocess.run(["git", "-C", d, "ls-files"], capture_output=True, text=True).stdout.splitlines()
    for f in files:
        if SKIP_DIR.search(f) or os.path.splitext(f)[1].lower() not in EXT or f.endswith((".min.js", "-lock.json", ".lock")):
            continue
        p = os.path.join(d, f)
        try:
            sz = os.path.getsize(p)
            if not (200 <= sz <= 200_000):
                continue
            text = open(p, encoding="utf-8").read()
        except Exception:
            continue
        lines = text.count("\n") + 1
        if len(text) / lines > 200:
            continue
        if SECRET.search(text):
            n_drop += 1
            continue
        out.write(json.dumps({"repo": r, "path": f, "date": last.get(f, 0), "text": text}) + "\n")
        n_files += 1
        n_chars += len(text)
    print(r, "done", n_files, flush=True)
print(f"files {n_files} chars {n_chars/1e6:.1f}M dropped-for-secrets {n_drop}")
