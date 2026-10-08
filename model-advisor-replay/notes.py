"""Tally the model advisor's own verdict lines from archived interactive transcripts.

Run inside a claude-workspace clone with the `transcripts` branch fetched (private):
    git fetch --shallow-since=2026-09-10 origin transcripts
    git log --format='%H %s' FETCH_HEAD | grep -v scheduled | awk '!seen[$4]++ {print $1}' > commits.txt
    python3 notes.py commits.txt > notes.tsv          # session, date, first "Model advisor:" line
"""
import re
import subprocess
import sys

for c in open(sys.argv[1]).read().split():
    f = subprocess.run(["git", "ls-tree", "-r", "--name-only", c], capture_output=True, text=True).stdout.split("\n")[0]
    blob = subprocess.run(["git", "show", f"{c}:{f}"], capture_output=True, text=True, errors="replace").stdout
    m = re.search(r'Model advisor:[^"\\]*', blob)
    date = f.split("/")[-1][:8]
    print(f"{f.split('-')[1]}\t{date[:4]}-{date[4:6]}-{date[6:]}\t{m.group(0) if m else ''}")
