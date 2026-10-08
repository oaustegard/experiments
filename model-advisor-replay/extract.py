"""Build replay rows from the transcripts branch: first human prompt, entrypoint, model sequence.

Run inside a claude-workspace clone with the `transcripts` branch fetched (see notes.py):
    python3 extract.py commits.txt > rows.jsonl
Each row is what `model_advisor.py replay` reads. The rows hold the prompts, so they stay private.
"""
import json, subprocess, sys
rows = []
for c in open(sys.argv[1]).read().split():
    f = subprocess.run(["git", "ls-tree", "-r", "--name-only", c], capture_output=True, text=True).stdout.split("\n")[0]
    blob = subprocess.run(["git", "show", f"{c}:{f}"], capture_output=True, text=True, errors="replace").stdout
    first, entry, seq, sid = None, None, [], None
    for line in blob.splitlines():
        try:
            d = json.loads(line)
        except ValueError:
            continue
        if d.get("isSidechain"):
            continue
        sid = sid or d.get("sessionId")
        if d.get("type") == "user" and first is None and not d.get("isMeta"):
            c_ = d["message"]["content"]
            if isinstance(c_, list):
                c_ = " ".join(b.get("text", "") for b in c_ if isinstance(b, dict) and b.get("type") == "text")
            if c_ and c_.strip() and not c_.lstrip().startswith("<"):
                first, entry = c_, d.get("entrypoint")
        if d.get("type") == "assistant":
            m = d["message"].get("model")
            if m and m != "<synthetic>" and (not seq or seq[-1] != m):
                seq.append(m)
    if not first:
        continue
    models = {}
    for m in seq:
        models[m] = models.get(m, 0) + 1
    rows.append({"session": (sid or f)[:8], "file": f, "first_prompt": first, "entrypoint": entry,
                 "model_seq": seq, "models": {seq[0]: 1} if seq else {}})
for r in rows:
    print(json.dumps(r))
