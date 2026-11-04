#!/bin/bash
# Label data/batchN.txt (100 numbered Bluesky posts) as about AI or not, with a headless Claude child.
# Writes data/labelsN.json: 100 integers (1 = about AI, 0 = not) in post order, from a keyed reply
# (bare 100-item arrays came back 101-108 long, which would misalign every label after the slip).
b=$1
PROMPT='Below are 100 numbered Bluesky posts. For each, decide whether it is ABOUT AI: its subject is artificial intelligence, machine learning, LLMs or chatbots, AI products, companies, research, policy, safety, art, jobs or energy use, or using AI tools. A post that only mentions tech, computing, science or politics without AI is 0. A short conversational reply whose topic cannot be told is 0. Reply with ONLY a JSON object mapping each post number (as a string, "0" to "99") to 1 or 0, nothing else.'
claude -p --model haiku --output-format text "$PROMPT

$(cat data/batch$b.txt)" > data/labels$b.raw 2>data/labels$b.err
python3 - "$b" <<'PY'
import json, re, sys
b = sys.argv[1]
t = open(f"data/labels{b}.raw").read()
d = json.loads(re.search(r"\{[\s\S]*\}", t).group(0))
assert set(d) == {str(i) for i in range(100)} and set(d.values()) <= {0, 1}, (len(d), set(d.values()))
a = [d[str(i)] for i in range(100)]
json.dump(a, open(f"data/labels{b}.json", "w"))
print(b, "ok", sum(a), "of 100 about AI")
PY
