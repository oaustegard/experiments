#!/bin/bash
# Judge a batch of Bluesky accounts (vet.py writes the file). Prints {"0": {"keep": 0|1, "why": "..."}, ...}.
f=$1; n=$2
PROMPT="Below are $n Bluesky accounts, numbered [0] to [$((n-1))], each with handle, display name, bio, posting stats and six sample posts. They are candidates for a curated list of PEOPLE who post thoughtful commentary about AI: researchers, engineers, builders, writers, analysts, policy people, artists engaging with AI. Keep (1) an account only if it is an individual human whose own posts discuss AI substantively. Reject (0) bots, automated or scheduled feeds, news or link aggregators, brand or company accounts, marketers and growth or 'make money with AI' accounts, AI agents or personas posting as themselves, spam, and accounts whose posts are mostly not about AI. Reply with ONLY a JSON object mapping each number as a string to {\"keep\": 1 or 0, \"why\": \"at most 8 words\"}, nothing else."
raw=$(claude -p --model haiku --output-format text "$PROMPT

$(cat "$f")")
python3 - "$n" "$raw" <<'PY'
import json, re, sys
n, raw = int(sys.argv[1]), sys.argv[2]
d = json.loads(re.search(r"\{[\s\S]*\}", raw).group(0))
assert set(d) == {str(i) for i in range(n)}, sorted(d)
print(json.dumps(d))
PY
