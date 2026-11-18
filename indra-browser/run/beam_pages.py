"""Pages where the n-gram beam search read at least one line (opb_eval.py's
"beams" count): the only pages an n-gram arm can change.
    python3 run/beam_pages.py /tmp/indra-eval/out/fp32b > /tmp/indra-eval/beam_pages.txt"""
import json
import sys
from pathlib import Path

n = 0
for f in sorted(Path(sys.argv[1], "pages").glob("*.json")):
    r = json.loads(f.read_text())
    n += 1
    if r.get("beams", 0) > 0:
        print(r["page_id"])
print(f"{n} pages read", file=sys.stderr)
