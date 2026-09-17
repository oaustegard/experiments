"""Replace every <!-- table: NAME --> block in RESULTS.md with make_tables.py output."""
import re, subprocess, sys
from pathlib import Path
here = Path(__file__).parent
out = subprocess.run([sys.executable, str(here / "make_tables.py")], capture_output=True, text=True, check=True).stdout
secs = {m.group(1): m.group(2).strip() for m in re.finditer(r"## (.+?)\n\n(.*?)(?=\n\n## |\Z)", out, re.S)}
doc = (here / "RESULTS.md").read_text()
# A block body may be empty and must never cross into the next marker.
pat = re.compile(r"<!-- table: (.+?) -->\n((?:(?!<!-- table:|<!-- /table -->).)*?)<!-- /table -->", re.S)
n_before = len(re.findall("<!-- table:", doc))
doc2 = pat.sub(lambda m: f"<!-- table: {m.group(1)} -->\n{secs[m.group(1)]}\n<!-- /table -->", doc)
assert len(re.findall("<!-- table:", doc2)) == n_before == len(re.findall("<!-- /table -->", doc2)), "marker count changed"
(here / "RESULTS.md").write_text(doc2)
print("regenerated", n_before, "tables")
