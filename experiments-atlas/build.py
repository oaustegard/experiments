"""Build web/index.html from tpl.html and data.json (replaces __DATA__)."""
import json
from pathlib import Path

here = Path(__file__).resolve().parent
tpl = (here / "tpl.html").read_text()
data = json.loads((here / "data.json").read_text())
out = tpl.replace("__DATA__", json.dumps(data, ensure_ascii=False))
(here / "web" / "index.html").write_text(out)
print(f"{len(data)} experiments -> web/index.html")
