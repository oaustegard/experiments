"""Fill run/kaggle_gen.py's CONFIG block and push it as a Kaggle GPU job.

    python3 run/launch.py SLUG --pages pages.txt [--engines vllm hf] [--dtype float32] [--skip DIR ...]
        [--set 'PROMPT="..."' --set MAX_LEN=16384 ...]

--skip names folders of page JSONs already pulled; their pages are left out, so a job that hit its
deadline is resumed by launching the rest. The script appends the job's own source (base64) as
SELF_SRC, because the workers re-run that file and Kaggle may not give the job a __file__.
"""
from __future__ import annotations

import argparse
import base64
import json
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HUB = Path("/home/user/claude-workspace")


def build(pages, engines, dtype, max_new, deadline_h, batch, extra=()):
    src = (HERE / "kaggle_gen.py").read_text()
    sets = {"PAGES": json.dumps(pages), "ENGINES": json.dumps(engines), "DTYPE": json.dumps(dtype),
            "MAX_NEW": str(max_new), "DEADLINE_S": f"{deadline_h} * 3600", "BATCH": str(batch)}
    for kv in extra:  # KEY=VALUE, VALUE a Python literal ('"text"', 4096)
        k, v = kv.split("=", 1)
        sets[k] = v
    for k, v in sets.items():
        src, n = re.subn(rf"(?m)^{k} = .*$", lambda m: f"{k} = {v}", src, count=1)
        assert n == 1, k
    pack = base64.b64encode((HERE / "pack_quant.py").read_bytes()).decode()
    return src + f"\nPACK_SRC = {pack!r}\nSELF_SRC = {base64.b64encode(src.encode()).decode()!r}\n" \
        "if __name__ == '__main__':\n    main()\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("--pages", type=Path, required=True)
    ap.add_argument("--engines", nargs="+", default=["vllm"])
    ap.add_argument("--dtype", default="float32")
    ap.add_argument("--max-new", type=int, default=8192)
    ap.add_argument("--deadline-h", type=float, default=10.5)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--skip", type=Path, nargs="*", default=[])
    ap.add_argument("--set", action="append", default=[], help="KEY=LITERAL for any other CONFIG line")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    pages = a.pages.read_text().split()
    done = {p.stem for d in a.skip for p in d.glob("*.json")}
    pages = [p for p in pages if p not in done]
    print(f"{len(pages)} pages to run ({len(done)} skipped)")
    src = build(pages, a.engines, a.dtype, a.max_new, a.deadline_h, a.batch, a.set)
    # The base file ends in its own `if __name__` guard; the appended one runs main() after SELF_SRC exists.
    src = src.replace("\n\nif __name__ == \"__main__\":\n    main()\n", "\n", 1)
    out = HERE.parent / "jobs" / f"{a.slug}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(src)
    compile(src, str(out), "exec")
    if a.dry:
        print("wrote", out)
        return
    sys.exit(subprocess.call([sys.executable, str(HUB / "scripts/kaggle_job.py"), "push", a.slug, str(out),
                              "--accel", "gpu", "--internet"]))


if __name__ == "__main__":
    main()
