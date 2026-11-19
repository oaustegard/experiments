"""Run granite-docling-258M with llama.cpp (llama-mtmd-cli) over a page sample, one arm at a time.

Runs INSIDE scripts/jail.sh (llama.cpp is code nobody here wrote). Each page is one llama-mtmd-cli
process: greedy decode, "Convert this page to docling.", up to 8192 new tokens, 4 threads. Per page
it writes OUT/<arm>/<page>.json with the DocTags text, wall time, peak RSS (os.wait4 ru_maxrss of
that process) and llama.cpp's own timing lines. A page already written is skipped (checkpoint), so a
killed run resumes where it stopped.

    sh scripts/jail.sh -C /tmp/jail -- python3 sweep.py --arm q8 \
        --model models/q8.gguf --mmproj models/mmproj-model-f16.gguf [--kv q8_0]
"""
import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--mmproj", required=True)
ap.add_argument("--kv", default="f16", help="KV cache type for K and V (f16, q8_0, ...)")
ap.add_argument("--pages", default="opb/pages.txt")
ap.add_argument("--img", default="opb/img")
ap.add_argument("--out", default="out")
ap.add_argument("--bin", default="llama.cpp/build/bin/llama-mtmd-cli")
ap.add_argument("--max-new", type=int, default=8192)
ap.add_argument("--ctx", type=int, default=16384)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--prompt", default="Convert this page to docling.")
a = ap.parse_args()

out = Path(a.out) / a.arm
out.mkdir(parents=True, exist_ok=True)
pages = Path(a.pages).read_text().split()
if a.limit:
    pages = pages[: a.limit]
(out / "arm.json").write_text(json.dumps(vars(a), indent=1))
TIMING = re.compile(r"(load time|prompt eval|eval time|total time|encod|decod|n_tokens|image slice|batch)", re.I)
for i, pid in enumerate(pages):
    dst = out / f"{pid}.json"
    if dst.exists():
        continue
    cmd = [a.bin, "-m", a.model, "--mmproj", a.mmproj, "--image", f"{a.img}/{pid}.png",
           "-p", a.prompt, "--temp", "0", "-n", str(a.max_new),
           "-t", str(a.threads), "-c", str(a.ctx), "-ctk", a.kv, "-ctv", a.kv]
    t0 = time.time()
    with open(out / "_stdout", "w") as so, open(out / "_stderr", "w") as se:
        p = subprocess.Popen(cmd, stdout=so, stderr=se)
        _, status, ru = os.wait4(p.pid, 0)
    wall = time.time() - t0
    err = (out / "_stderr").read_text(errors="replace")
    rec = {"page_id": pid, "arm": a.arm, "status": os.waitstatus_to_exitcode(status),
           "wall_s": round(wall, 2), "peak_rss_mb": round(ru.ru_maxrss / 1024, 1),
           "text": (out / "_stdout").read_text(errors="replace"),
           "timing": [ln for ln in err.splitlines() if TIMING.search(ln)][-20:]}
    if rec["status"] != 0:
        rec["stderr_tail"] = err[-3000:]
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, ensure_ascii=False))
    tmp.rename(dst)
    print(f"{i+1}/{len(pages)} {pid} {wall:.1f}s rss {rec['peak_rss_mb']} MB status {rec['status']}", flush=True)
