"""MinerU2.5-Pro through llama.cpp's llama-server and MinerU's own two-step client, page by page.

Runs INSIDE scripts/jail.sh (llama.cpp and mineru-vl-utils are code nobody here wrote); the jail
has only loopback, which is all the server and client need. One llama-server process serves every
page (4 threads, one slot); mineru_vl_utils.MinerUClient(backend="http-client") runs layout
detection, then one request per block crop, then its own post-processing, and json2md makes the
Markdown. Per page it writes OUT/<arm>/<page>.json with the Markdown, the blocks, wall time and the
server's VmRSS / VmHWM after the page (VmHWM is the server's peak so far). A page already written
is skipped, so a killed run resumes.

    sh scripts/jail.sh -C /tmp/jail -e PYTHONPATH=/tmp/jail/mineru/.deps -- python3 mineru_sweep.py \
        --arm mineru-q8 --model models/mineru/X.Q8_0.gguf --mmproj models/mineru/X.mmproj-Q8_0.gguf
"""
import argparse
import json
import subprocess
import time
import urllib.request
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--arm", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--mmproj", required=True)
ap.add_argument("--pages", default="opb/pages.txt")
ap.add_argument("--img", default="opb/img")
ap.add_argument("--out", default="out")
ap.add_argument("--bin", default="llama.cpp/build/bin/llama-server")
ap.add_argument("--ctx", type=int, default=8192)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--port", type=int, default=8090)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--server-args", default="", help="extra llama-server arguments, space-separated")
a = ap.parse_args()

out = Path(a.out) / a.arm
out.mkdir(parents=True, exist_ok=True)
pages = Path(a.pages).read_text().split()[: a.limit or None]
(out / "arm.json").write_text(json.dumps(vars(a), indent=1))
todo = [p for p in pages if not (out / f"{p}.json").exists()]
print(f"{len(pages)} pages, {len(todo)} to run", flush=True)
if not todo:
    raise SystemExit(0)


def mem(pid):
    d = {}
    for ln in Path(f"/proc/{pid}/status").read_text().splitlines():
        k, _, v = ln.partition(":")
        if k in ("VmRSS", "VmHWM"):
            d[k] = round(int(v.split()[0]) / 1024, 1)
    return d


log = open(out / "_server.log", "a")
srv = subprocess.Popen([a.bin, "-m", a.model, "--mmproj", a.mmproj, "--host", "127.0.0.1",
                        "--port", str(a.port), "-t", str(a.threads), "-c", str(a.ctx), "-np", "1",
                        "--no-webui",
                        # MinerU's layout parser reads <|box_start|>...<|ref_start|> tokens; llama-server
                        # drops special tokens unless --special (it ignores skip_special_tokens=false).
                        # Not --image-min-tokens 1024 (llama.cpp's Qwen-VL grounding hint): it inflates
                        # every block crop to 1024 tokens, ~80 s of prefill each on 2 threads; the
                        # full-page layout request already exceeds 1024 tokens.
                        "--special", *a.server_args.split()], stdout=log, stderr=log)
url = f"http://127.0.0.1:{a.port}"
t0 = time.time()
while True:
    try:
        if urllib.request.urlopen(url + "/health", timeout=2).status == 200:
            break
    except Exception:  # noqa: BLE001 - not up yet
        pass
    if srv.poll() is not None or time.time() - t0 > 300:
        raise SystemExit(f"server did not start (exit {srv.poll()}), see {out / '_server.log'}")
    time.sleep(1)
print(f"server up in {time.time() - t0:.1f}s, {mem(srv.pid)}", flush=True)

from PIL import Image  # noqa: E402
from mineru_vl_utils import MinerUClient  # noqa: E402
from mineru_vl_utils.post_process.json2markdown import json2md  # noqa: E402

client = MinerUClient(backend="http-client", server_url=url, http_timeout=3600, max_concurrency=1)
try:
    for i, pid in enumerate(todo):
        img = Image.open(f"{a.img}/{pid}.png").convert("RGB")
        t = time.time()
        rec = {"page_id": pid, "arm": a.arm}
        try:
            blocks = client.two_step_extract(img)
            rec["blocks"] = [dict(b) for b in blocks]
            rec["text"] = json2md(rec["blocks"])
        except Exception as e:  # noqa: BLE001 - a failed page is recorded, not fatal
            rec["error"] = f"{type(e).__name__}: {e}"
            rec["text"] = ""
        rec["wall_s"] = round(time.time() - t, 2)
        rec.update({f"server_{k}_mb": v for k, v in mem(srv.pid).items()})
        dst = out / f"{pid}.json"
        tmp = dst.with_suffix(".tmp")
        tmp.write_text(json.dumps(rec, ensure_ascii=False, default=str))
        tmp.rename(dst)
        print(f"{i + 1}/{len(todo)} {pid} {rec['wall_s']}s hwm {rec.get('server_VmHWM_mb')} MB"
              f"{' ERROR ' + rec['error'][:200] if 'error' in rec else ''}", flush=True)
finally:
    srv.terminate()
    srv.wait(30)
