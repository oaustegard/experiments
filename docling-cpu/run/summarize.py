"""Summarize the CPU arms: time per page, peak memory, pages that hit the token cap.

    python3 run/summarize.py /tmp/sweep-out results/summary.json

Reads OUT/<arm>/<page>.json written by sweep.py (llama-mtmd-cli per page: peak_rss_mb is that
process's ru_maxrss) or mineru_sweep.py (one llama-server: server_VmHWM_mb is the server's peak so
far, so its maximum is the arm's peak). A sweep.py page "hit the cap" when llama.cpp's decode log
says it generated max_new tokens. llama-mtmd-cli printed no such line in these runs, so: docling
counts as capped when its output does not end with </doctag>; the 4B when the page took over
450 s (see the comment below). MinerU sends its own anti-repetition settings per request and has
no single cap to hit.
"""
import json
import re
import statistics
import sys
from pathlib import Path


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q * (len(xs) - 1))))]


root, dst = Path(sys.argv[1]), Path(sys.argv[2])
summary = {}
for arm_dir in sorted(p for p in root.iterdir() if p.is_dir() and not p.name.endswith("smoke")):
    arm = json.loads((arm_dir / "arm.json").read_text())
    recs = [json.loads(p.read_text()) for p in sorted(arm_dir.glob("*.json")) if p.name != "arm.json"]
    if not recs:
        continue
    walls = [r["wall_s"] for r in recs]
    if "server_VmHWM_mb" in recs[0]:
        peak = max(r["server_VmHWM_mb"] for r in recs)
        mem_note = "llama-server VmHWM over the whole arm"
        capped = []
        errors = [r["page_id"] for r in recs if r.get("error")]
    else:
        peak = max(r["peak_rss_mb"] for r in recs)
        mem_note = "max per-page ru_maxrss of llama-mtmd-cli"
        cap = arm.get("max_new", 8192)
        capped = []
        for r in recs:
            t = r["text"].strip()
            n = None
            for ln in r.get("timing", []):
                m = re.search(r"(\d+)\s+(?:runs|tokens)", ln)
                if "eval time" in ln and "prompt" not in ln and m:
                    n = int(m.group(1))
            if n is not None:
                hit = n >= cap - 1
            elif "docling" in arm_dir.name:
                hit = not t.endswith("</doctag>")
            else:
                # 4B on 4 cores: ~34 s fixed + ~7-9 tokens/s, so 4096 tokens take ~490-620 s.
                # A page past 450 s generated at least ~3,300 tokens: counted as a cap/loop page.
                hit = r["wall_s"] > 450
            if hit:
                capped.append(r["page_id"])
        errors = [r["page_id"] for r in recs if r.get("status", 0) != 0]
    summary[arm_dir.name] = {
        "pages": len(recs), "median_s": round(statistics.median(walls), 1),
        "p90_s": round(pct(walls, 0.9), 1), "max_s": round(max(walls), 1),
        "total_min": round(sum(walls) / 60, 1),
        "peak_mem_mb": peak, "mem_note": mem_note,
        "capped_pages": capped, "error_pages": errors,
        "model": arm.get("model"), "mmproj": arm.get("mmproj"),
        "median_s_uncapped": round(statistics.median([r["wall_s"] for r in recs
                                                      if r["page_id"] not in capped] or [0]), 1),
    }
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps(summary, indent=1))
for k, v in summary.items():
    print(k, {x: v[x] for x in ("pages", "median_s", "median_s_uncapped", "p90_s", "total_min", "peak_mem_mb")},
          "capped", len(v["capped_pages"]), "errors", len(v["error_pages"]))
