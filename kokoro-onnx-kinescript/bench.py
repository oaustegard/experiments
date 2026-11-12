"""Speak requests.json with kinescript_kokoro.synth in one engine and record words, phones, times and memory.

    python bench.py ARM OUTDIR     (KOKORO_ENGINE and the model paths come from the environment)

Load time is measured apart from speaking: the pipeline for the voice's language is built first.
"""
import json
import resource
import sys
import time
from pathlib import Path

arm, out = sys.argv[1], Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
t0 = time.perf_counter()
import kinescript_kokoro as kk  # noqa: E402

reqs = json.loads(Path("requests.json").read_text())
kk._pipe(reqs[0]["speaker"][0])
load = time.perf_counter() - t0
lines = []
for r in reqs:
    t = time.perf_counter()
    res = kk.synth({**r, "out": str(out / f"{r['id']}.wav")})
    res.update(id=r["id"], synth_s=round(time.perf_counter() - t, 3))
    lines.append(res)
summary = {"arm": arm, "engine": kk._engine(), "python": sys.version.split()[0], "version": lines[0]["version"],
           "load_s": round(load, 3), "synth_s": round(sum(x["synth_s"] for x in lines), 3),
           "audio_s": round(sum(x["seconds"] for x in lines), 3),
           "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024), "lines": lines}
(out / "result.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
print(arm, {k: v for k, v in summary.items() if k != "lines"})
