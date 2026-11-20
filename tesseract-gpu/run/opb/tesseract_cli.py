"""Stock Tesseract on each subset page's grayscale image, as the page's text for score.py.

    python3 run/opb/tesseract_cli.py [--workers 4]

Per page: `tesseract data/opb/gray/<page_id>.png - --psm 3 --oem 1 -l eng` (the apt tesseract 5.3.4,
eng.traineddata from tessdata_fast). Its plain-text output is already lines joined with newlines and
blocks separated by blank lines; the form feed Tesseract ends each page with is dropped. Writes
data/opb/out/tess_cli/<page_id>.md. An existing .md is kept, so a rerun resumes.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "opb"
OUT = DATA / "out" / "tess_cli"
CMD = ["tesseract", "{png}", "-", "--psm", "3", "--oem", "1", "-l", "eng"]


def one(page_id: str) -> str | None:
    dst = OUT / f"{page_id}.md"
    if dst.exists():
        return None
    png = DATA / "gray" / f"{page_id}.png"
    if not png.is_file():
        return f"{page_id}: no rendered page at {png}"
    cmd = [c.format(png=png) for c in CMD]
    env = dict(os.environ, OMP_THREAD_LIMIT="1")
    proc = subprocess.run(cmd, capture_output=True, env=env)
    if proc.returncode:
        return f"{page_id}: tesseract exited {proc.returncode}: {proc.stderr.decode(errors='replace')[-300:]}"
    text = proc.stdout.decode("utf-8").replace("\f", "").strip("\n") + "\n"
    tmp = dst.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.rename(dst)
    return None


def version() -> str:
    return subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    subset = json.loads((DATA / "subset.json").read_text())
    ids = [p["page_id"] for p in subset["pages"]]
    print(version(), flush=True)
    with ThreadPoolExecutor(a.workers) as ex:
        errs = [e for e in ex.map(one, ids) if e]
    print(f"{len(ids)} pages into {OUT}, {len(errs)} errors")
    for e in errs[:20]:
        print("  ", e)
    if errs:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
