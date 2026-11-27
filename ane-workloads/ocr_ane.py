#!/usr/bin/env python3
"""Apple Vision OCR on Omni Parse Bench pages, GPU idle and GPU busy.

Two arms, on the pages and with the grader this repo's Indra, OCRia and Nemotron Parse
experiments used (datalab-to/omni_parse_bench, scorer omni-parse-bench 1.8.0, through
ocria-pack/run/grade.py):

  text   VNRecognizeTextRequest, recognition level accurate, through pyobjc: text lines
         with boxes. This script groups the lines into paragraphs; no tables.
  docs   RecognizeDocumentsRequest (macOS 26; a Swift-only API, so docs_ocr.swift is
         compiled with swiftc and driven over a pipe): paragraphs, tables, lists, a title.

Every page is rasterised first (PDF pages through pdfium at 200 DPI, as
indra-browser/run/prerender.py does, but in colour), so the timed part is Vision alone:
ms/page is the request's perform call on a decoded image, in a warm process.

The whole page set runs once with the GPU idle (that run is graded). GPU busy re-reads
--busy-pages of the same pages under a llama-bench decode and is compared with those
pages' idle times.

    python3 ocr_ane.py                         # all 2,937 pages, both arms, about 1.5 GB of pages
    python3 ocr_ane.py --pages ../nemotron-parse-pack/run/sample250.txt   # the 250-page sample
    python3 ocr_ane.py --sample 600            # opb_eval.py's seeded sample
    python3 ocr_ane.py --grade                 # also score here (pip-installs the benchmark's scorer)
    python3 ocr_ane.py --readapt results/<run>/ocr   # rebuild page JSONs from saved raw output

Raw Vision output is kept per page (results/<run>/ocr/<arm>/raw.jsonl.gz), and the page
JSONs the grader reads are derived from it by `adapt_text` / `adapt_docs` below, which need
no Mac: --readapt reruns them after a change. Grading without --grade: commit the results
directory and run ocria-pack/run/grade.py on <arm>/ in a hub session (inside scripts/jail.sh).

Requires: python3 >= 3.11, Xcode command line tools (swiftc) for the docs arm.
"""
from __future__ import annotations

import argparse
import gzip
import html
import json
import random
import subprocess
import sys
import time
from pathlib import Path

import common
from common import log, run

PINS = ["pyobjc-framework-Vision==13.1", "pyobjc-framework-Quartz==13.1",
        "pyobjc-framework-CoreML==13.1", "pypdfium2==5.15.2", "pillow", "pyarrow",
        "huggingface_hub>=0.30"]
GRADER_PINS = ["omni-parse-bench[benchmark]==1.8.0", "pyarrow"]  # extras: the scorer imports httpx, openai, markdown-it
DATASET = "datalab-to/omni_parse_bench"
DATASET_REV = "c7afc687911ff903eb36ca2daa450008629679f1"  # 2,937 pages
DPI = 200.0
MAX_PAGE_PIXELS = 100e6


# ---------------------------------------------------------------- adapters (no Mac needed)

def _union(boxes):
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[0] + b[2] for b in boxes)
    y1 = max(b[1] + b[3] for b in boxes)
    return [x0, y0, x1 - x0, y1 - y0]


def _block(kind, bbox, text):
    return {"kind": kind, "text": text,
            "bbox": {"x": bbox[0], "y": bbox[1], "width": bbox[2], "height": bbox[3]}}


def group_lines(lines):
    """Vision's line order, cut into paragraphs. A new paragraph starts when the next line
    sits more than 0.7 of a line height below the previous one, starts above it (a new
    column), or does not overlap it horizontally."""
    paras, cur = [], []
    for ln in lines:
        if not ln["text"].strip():
            continue
        if cur:
            p = cur[-1]["bbox"]
            b = ln["bbox"]
            h = min(p[3], b[3]) or 1.0
            gap = b[1] - (p[1] + p[3])
            overlap = min(p[0] + p[2], b[0] + b[2]) - max(p[0], b[0])
            if gap > 0.7 * h or b[1] + b[3] < p[1] + 0.5 * p[3] or overlap <= 0:
                paras.append(cur)
                cur = []
        cur.append(ln)
    if cur:
        paras.append(cur)
    return paras


def join_lines(texts):
    """Lines of one paragraph as running text; a line-end hyphen after a letter is joined."""
    out = ""
    for t in texts:
        t = t.strip()
        if out.endswith("-") and len(out) > 1 and out[-2].isalpha() and t[:1].islower():
            out = out[:-1] + t
        else:
            out = f"{out} {t}" if out else t
    return out


def adapt_text(raw):
    """VNRecognizeTextRequest lines -> the page JSON ocria-pack/run/grade.py reads."""
    parts, blocks = [], []
    for para in group_lines(raw.get("lines") or []):
        text = join_lines([ln["text"] for ln in para])
        parts.append(text)
        blocks.append(_block("text", _union([ln["bbox"] for ln in para]), text))
    return {"page_id": raw["page_id"], "markdown": "\n\n".join(parts), "width": raw["width"],
            "height": raw["height"], "blocks": blocks, "ms": raw.get("ms")}


def table_cells(t):
    """A table's cells once each: a cell that spans rows is listed in every row it spans."""
    seen, out = set(), []
    for c in t.get("cells") or []:
        key = (c["r0"], c["c0"])
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def table_html(t):
    """Cells with row and column ranges -> an HTML table with rowspan and colspan."""
    rows = {}
    for c in table_cells(t):
        rows.setdefault(c["r0"], []).append(c)
    out = ["<table>"]
    for r in sorted(rows):
        tds = []
        for c in sorted(rows[r], key=lambda c: c["c0"]):
            span = ""
            if c["r1"] > c["r0"]:
                span += f' rowspan="{c["r1"] - c["r0"] + 1}"'
            if c["c1"] > c["c0"]:
                span += f' colspan="{c["c1"] - c["c0"] + 1}"'
            tds.append(f"<td{span}>{html.escape(' '.join(c['text'].split()))}</td>")
        out.append("<tr>" + "".join(tds) + "</tr>")
    out.append("</table>")
    return "\n".join(out)


def _inside(b, region):
    cx, cy = b[0] + b[2] / 2, b[1] + b[3] / 2
    return region[0] <= cx <= region[0] + region[2] and region[1] <= cy <= region[1] + region[3]


def adapt_docs(raw):
    """RecognizeDocumentsRequest output -> page JSON. Paragraphs keep Vision's order; a
    paragraph inside a table or list region is dropped (the table or list carries its text);
    each table and list is placed before the first remaining paragraph that starts below it."""
    parts, blocks = [], []
    for doc in raw.get("documents") or []:
        regions = [("table", t["bbox"], t) for t in doc.get("tables") or []]
        regions += [("list", l["bbox"], l) for l in doc.get("lists") or []]
        regions.sort(key=lambda r: r[1][1])
        title = " ".join((doc.get("title") or "").split())
        items = []
        for p in doc.get("paragraphs") or []:
            if any(_inside(p["bbox"], r[1]) for r in regions):
                continue
            text = join_lines([ln["text"] for ln in p.get("lines") or []]) or " ".join(p["text"].split())
            if not text:
                continue
            is_title = (title and " ".join(text.split()) == title) or (
                p.get("lines") and all(ln.get("title") for ln in p["lines"]))
            items.append(("heading" if is_title else "text", p["bbox"], text))
        merged = []
        for it in items:
            while regions and regions[0][1][1] < it[1][1]:
                merged.append(regions.pop(0))
            merged.append(it)
        merged += regions
        for kind, bbox, x in merged:
            if kind == "table":
                parts.append(table_html(x))
                text = " ".join(" ".join(c["text"].split()) for c in table_cells(x))
            elif kind == "list":
                rows = [f"- {' '.join(i['text'].split())}" for i in x.get("items") or []]
                parts.append("\n".join(rows))
                text = " ".join(r[2:] for r in rows)
            elif kind == "heading":
                parts.append(f"# {x}")
                text = x
            else:
                parts.append(x)
                text = x
            blocks.append(_block(kind, bbox, text))
    return {"page_id": raw["page_id"], "markdown": "\n\n".join(parts), "width": raw["width"],
            "height": raw["height"], "blocks": blocks, "ms": raw.get("ms")}


ADAPT = {"text": adapt_text, "docs": adapt_docs}


def write_pages(arm_dir, arm, raws):
    pages = arm_dir / "pages"
    pages.mkdir(parents=True, exist_ok=True)
    n = 0
    for raw in raws:
        rec = {"page_id": raw["page_id"], "error": raw["error"]} if raw.get("error") else ADAPT[arm](raw)
        (pages / f"{raw['page_id']}.json").write_text(json.dumps(rec, ensure_ascii=False))
        n += 1
    return n


def read_raw(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f]


def readapt(run_dir):
    for arm in ADAPT:
        f = run_dir / arm / "raw.jsonl.gz"
        if f.exists():
            print(f"{arm}: {write_pages(run_dir / arm, arm, read_raw(f))} pages -> {run_dir / arm / 'pages'}")


# ---------------------------------------------------------------- pages

def dataset(args, with_gold):
    from huggingface_hub import snapshot_download

    ds = common.WORK / "ocr" / "ds"
    pats = ["manifest.parquet", "pages/*"] + (["gold/*", "results.parquet"] if with_gold else [])
    log("fetching Omni Parse Bench pages (about 1.5 GB the first time)")
    snapshot_download(DATASET, repo_type="dataset", revision=DATASET_REV, local_dir=str(ds),
                      allow_patterns=pats)
    return ds


def select_rows(args, ds):
    import pyarrow.parquet as pq

    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    if args.pages:
        want = set(args.pages.read_text().split())
        rows = [r for r in rows if r["page_id"] in want]
    elif args.sample:
        rows = random.Random(0).sample(rows, args.sample)  # indra-browser/run/opb_eval.py's sample
    return rows


def rasterise(rows, ds):
    """{page_id: image path}. PDFs through pdfium at 200 DPI (capped at 100 MP) to RGB PNG."""
    import pypdfium2 as pdfium

    png = common.WORK / "ocr" / "png"
    png.mkdir(parents=True, exist_ok=True)
    out, errs = {}, {}
    for i, r in enumerate(rows):
        src = ds / r["doc_path"]
        try:
            with open(src, "rb") as f:
                is_pdf = b"%PDF-" in f.read(1024)
            if not is_pdf:
                out[r["page_id"]] = src
                continue
            dst = png / f"{r['page_id']}.png"
            if not dst.exists():
                doc = pdfium.PdfDocument(str(src))
                try:
                    page = doc[0]
                    w, h = page.get_size()
                    k = min(DPI / 72.0, (MAX_PAGE_PIXELS / max(w * h, 1.0)) ** 0.5)
                    tmp = dst.with_suffix(".tmp.png")
                    page.render(scale=k).to_pil().convert("RGB").save(tmp, compress_level=3)
                    tmp.rename(dst)
                finally:
                    doc.close()
            out[r["page_id"]] = dst
        except Exception as e:  # noqa: BLE001 - an unreadable page fails its tests, not the run
            errs[r["page_id"]] = f"{type(e).__name__}: {e}"
        if (i + 1) % 500 == 0:
            log(f"rasterised {i + 1}/{len(rows)}")
    return out, errs


# ---------------------------------------------------------------- arm: VNRecognizeTextRequest

class TextArm:
    name = "text"

    def setup(self, args):
        import Quartz
        import Vision

        self.V, self.Q = Vision, Quartz
        req = Vision.VNRecognizeTextRequest.alloc().init()
        req.setRecognitionLevel_(Vision.VNRequestTextRecognitionLevelAccurate)
        req.setUsesLanguageCorrection_(True)
        req.setAutomaticallyDetectsLanguage_(True)
        info = {"request": "VNRecognizeTextRequest", "level": "accurate",
                "revision": int(req.revision()), "language_correction": True,
                "automatically_detects_language": True, "asked_device": args.device}
        try:  # compute devices per stage (macOS 14+)
            sup, err = req.supportedComputeStageDevicesAndReturnError_(None)
            info["supported_devices"] = {str(k): [type(d).__name__ for d in v] for k, v in (sup or {}).items()}
            if args.device != "auto":
                key = {"ne": "NeuralEngine", "gpu": "GPU", "cpu": "CPU"}[args.device]
                for stage, devs in sup.items():
                    for d in devs:
                        if key in type(d).__name__:
                            req.setComputeDevice_forComputeStage_(d, stage)
            info["assigned_devices"] = {
                str(k): (type(req.computeDeviceForComputeStage_(k)).__name__
                         if req.computeDeviceForComputeStage_(k) is not None else "auto")
                for k in (sup or {})}
        except Exception as e:  # noqa: BLE001
            info["device_error"] = repr(e)
        self.req = req
        return info

    def read(self, page_id, path):
        import objc
        from Foundation import NSURL

        V, Q = self.V, self.Q
        raw = {"page_id": page_id}
        with objc.autorelease_pool():
            t0 = time.perf_counter()
            src = Q.CGImageSourceCreateWithURL(NSURL.fileURLWithPath_(str(path)), None)
            img = Q.CGImageSourceCreateImageAtIndex(src, 0, None) if src is not None else None
            if img is None:
                return raw | {"error": "unreadable image"}
            W, H = Q.CGImageGetWidth(img), Q.CGImageGetHeight(img)
            t1 = time.perf_counter()
            handler = V.VNImageRequestHandler.alloc().initWithCGImage_options_(img, None)
            ok, err = handler.performRequests_error_([self.req], None)
            t2 = time.perf_counter()
            if not ok:
                return raw | {"error": str(err), "width": W, "height": H}
            lines = []
            for obs in self.req.results() or []:
                cands = obs.topCandidates_(1)
                if not cands:
                    continue
                bb = obs.boundingBox()  # normalised, origin bottom-left
                x, y, w, h = bb.origin.x, bb.origin.y, bb.size.width, bb.size.height
                lines.append({"text": str(cands[0].string()), "conf": round(float(cands[0].confidence()), 3),
                              "bbox": [round(x * W, 1), round((1 - y - h) * H, 1), round(w * W, 1), round(h * H, 1)]})
        return raw | {"width": W, "height": H, "lines": lines,
                      "ms_decode": round(1e3 * (t1 - t0), 2), "ms": round(1e3 * (t2 - t1), 2)}

    def close(self):
        pass


# ---------------------------------------------------------------- arm: RecognizeDocumentsRequest

class DocsArm:
    name = "docs"

    def setup(self, args):
        src = common.HERE / "docs_ocr.swift"
        self.bin = common.WORK / "ocr" / "docs_ocr"
        if not self.bin.exists() or self.bin.stat().st_mtime < src.stat().st_mtime:
            log("compiling docs_ocr.swift")
            self.bin.parent.mkdir(parents=True, exist_ok=True)
            rc, out, err = run(["swiftc", "-O", "-parse-as-library", src, "-o", self.bin], timeout=900)
            if rc != 0:
                raise RuntimeError("swiftc failed:\n" + (err or out)[-4000:])
        self.cmd = [str(self.bin), "--device", args.device]
        self._open()
        return {"request": "RecognizeDocumentsRequest", "swiftc": common.sh(["swiftc", "--version"]),
                **{k: v for k, v in self.header.items() if k != "header"}}

    def _open(self):
        self.proc = subprocess.Popen(self.cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
        self.header = json.loads(self.proc.stdout.readline() or "{}")
        if self.header.get("error"):
            raise RuntimeError(self.header["error"])

    def read(self, page_id, path):
        if self.proc.poll() is not None:  # the helper died on the previous page
            self._open()
        try:
            self.proc.stdin.write(f"{path}\n")
            self.proc.stdin.flush()
            line = self.proc.stdout.readline()
        except (BrokenPipeError, OSError) as e:
            return {"page_id": page_id, "error": f"helper: {e!r}"}
        if not line:
            return {"page_id": page_id, "error": f"helper exited rc={self.proc.poll()}"}
        d = json.loads(line)
        d.pop("path", None)
        return {"page_id": page_id} | d

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=30)
        except Exception:  # noqa: BLE001
            self.proc.kill()


ARMS = {"text": TextArm, "docs": DocsArm}


# ---------------------------------------------------------------- run

def stats(ms):
    ms = sorted(ms)
    if not ms:
        return {}
    return {"pages": len(ms), "ms_median": round(common.median(ms), 1),
            "ms_mean": round(sum(ms) / len(ms), 1), "ms_p90": round(ms[int(0.9 * (len(ms) - 1))], 1),
            "pages_per_s": round(1e3 * len(ms) / sum(ms), 2)}


def text_of(raw):
    if "lines" in raw:
        return "\n".join(l["text"] for l in raw["lines"])
    return json.dumps(raw.get("documents"), sort_keys=True)


def grade(args, ds, out, arms):
    py = common.ensure_venv("opb-grader", GRADER_PINS)
    grader = common.experiment("ocria-pack") / "run" / "grade.py"
    cmd = [py, grader, "--ds", ds, "--results", ds / "results.parquet",
           "--systems", args.systems, "--out", out / "opb.json"]
    for a in arms:
        cmd += ["--run", out / a]
    log("grading with omni-parse-bench (ocria-pack/run/grade.py)")
    rc, o, err = run(cmd, timeout=7200)
    (out / "raw" / "grade.log").write_text(f"rc={rc}\n{o}\n--- stderr\n{err[-8000:]}")
    if rc != 0 or not (out / "opb.json").exists():
        return {"error": (err or o)[-800:]}
    return json.loads((out / "opb.json").read_text())


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--only", choices=list(ARMS), action="append")
    ap.add_argument("--pages", type=Path, help="file of page ids, one per line")
    ap.add_argument("--sample", type=int, default=0, help="a fixed random sample of N pages (seed 0)")
    ap.add_argument("--device", choices=["auto", "ne", "gpu", "cpu"], default="auto",
                    help="compute device asked of Vision for every stage that offers it")
    ap.add_argument("--busy-pages", type=int, default=150, help="pages re-read with the GPU busy")
    ap.add_argument("--grade", action="store_true", help="score here with the benchmark's scorer")
    ap.add_argument("--systems", default="Indra OCR,Tesseract",
                    help="published rows to pair against when grading")
    ap.add_argument("--readapt", type=Path, help="rebuild page JSONs from a run's raw output and exit")
    common.add_gpu_args(ap)
    args = ap.parse_args()
    if args.readapt:
        return readapt(args.readapt)
    if not common.is_mac():
        sys.exit("This script needs Apple Silicon macOS (--readapt runs anywhere).")
    if sys.version_info < (3, 11):
        sys.exit("python3 >= 3.11 needed (pyobjc 13); try `brew install python@3.12`")
    common.reexec_in_venv("ocr", PINS)

    env, out = common.start("ocr", args)
    S = {"env": env, "args": {k: str(v) for k, v in vars(args).items()},
         "dataset": {"repo": DATASET, "revision": DATASET_REV}, "arms": {}}
    ds = dataset(args, args.grade)
    rows = select_rows(args, ds)
    images, raster_errs = rasterise(rows, ds)
    S["pages"] = {"selected": len(rows), "rasterise_errors": raster_errs}
    ids = [r["page_id"] for r in rows]
    busy_ids = [i for i in common_pick(ids, args.busy_pages) if i in images]
    log(f"{len(ids)} pages, {len(busy_ids)} of them again with the GPU busy")

    arms = {}
    for name in args.only or list(ARMS):
        a = ARMS[name]()
        try:
            S["arms"][name] = {"setup": a.setup(args)}
            arms[name] = a
        except Exception as e:  # noqa: BLE001 - e.g. the Swift helper does not compile
            S["arms"][name] = {"error": repr(e)}
            log(f"{name} arm unavailable: {e!r}")
        (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))

    idle_raw = {}
    for cond, load in common.gpu_conditions(args, out / "raw"):
        if isinstance(load, Exception):
            S["busy_error"] = repr(load)
            log(f"GPU-busy condition skipped: {load!r}")
            continue
        for name, a in arms.items():
            todo = ids if cond == "idle" else busy_ids
            log(f"{name}, GPU {cond}: {len(todo)} pages")
            if cond == "idle":  # the first request loads Vision's models: keep it out of the timings
                first = a.read(todo[0], images[todo[0]]) if todo[0] in images else {}
                S["arms"][name]["first_request_ms"] = first.get("ms")
            raws, t0 = [], time.time()
            for i, pid in enumerate(todo):
                if pid not in images:
                    raws.append({"page_id": pid, "error": "rasterise: " + raster_errs.get(pid, "?")})
                    continue
                if load is not None and i % 25 == 0:
                    load.wait_busy()
                raws.append(a.read(pid, images[pid]))
                if (i + 1) % 250 == 0:
                    log(f"  {i + 1}/{len(todo)} {time.time() - t0:.0f}s")
            t1 = time.time()
            ok = [r for r in raws if "error" not in r]
            R = stats([r["ms"] for r in ok]) | {"errors": len(raws) - len(ok), "wall_s": round(t1 - t0, 1)}
            if cond == "idle":
                idle_raw[name] = {r["page_id"]: r for r in raws}
                (out / name).mkdir(exist_ok=True)
                with gzip.open(out / name / "raw.jsonl.gz", "wt") as f:
                    for r in raws:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
                write_pages(out / name, name, raws)
            else:
                idle = idle_raw.get(name, {})
                same = [idle[r["page_id"]] for r in ok if "ms" in idle.get(r["page_id"], {})]
                R["idle_same_pages"] = stats([r["ms"] for r in same])
                R["same_output_as_idle"] = sum(text_of(r) == text_of(idle[r["page_id"]])
                                               for r in ok if r["page_id"] in idle)
                R.update(common.gpu_record(load, t0, t1))
            S["arms"][name][cond] = R
            (out / "summary.json").write_text(json.dumps(S, indent=1, default=str))
    for a in arms.values():
        a.close()
    if args.grade and arms:
        S["grade"] = grade(args, ds, out, list(arms))
    common.finish(out, S, summarize(S))


def common_pick(ids, n):
    """n ids spread evenly over the list (all of them if n >= len)."""
    if n >= len(ids):
        return list(ids)
    return [ids[(2 * k + 1) * len(ids) // (2 * n)] for k in range(n)]


def summarize(S):
    e = S["env"]
    L = [f"# Apple Vision OCR on Omni Parse Bench: {e.get('chip')}, {e['timestamp']}", "",
         f"macOS {e.get('macos')}, AC {e.get('on_ac_power')}, low power {e.get('low_power_mode')}. "
         f"{S['pages']['selected']} pages ({len(S['pages']['rasterise_errors'])} could not be rasterised).", "",
         "| arm | GPU | pages | ms/page median | mean | p90 | pages/s | idle median, same pages | "
         "errors | GPU tok/s alone -> during |", "|---|---|---|---|---|---|---|---|---|---|"]
    for name, A in S["arms"].items():
        if "error" in A:
            L.append(f"| {name} | | unavailable: {A['error'][:200]} | | | | | | | |")
            continue
        for cond in ("idle", "busy"):
            r = A.get(cond)
            if not r:
                continue
            g = (f"{r.get('gpu_alone_tok_s')} -> {r.get('gpu_during_tok_s')} (busy {r.get('gpu_busy_fraction')})"
                 if cond == "busy" else "")
            L.append(f"| {name} | {cond} | {r.get('pages')} | {r.get('ms_median')} | {r.get('ms_mean')} | "
                     f"{r.get('ms_p90')} | {r.get('pages_per_s')} | "
                     f"{(r.get('idle_same_pages') or {}).get('ms_median', '')} | {r.get('errors')} | {g} |")
    L.append("")
    for name, A in S["arms"].items():
        if A.get("setup"):
            L.append(f"- {name}: {json.dumps(A['setup'])}; first request {A.get('first_request_ms')} ms")
    G = S.get("grade")
    if G and "error" not in G:
        L += ["", f"Omni Parse Bench, {G['pages']} pages (headline = mean of the families):", "",
              "| system | headline | " + " | ".join(next(iter(G["summaries"].values()))["families"]) + " |",
              "|---|---|" + "---|" * len(next(iter(G["summaries"].values()))["families"])]
        for name, s in G["summaries"].items():
            L.append(f"| {name} | {100 * s['score']:.2f} | " +
                     " | ".join(f"{100 * v['score']:.1f}" for v in s["families"].values()) + " |")
    elif G:
        L += ["", f"Grading failed: {G['error'][:400]}"]
    else:
        L += ["", "Not graded here: run ocria-pack/run/grade.py on the arm folders (see README)."]
    if S.get("busy_error"):
        L += ["", f"GPU-busy condition skipped: {S['busy_error']}"]
    return "\n".join(L)


if __name__ == "__main__":
    main()
