"""Grade LiteParse + OCRia-nano-1 runs on OmniParseBench with the benchmark's own scorer, and compare them
with the published rows (Indra OCR, Tesseract, ...) on the same pages.

Runs INSIDE scripts/jail.sh: it imports omni_parse_bench, code nobody here wrote.

A run folder holds pages/<page_id>.json from run/lp_pages.mjs. Each page becomes the prediction a
benchmark adapter would make: html is LiteParse's markdown through the benchmark's markdown.to_html,
blocks are LiteParse's layout blocks (kinds mapped onto the benchmark's labels, boxes divided by the page
size). A page that errored, or is missing, fails every test. A published system's verdicts come from
results.parquet and go through the same page-line code, so both sides are summarized and paired
(uncertainty.paired: 95% bootstrap over documents) the same way.

    python3 run/grade.py --ds DS --results results.parquet --run out/ARM [--run out/ARM2 ...] \
        --systems "Indra OCR,Tesseract" --out results/opb_ARM.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# LiteParse block kinds -> views.LABELS. A rule (a drawn line) carries no text and is dropped.
KINDS = {"heading": "heading", "paragraph": "text", "list_item": "list", "code": "code",
         "table": "table", "grid_fallback": "table", "figure": "figure"}


def prediction(rec: dict, wants: tuple[str, ...]) -> dict:
    from omni_parse_bench import markdown

    out = {}
    if "html" in wants:
        out["html"] = markdown.to_html(rec.get("markdown") or rec.get("text") or "")
    if "blocks" in wants:
        W, H = float(rec["width"]), float(rec["height"])
        blocks = []
        for b in rec.get("blocks") or []:
            lab, bb = KINDS.get(b["kind"]), b.get("bbox")
            if lab is None or not bb:
                continue
            x0 = min(max(bb["x"] / W, 0.0), 1.0)
            y0 = min(max(bb["y"] / H, 0.0), 1.0)
            x1 = min(max((bb["x"] + bb["width"]) / W, 0.0), 1.0)
            y1 = min(max((bb["y"] + bb["height"]) / H, 0.0), 1.0)
            if x1 <= x0 or y1 <= y0:
                continue
            text = b.get("text")
            if text is None and b.get("lines"):
                text = "\n".join(b["lines"])
            if text is None and b.get("rows"):
                text = "\n".join(" ".join(r) for r in ([b["header"]] if b.get("header") else []) + b["rows"])
            blocks.append({"label": lab, "bbox": [x0, y0, x1, y1], "text": " ".join((text or "").split())})
        out["blocks"] = blocks
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--run", type=Path, action="append", default=[])
    ap.add_argument("--systems", default="Indra OCR,Tesseract")
    ap.add_argument("--pages", type=Path, help="restrict to these page ids (default: the pages the first run has)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    import pyarrow.parquet as pq
    from omni_parse_bench import data, uncertainty
    from omni_parse_bench.scorers import unit_tests
    from omni_parse_bench.scorers.unit_tests import compare

    rows = pq.read_table(a.ds / "manifest.parquet").to_pylist()
    if a.pages:
        want = set(a.pages.read_text().split())
    else:
        want = {p.stem for p in (a.run[0] / "pages").glob("*.json")}
    rows = [r for r in rows if r["page_id"] in want]
    pages = {r["page_id"]: data.Page(r["page_id"], r["document"], r["suite"], a.ds / r["doc_path"],
                                     a.ds / r["gt_path"], tuple(r["outputs"])) for r in rows}
    tests = {pid: unit_tests.tests.read(p.gt_path) for pid, p in pages.items()}
    U = unit_tests.UnitTests()

    def line(pid, score):
        p = pages[pid]
        return score.result | {"row": {"document": p.document, "suite": p.suite}, "page_id": pid}

    lines: dict[str, list[dict]] = {}
    extra: dict[str, dict] = {}
    for run in a.run:
        name = run.name
        ls, errs, ms = [], 0, []
        for pid, p in pages.items():
            f = run / "pages" / f"{pid}.json"
            rec = json.loads(f.read_text()) if f.exists() else {"error": "missing"}
            if "error" in rec:
                errs += 1
                ls.append(line(pid, U.score_failure(tests[pid])))
                continue
            ms.append(rec.get("ms", 0))
            try:
                pred = prediction(rec, p.outputs)
                ls.append(line(pid, U.score(pred, tests[pid])))
            except Exception as e:  # noqa: BLE001 - a malformed output fails its page
                errs += 1
                print(f"{name} {pid}: {type(e).__name__}: {e}", file=sys.stderr)
                ls.append(line(pid, U.score_failure(tests[pid])))
        lines[name] = ls
        extra[name] = {"failed_pages": errs, "ms_median": sorted(ms)[len(ms) // 2] if ms else None,
                       "ms_mean": round(sum(ms) / len(ms), 1) if ms else None}

    systems = [s for s in a.systems.split(",") if s]
    if systems:
        res = pq.read_table(a.results, columns=["system", "page_id", "test_id", "result", "why"],
                            filters=[("system", "in", systems)]).to_pylist()
        by = {}
        for x in res:
            if x["page_id"] in pages:
                by.setdefault(x["system"], {}).setdefault(x["page_id"], []).append(
                    unit_tests.Verdict(x["test_id"], x["result"], x["why"] or ""))
        real = compare.score
        try:
            for s in systems:
                ls = []
                for pid in pages:
                    vs = by.get(s, {}).get(pid)
                    compare.score = (lambda pred, ts, vs=vs: vs)
                    ls.append(line(pid, U.score({}, tests[pid]) if vs else U.score_failure(tests[pid])))
                lines[s] = ls
        finally:
            compare.score = real

    out = {"pages": len(pages), "summaries": {}, "paired": {}}
    for name, ls in lines.items():
        s = U.summarize(ls)
        out["summaries"][name] = {"score": s["score"],
                                  "families": {k: {"score": v["score"], "tests": v["tests"]} for k, v in s["families"].items()},
                                  "per_suite": {k: round(v["score"], 4) for k, v in s["per_suite"].items()}} | extra.get(name, {})
    names = list(lines)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            p = uncertainty.paired(lines[x], lines[y], resamples=1000, seed=0)
            keep = {k: p["headline"][k] for k in ("a", "b", "gap", "low", "high", "excludes_zero")}
            fam = {f: {k: v[k] for k in ("gap", "low", "high")} for f, v in p.get("per_family", p.get("families", {})).items()} \
                if isinstance(p.get("per_family", p.get("families")), dict) else {}
            out["paired"][f"{x} vs {y}"] = keep | {"families": fam}
    per_page = {name: {l["page_id"]: [l["tests"], l["passed"]] for l in ls} for name, ls in lines.items()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1))
    a.out.with_suffix(".pages.json").write_text(json.dumps(per_page))
    for name, s in out["summaries"].items():
        print(f"{name:28s} {100 * s['score']:6.2f}  " + "  ".join(
            f"{k} {100 * v['score']:.1f}" for k, v in s["families"].items()))
    for k, v in out["paired"].items():
        print(f"{k}: gap {100 * v['gap']:+.2f} [{100 * v['low']:+.2f}, {100 * v['high']:+.2f}]"
              if v["low"] is not None else f"{k}: gap {100 * v['gap']:+.2f}")


if __name__ == "__main__":
    main()
