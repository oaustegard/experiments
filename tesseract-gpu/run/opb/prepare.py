"""OmniParseBench: download the dataset, render a fixed text-test subset, write its index.

    python3 -I run/opb/prepare.py

- data/opb/ds: the dataset at the pinned revision (manifest, gold, pages, results.parquet).
- data/opb/gray/<page_id>.png: each subset page rendered as the grayscale image an OCR reads. The
  render is prerender.py's (indra-browser/run/prerender.py): pdfium, grayscale, 200 DPI lowered past
  100 MP, page 0 of a PDF; an image page is cv2.imread as IMREAD_GRAYSCALE. Only the subset is
  rendered; prerender.py renders all 2937 pages.
- data/opb/subset.json: the subset with each page's suite, source and text-test count.
- results/opb_subset.json: the same, summarized (page ids and text-test counts).

The subset: pages carrying at least one text-family test (present, order, repeat, page_near_empty:
omni_parse_bench vocab.FAMILIES["text"]), allocated across strata of (suite, source) in proportion
to their eligible pages, largest remainder, then sampled per stratum with random.Random(SEED).
The gold files are read as plain JSON here, so this script does not import the benchmark package:
that runs only inside scripts/jail.sh (see run/opb/score.py).
"""
from __future__ import annotations

import argparse
import collections
import json
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data" / "opb"
REVISION = "d70182b9f901d1660a27e60bf6be6d6e6ea83013"   # datalab-to/omni_parse_bench, as read 2026-11-20
TEXT_TYPES = frozenset({"present", "order", "repeat", "page_near_empty"})
N_PAGES = 400
SEED = 0
DPI = 200.0
MAX_PAGE_PIXELS = 100e6


def download(ds: Path) -> None:
    from huggingface_hub import snapshot_download

    snapshot_download(repo_type="dataset", repo_id="datalab-to/omni_parse_bench", revision=REVISION,
                      local_dir=str(ds), max_workers=8)


def text_test_count(ds: Path, row: dict) -> tuple[int, int]:
    tests = json.loads((ds / row["gt_path"]).read_text())["tests"]
    return sum(t["test_type"] in TEXT_TYPES for t in tests), len(tests)


def source(row: dict) -> str:
    return "pdf" if row["doc_path"].endswith(".pdf") else "png"


def allocate(sizes: dict[tuple[str, str], int], n: int) -> dict[tuple[str, str], int]:
    """Largest-remainder allocation of n draws across strata in proportion to their sizes."""
    total = sum(sizes.values())
    exact = {k: n * s / total for k, s in sizes.items()}
    quota = {k: min(int(v), sizes[k]) for k, v in exact.items()}
    left = n - sum(quota.values())
    order = sorted(sizes, key=lambda k: (-(exact[k] - int(exact[k])), k))
    for k in order:
        if left <= 0:
            break
        if quota[k] < sizes[k]:
            quota[k] += 1
            left -= 1
    return quota


def select(ds: Path, rows: list[dict]) -> tuple[list[dict], int]:
    counted = []
    for r in rows:
        n_text, n_all = text_test_count(ds, r)
        if n_text:
            counted.append(dict(r, source=source(r), n_text_tests=n_text, n_tests=n_all))
    strata: dict[tuple[str, str], list[dict]] = collections.defaultdict(list)
    for r in counted:
        strata[(r["suite"], r["source"])].append(r)
    quota = allocate({k: len(v) for k, v in strata.items()}, N_PAGES)
    rng = random.Random(SEED)
    chosen = []
    for k in sorted(strata):
        pool = sorted(strata[k], key=lambda r: r["page_id"])
        chosen += rng.sample(pool, quota[k])
    order = {r["page_id"]: i for i, r in enumerate(rows)}
    chosen.sort(key=lambda r: order[r["page_id"]])
    print(f"eligible {len(counted)} of {len(rows)} pages; {len(strata)} strata; {len(chosen)} chosen")
    return chosen, len(counted)


def _render(job):
    src, dst = job
    import cv2
    import numpy as np

    if dst.exists():
        return None
    try:
        with open(src, "rb") as f:
            head = f.read(1024)
        if b"%PDF-" in head:
            import pypdfium2 as pdfium

            doc = pdfium.PdfDocument(str(src))
            try:
                page = doc[0]
                w, h = page.get_size()
                k = min(DPI / 72.0, (MAX_PAGE_PIXELS / max(w * h, 1.0)) ** 0.5)
                arr = page.render(scale=k, grayscale=True).to_numpy()
                gray = np.ascontiguousarray(arr[..., 0] if arr.ndim == 3 else arr)
            finally:
                doc.close()
        else:
            gray = cv2.imread(str(src), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                return f"{src.name}: unreadable"
        tmp = dst.with_suffix(".tmp.png")
        cv2.imwrite(str(tmp), gray, [cv2.IMWRITE_PNG_COMPRESSION, 3])
        tmp.rename(dst)
    except Exception as e:  # noqa: BLE001
        return f"{src.name}: {type(e).__name__}: {e}"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-download", action="store_true")
    a = ap.parse_args()
    ds = DATA / "ds"
    gray = DATA / "gray"
    if not a.skip_download:
        download(ds)
    import pyarrow.parquet as pq

    rows = pq.read_table(ds / "manifest.parquet").to_pylist()
    chosen, eligible = select(ds, rows)

    DATA.mkdir(parents=True, exist_ok=True)
    gray.mkdir(parents=True, exist_ok=True)
    jobs = [(ds / r["doc_path"], gray / f"{r['page_id']}.png") for r in chosen]
    with ProcessPoolExecutor(a.workers) as ex:
        errs = [e for e in ex.map(_render, jobs, chunksize=8) if e]
    print(f"rendered {len(jobs)} pages to {gray}, {len(errs)} errors")
    for e in errs[:20]:
        print("  ", e)
    if errs:
        raise SystemExit(1)

    pages = [{"page_id": r["page_id"], "document": r["document"], "suite": r["suite"], "source": r["source"],
              "doc_path": r["doc_path"], "gt_path": r["gt_path"], "n_text_tests": r["n_text_tests"],
              "n_tests": r["n_tests"]} for r in chosen]
    subset = {"revision": REVISION, "seed": SEED, "n_pages": len(pages), "eligible_pages": eligible,
              "text_types": sorted(TEXT_TYPES), "allocation": "largest remainder over (suite, source) strata",
              "pages": pages}
    (DATA / "subset.json").write_text(json.dumps(subset, indent=1))

    summary = {"revision": REVISION, "seed": SEED, "n_pages": len(pages), "eligible_pages": eligible,
               "n_text_tests": sum(p["n_text_tests"] for p in pages),
               "by_suite": dict(sorted(collections.Counter(p["suite"] for p in pages).items())),
               "by_source": dict(sorted(collections.Counter(p["source"] for p in pages).items())),
               "pages": {p["page_id"]: p["n_text_tests"] for p in pages}}
    out = REPO / "results" / "opb_subset.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"wrote {DATA / 'subset.json'} and {out}: {len(pages)} pages, {summary['n_text_tests']} text tests")


if __name__ == "__main__":
    main()
