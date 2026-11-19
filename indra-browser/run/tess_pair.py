"""Where does Tesseract pass an OmniParseBench test that Indra OCR fails?

Pairs, test by test, a local Indra run (opb_eval.py's OUT/pages/*.json) with the
benchmark's published verdicts (results.parquet: one row per system and test).
Tesseract's published run is the benchmark's own adapter (html only, the page's
script detected with --psm 0, PDFs rendered at 300 DPI, no text layer). The
published "Indra OCR" row is indra-ocr 0.3.2 in mode auto (it reads a PDF's text
layer) with math on; the local run is 1.1.2 in OCR mode with math off, the
engine the browser port reproduces. Only text-family tests (present, order,
repeat) are compared: Tesseract returns no tables or layout.

    python3 run/tess_pair.py --run /tmp/indra-eval/out/fp32 --results /tmp/opb-data/ds/results.parquet \
        --manifest /tmp/opb-data/ds/manifest.parquet --out results/tess_pair.json
"""
from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

import pandas as pd

TEXT = {"present", "order", "repeat"}


def local(run: Path) -> pd.DataFrame:
    rows = []
    for f in sorted((run / "pages").glob("*.json")):
        rec = json.loads(f.read_text())
        rows += [{"test_id": ln["test_id"], "result": ln["result"]} for ln in rec["lines"]]
    return pd.DataFrame(rows).set_index("test_id")


def table(df: pd.DataFrame, key: str, a: str, b: str) -> list[dict]:
    out = []
    for k, g in df.groupby(key):
        out.append({key: k, "tests": len(g), f"{a}_pass": round(float((g[a] == "pass").mean()), 3),
                    f"{b}_pass": round(float((g[b] == "pass").mean()), 3),
                    "rescued": int(((g[a] == "fail") & (g[b] == "pass")).sum()),
                    "lost": int(((g[a] == "pass") & (g[b] == "fail")).sum())})
    return sorted(out, key=lambda r: -r["rescued"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--results", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    pub = pd.read_parquet(a.results)
    pub = pub[pub.test_type.isin(TEXT)]
    meta = pub[pub.system == "Tesseract"].set_index("test_id")[["page_id", "suite", "test_type", "tags", "why"]]
    df = meta.rename(columns={"why": "tess_why"})
    df["tess"] = pub[pub.system == "Tesseract"].set_index("test_id").result
    df["pub_indra"] = pub[pub.system == "Indra OCR"].set_index("test_id").result
    loc = local(a.run)
    df["indra"] = loc.result
    missing = int(df.indra.isna().sum())
    df = df.dropna(subset=["indra"])
    man = pd.read_parquet(a.manifest).set_index("page_id")
    df["source"] = man.doc_path.reindex(df.page_id).str.lower().str.endswith(".pdf").map(
        {True: "pdf", False: "image"}).values
    df = df[df.tess != "unsupported"]

    res = {"tests": len(df), "tests_missing_locally": missing}
    for arm in ("indra", "pub_indra"):
        x = df[df[arm] != "unsupported"]
        res[arm] = {"pass": round(float((x[arm] == "pass").mean()), 4),
                    "crosstab": {f"{i}/{t}": int(n) for (i, t), n in
                                 x.groupby([arm, "tess"]).size().items()}}
    res["tess_pass"] = round(float((df.tess == "pass").mean()), 4)
    resc = df[(df.indra == "fail") & (df.tess == "pass")]
    res["rescued"] = len(resc)
    res["rescued_share_of_indra_fails"] = round(len(resc) / max(1, int((df.indra == "fail").sum())), 4)
    res["oracle_text_pass"] = round(float(((df.indra == "pass") | (df.tess == "pass")).mean()), 4)
    res["rescues_per_page"] = dict(sorted(collections.Counter(resc.page_id.value_counts()).items()))
    res["by_source"] = table(df, "source", "indra", "tess")
    res["by_suite"] = table(df, "suite", "indra", "tess")
    res["by_test_type"] = table(df, "test_type", "indra", "tess")
    tags = df.explode("tags")
    res["by_tag"] = [r for r in table(tags, "tags", "indra", "tess") if r["tests"] >= 15][:30]
    res["top_pages"] = resc.page_id.value_counts().head(15).to_dict()
    a.out.write_text(json.dumps(res, indent=1, ensure_ascii=False))
    print(json.dumps({k: res[k] for k in ("tests", "tests_missing_locally", "indra", "pub_indra", "tess_pass",
                                          "rescued", "rescued_share_of_indra_fails", "oracle_text_pass",
                                          "rescues_per_page")}, indent=1))
    for k in ("by_source", "by_suite", "by_test_type", "by_tag"):
        print(k)
        print(pd.DataFrame(res[k]).to_string(index=False))


if __name__ == "__main__":
    main()
