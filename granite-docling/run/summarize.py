"""results/summary.json: the headline table, speed, sizes and loop counts, from the run outputs.

    python3 run/summarize.py --gen DIR [DIR ...] --indra OUT/indra-fp32 --out results/summary.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
MB = 1e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", type=Path, nargs="+", required=True)
    ap.add_argument("--indra", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    full = json.loads((HERE / "results/opb_full.json").read_text())
    brk = json.loads((HERE / "results/breakdown.json").read_text())
    # Generation: batch_seconds is a chunk's wall time divided by its pages, so its sum per GPU is that
    # GPU's busy time; s/page per T4 is busy time over pages.
    busy, n, toks, capped, gen_tokens = defaultdict(float), defaultdict(int), [], 0, []
    for d in a.gen:
        for f in d.glob("*.json"):
            r = json.loads(f.read_text())
            key = (d.parent.name, r["gpu"])
            busy[key] += r["batch_seconds"]
            n[key] += 1
            capped += r["finish"] == "length"
            gen_tokens.append(r["gen_tokens"])
    pages = sum(n.values())
    s_per_page = sum(busy.values()) / pages
    indra_ms = [json.loads(f.read_text())["ms"] for f in (a.indra / "pages").glob("*.json")]
    runs = full["runs"]

    def row(k):
        r = runs[k]
        return {"headline": round(100 * r["headline"], 2), "text": round(100 * r["families"]["text"], 2),
                "tables": round(100 * r["families"]["tables"], 2), "layout": round(100 * r["families"]["layout"], 2),
                **({"delta": round(100 * r["delta"], 2), "ci95": [round(100 * x, 2) for x in r["ci95"]]}
                   if "delta" in r else {})}

    s = {
        "pages": full["pages"], "documents": full["documents"], "tests": full["tests"],
        "benchmark": "datalab-to/omni_parse_bench @ d70182b9, scorer omni-parse-bench 1.6.0, OCR mode, math off, furniture on",
        "docling_score": row("docling-fp32-html")["headline"],
        "docling_pages": full["pages"],
        "opb": {"indra_fp32": row("indra-fp32"), "indra_f16": row("indra-f16"), "indra_mix8": row("indra-mix8"),
                "docling_fp32_docling_html": row("docling-fp32-html"), "docling_fp32_markdown": row("docling-fp32")},
        "docling_failed_pages": runs["docling-fp32-html"]["failed_pages"],
        "loops": {"capped_pages": capped, "share": round(capped / pages, 4),
                  "ended": {"pages": brk["ended"]["pages"],
                            "indra_fp32": round(100 * brk["ended"]["indra-fp32"]["headline"], 2),
                            "docling": round(100 * brk["ended"]["docling-fp32-html"]["headline"], 2),
                            "ci95": [round(100 * x, 2) for x in brk["ended"]["docling-fp32-html"]["ci95"]]},
                  "capped": {"pages": brk["capped"]["pages"],
                             "indra_fp32": round(100 * brk["capped"]["indra-fp32"]["headline"], 2),
                             "docling": round(100 * brk["capped"]["docling-fp32-html"]["headline"], 2)}},
        "speed": {"docling_s_per_page_per_t4": round(s_per_page, 2),
                  "docling_hardware": "Tesla T4 (Kaggle), vLLM 0.34.0, fp32 compute, max 8192 new tokens, greedy",
                  "docling_gen_tokens_mean": round(statistics.mean(gen_tokens)),
                  "docling_gen_tokens_median": statistics.median(gen_tokens),
                  "docling_tokens_per_s_per_t4": round(sum(gen_tokens) / sum(busy.values())),
                  "indra_ms_per_page_median": statistics.median(indra_ms),
                  "indra_hardware": "1 CPU thread per page, ONNX Runtime 1.32 (indra-ocr 1.1.2), 4 workers, container x86"},
        "download_mb": {
            "indra_f16_core_gzip": 4.05, "indra_mix8_core_gzip": 2.24,
            "docling_bf16_safetensors": 515.09,
            "docling_webgpu_space": round((115605504 + 374043648 + 658121984 + 296983 + 202896 + 637) / MB, 1),
            "docling_webgpu_space_parts": {"embed_tokens_fp16": 115.61, "vision_encoder_fp32": 374.04,
                                           "decoder_model_merged_fp32": 658.12},
            "docling_onnx_smallest_listed": round((57802752 + 54753792 + 93352576) / MB, 1),
            "docling_pack_recipe": 237.31, "docling_pack_recipe_head8": 295.31,
        },
        "surya_25_opb_reported": 85.9,
        "indra_opb_reported": 69.5,
    }
    a.out.write_text(json.dumps(s, indent=1))
    print(json.dumps({k: s[k] for k in ("docling_score", "docling_pages")}), s["speed"])


if __name__ == "__main__":
    main()
