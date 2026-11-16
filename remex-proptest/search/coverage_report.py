"""Turn the event() counters of one harness run into COVERAGE.md.

    REMEX_PT_EVENTS=events.json pytest -q test_search_props.py --hypothesis-show-statistics \
        > hypothesis_statistics.txt
    python coverage_report.py events.json [hypothesis_statistics.txt]

The JSON counts every `ev(...)` call over the whole run (one count per
generated case or per executed operation, depending on the label), so the
numbers say what the generators actually produced, not what they could.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
counts = json.loads(Path(sys.argv[1]).read_text())
stats = Path(sys.argv[2]).read_text() if len(sys.argv) > 2 else ""

GROUPS = [
    ("Quantizer mode", r"^mode="),
    ("Rotation", r"^rotation="),
    ("Bits", r"^bits="),
    ("Dimension d", r"^d="),
    ("Corpus size n", r"^n="),
    ("k relative to n", r"^k(<|==|>|=)"),
    ("Corpus content classes", r"^(duplicate_rows|zero_rows|tiny_or_zero_norm_rows|zero_query|empty_corpus|threaded_kernel)"),
    ("Packing alignment (PackedVectors)", r"^row_(un)?aligned"),
    ("Centred mean kinds", r"^centred_mean="),
    ("Script length (ops replayed on one container)", r"^script_len="),
    ("CPU operations", r"^cpu_op=[a-z_]+$"),
    ("CPU operation x precision class", r"^cpu_op=.*:precision="),
    ("GPU(numpy) operations", r"^gpu_op="),
    ("Cache state", r"^(gpu:cpu_warmed_first|derive:caches_warmed)"),
    ("IVF configuration", r"^ivf_"),
    ("Ties and near-ties actually met", r"(tie|identical_rows_unequal|path_vs_path)"),
    ("Metamorphic / derived containers", r"^(permuted|derive:|scaling_checked|identical_code|adc_chunk|precision_bits_eq|renorm_reader|score_vectors_packed|byte_identical|offset_reorder|encode_block|query_dtype|invalid_precision|roundtrip)"),
]

used = set()
out = ["# Coverage of generated cases, area A", "",
       "Event counts from one pytest run of `test_search_props.py` "
       "(`REMEX_PT_EVENTS`).  Counts are per generated case for case-class labels and per executed "
       "operation for `cpu_op=` / `gpu_op=` / `roundtrip:` / `adc_chunk_invariance:` labels.", ""]
for title, pat in GROUPS:
    rows = [(k, v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
            if re.search(pat, k) and k not in used]
    if not rows:
        continue
    used.update(k for k, _ in rows)
    out += [f"## {title}", "", "| event | count |", "|---|---:|"] + [f"| `{k}` | {v} |" for k, v in rows] + [""]
rest = [(k, v) for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])) if k not in used]
if rest:
    out += ["## Other", "", "| event | count |", "|---|---:|"] + [f"| `{k}` | {v} |" for k, v in rest] + [""]
if stats:
    out += ["## Hypothesis statistics (per property)", "",
            "Passing / failing / invalid test cases per property, from `--hypothesis-show-statistics`.", ""]
    for m in re.finditer(r"test_search_props\.py::(\w+):\n((?:  .*\n|\n)+?)(?=test_search_props|\Z)", stats):
        gen = re.search(r"during generate phase.*?\n\s+- Typical runtimes: ([^\n]+)\n\s+- ([^\n]+)", m.group(2), re.S)
        if gen:
            out.append(f"- `{m.group(1)}`: {gen.group(2)}; runtimes {gen.group(1)}")
    out.append("")
(HERE / "COVERAGE.md").write_text("\n".join(out))
print("wrote", HERE / "COVERAGE.md", len(counts), "event labels")
