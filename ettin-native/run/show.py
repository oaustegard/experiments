"""Print results/screen.json as a table."""
import json
from pathlib import Path

r = json.loads((Path(__file__).resolve().parent.parent / "results" / "screen.json").read_text())
for k, v in sorted(r.items(), key=lambda kv: -kv[1]["rx2"]["hit@10"][0]):
    x = v["rx2"]
    print(f"{k:55s} {v['params_m']:6}M d{v['dim']:<5} fp32 {v['fp32']['hit@10'][0]:.3f}  rx2 {x['hit@10'][0]:.3f}"
          f"  vsEG2 {x['minus_eg2'][0]:+.3f}  h@1 {x['hit@1']:.3f}  ovEG2 {x['real_overlap_with_eg2'][0]:.3f}  qp={v['query_prompt']!r}")
