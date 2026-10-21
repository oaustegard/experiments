"""Print results/onnx_NAME.json as a table: size, hit@10, real-query agreement, cosine to fp32."""
import json
import sys
from pathlib import Path

for name in sys.argv[1:]:
    r = json.loads((Path(__file__).resolve().parent.parent / "results" / f"onnx_{name}.json").read_text())
    for k, v in r.items():
        c, s = v.get("cos_to_fp32_real", {}), v.get("cos_to_fp32_synth", {})
        print(f"{name:8s} {k:18s} raw {v['mb']:6} br {v.get('br_mb', v.get('gz_mb'))!s:6} hit@10 {v['hit@10'][0]:.3f}"
              f" vsEG2 {v['minus_eg2'][0]:+.3f} ovEG2 {v['real_overlap_with_eg2'][0]:.3f}"
              f" cos real p05/min {c.get('p05', 1):.3f}/{c.get('min', 1):.3f} synth p05 {s.get('p05', 1):.3f}")
