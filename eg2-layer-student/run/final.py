"""Collect every arm into one table, with #102's reference rows beside them.

Usage: python3 -I run/final.py

L0 = results/l0.json (PyTorch, teacher weights, full vocabulary; no download built).
L1 = the last eval in results/l1_<tag>.json (PyTorch fp32, full vocabulary).
L2 = results/export_<tag>.json, the shipped ONNX itself (q4 unless named otherwise).
#102 rows are copied from eg2-query-student/results/final.json. Writes results/final.json.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PREV, ROOT  # noqa: E402

R = ROOT / "results"


def row(r, nbytes=None, **extra):
    out = {"bytes": nbytes, "mb": None if nbytes is None else round(nbytes / 1e6, 1), **extra}
    for t in ("real_test", "synth_test"):
        out[t] = {k: v for k, v in r[t].items()}
    return out


def main():
    res = {}
    prev = json.loads((PREV / "results" / "final.json").read_text())
    res["#102 EG2 (teacher)"] = {"bytes": prev["EG2 (teacher)"]["bytes"],
                                 "mb": round(prev["EG2 (teacher)"]["bytes"] / 1e6, 1),
                                 "synth_test": prev["EG2 (teacher)"]["synth_test"]}
    for name in ("A3 leaf_cont165k epoch0 int8", "A4 pruned EG2 K=32000"):
        res[f"#102 {name}"] = row(prev[name], prev[name]["bytes"])
    for k, r in json.loads((R / "l0.json").read_text()).items():
        res[f"L0 k={k}"] = row(r, params=r["params"])
    for p in sorted(R.glob("l1_k*.json")):
        r = json.loads(p.read_text())
        last = max(r["evals"], key=lambda s: int(s[4:]))
        res[f"L1 {p.stem[3:]} {last}"] = row(r["evals"][last], lr=r["lr"], trains=r["trains"],
                                             n_train=r["n_train"], init=r.get("init"), vocab=r["vocab"])
    for p in sorted(R.glob("export_l2_*.json")):
        r = json.loads(p.read_text())
        for f, v in r["formats"].items():
            if f == "fp32":
                continue
            res[f"L2 {p.stem[10:]} {f}"] = row(v, v["bytes"], cos_to_pytorch=v.get("cos_to_pytorch"), cos_to_q4=v.get("cos_to_q4"),
                                               weights=r["weights"], vocab=r["vocab"])
    (R / "final.json").write_text(json.dumps(res, indent=1))
    for n, v in res.items():
        cells = [f"{v['mb']} MB" if v.get("mb") else "-"]
        for t in ("real_test", "synth_test"):
            o = v.get(t, {}).get("overlap@10")
            cells.append(f"{o[0]:.3f}" if o else "-")
        h = v["synth_test"].get("hit@10")
        cells.append(f"{h[0]:.3f}" if h else "-")
        print(f"{n:42s} " + " | ".join(cells))


if __name__ == "__main__":
    main()
