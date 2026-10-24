"""Fold the check outputs into results/summary.json, the file RESULTS.md cites."""
import json
from pathlib import Path
R = Path(__file__).resolve().parent.parent / "results"
lv = json.loads((R / "lavapipe_check.json").read_text())
out = {
    "lavapipe": {
        "adapter_key": lv["adapter"]["key"],
        "kernels_compiled": lv["compile"]["kernels"],
        "kernels_failed": lv["compile"]["failed"],
        "configs_verified": sum(v["statuses"].get("verified", 0) for v in lv["verify"].values()),
        "configs_not_verified": sum(n for v in lv["verify"].values() for s, n in v["statuses"].items() if s != "verified"),
        "verify_seconds": round(sum(v["seconds"] for v in lv["verify"].values()), 1),
        "per_sweep": {k: v["statuses"].get("verified", 0) for k, v in lv["verify"].items()},
        "mutants_rejected": sum(v.get("wrong", 0) for v in lv["negative"].values()),
        "mutants_accepted": sum(v.get("verified", 0) for k, v in lv["negative"].items() if k != "bandwidth"),
        "timing_modes": {m: v["modes"] for m, v in lv["timed"].items()},
        "uncaptured_errors": len(lv["uncaptured"]),
        "seconds": lv["seconds"],
    },
}
p = R / "swiftshader_page.json"
if p.exists():
    sw = json.loads(p.read_text())
    out["swiftshader_page"] = {
        "adapter_key": sw["key"], "timing": sw["timing"], "seconds": sw["seconds"],
        "console_errors": len(sw["console_errors"]), "phone_overflow_px": sw["phone_horizontal_overflow_px"],
        "phone_saved_charts": sw.get("phone_saved_charts"),
        "configs_ok": sum(v.get("ok", 0) for v in sw["statuses"].values()),
        "configs_wrong": sum(v.get("wrong", 0) for v in sw["statuses"].values()),
        "configs_failed": sum(v.get("invalid", 0) + v.get("error", 0) for v in sw["statuses"].values()),
        "skipped": {k: v["skipped"] for k, v in sw["statuses"].items() if "skipped" in v},
        "statuses": sw["statuses"],
    }
(R / "summary.json").write_text(json.dumps(out, indent=1))
print(json.dumps(out, indent=1))
