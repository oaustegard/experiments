"""Differential test: every compute entry point in the corpus whose resources are all
buffers runs before and after minification on identical random inputs; every buffer
must come back bit-identical. The original runs twice first, and an entry point whose
two original runs differ (a data race or atomics ordering) is reported as
nondeterministic instead of compared.

    python3 tests/diff_compute.py [--json results/diff_compute.json]
"""
import base64
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
from forge.ast import Decl, Fn                # noqa: E402
from forge.minify import minify                # noqa: E402
from forge.reflect import bindings, buffer_type  # noqa: E402
from forge.scope import is_entry               # noqa: E402
from forge.wgsl import parse                   # noqa: E402
from forge.gpu import run_jobs                 # noqa: E402


def cases():
    for f in sorted((HERE / "corpus").glob("*/*.wgsl")):
        src = f.read_text()
        m = parse(src)
        entries = [d.name for d in m.decls if isinstance(d, Fn) and is_entry(d)
                   and any(a.name == "compute" for a in d.attrs)]
        if not entries:
            continue
        try:
            bs, lay = bindings(m)
        except ValueError as e:
            yield f, None, None, None, f"skip: {e}"
            continue
        if any(buffer_type(d) is None for _, _, d in bs):
            yield f, None, None, None, "skip: non-buffer resource"
            continue
        rng = np.random.default_rng(7)
        try:
            binds = [{"group": g, "binding": b, "type": buffer_type(d),
                      "data": base64.b64encode(lay.fill(d.type, rng)).decode()} for g, b, d in bs]
        except ValueError as e:
            yield f, None, None, None, f"skip: {e}"
            continue
        # overrides without a default must be set by the host; give them 1
        consts = {}
        for d in m.decls:
            if isinstance(d, Decl) and d.kind == "override" and d.init is None:
                ids = [a.args[0].text.rstrip("iu") for a in d.attrs if a.name == "id" and a.args]
                consts[ids[0] if ids else d.name] = 1
        yield f, src, entries, (binds, consts), None


def main():
    jobs, meta, report = [], [], {}
    for f, src, entries, bc, skip in cases():
        rel = str(f.relative_to(HERE))
        if skip:
            report[rel] = skip
            continue
        mini = minify(src)
        binds, consts = bc
        for e in entries:
            for tag, code in (("o1", src), ("o2", src), ("m", mini)):
                jobs.append({"id": f"{rel}|{e}|{tag}", "kind": "compute", "wgsl": code,
                             "entry": e, "dispatch": [2, 1, 1], "bindings": binds,
                             "constants": consts})
            meta.append((rel, e))
    res = run_jobs(jobs)
    counts = {"identical": 0, "differs": 0, "nondeterministic": 0, "error": 0,
              "files_skipped": sum(1 for v in report.values() if str(v).startswith("skip")),
              "files_run": len({rel for rel, _ in meta}), "entry_points": len(meta)}
    for rel, e in meta:
        o1, o2, m = (res[f"{rel}|{e}|{t}"] for t in ("o1", "o2", "m"))
        if not (o1["ok"] and o2["ok"] and m["ok"]):
            verdict = "error: " + "; ".join(r.get("error", "") for r in (o1, m) if not r["ok"])[:300]
            counts["error"] += 1
        elif o1["buffers"] != o2["buffers"]:
            verdict = "nondeterministic"
            counts["nondeterministic"] += 1
        elif o1["buffers"] == m["buffers"]:
            verdict = "identical"
            counts["identical"] += 1
        else:
            bad = [k for k in o1["buffers"] if o1["buffers"][k] != m["buffers"][k]]
            verdict = f"differs in {bad}"
            counts["differs"] += 1
        report[f"{rel}::{e}"] = verdict
    for k, v in report.items():
        print(f"{v:<40} {k}")
    print(counts)
    if "--json" in sys.argv:
        out = Path(sys.argv[sys.argv.index("--json") + 1])
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"counts": counts, "cases": report}, indent=1))
    return 1 if counts["differs"] else 0


if __name__ == "__main__":
    sys.exit(main())
