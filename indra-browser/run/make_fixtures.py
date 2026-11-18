"""Inputs and NumPy-reference outputs (ref.py) for the browser engine's parity check
(web/test/check_engine.mjs), from a pack directory.
    python3 run/make_fixtures.py PACK OUT"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ref  # noqa: E402

pack, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(1)
cases = {}


def save(name, arr):
    a = np.ascontiguousarray(arr, np.float32)
    (out / f"{name}.bin").write_bytes(a.tobytes())
    return {"file": f"{name}.bin", "shape": list(a.shape)}


x = (rng.random((96, 128)) < 0.15).astype(np.float32)
e, W = ref.load_pack(pack, "det")
cases["det"] = {"x": save("det_x", x), "want": save("det_y", ref.det(e, W, x).transpose(1, 2, 0))}
x = (rng.random((1, 256, 256)) < 0.05).astype(np.float32)
e, W = ref.load_pack(pack, "orient")
cases["orient"] = {"x": save("orient_x", x), "want": save("orient_y", ref.classify(e, W, x))}
x = (rng.random((3, 32, 200)) < 0.2).astype(np.float32)
wd = np.array([200, 130, 40])
e, W = ref.load_pack(pack, "script")
cases["script"] = {"x": save("script_x", x), "widths": wd.tolist(), "want": save("script_y", ref.classify(e, W, x, wd))}
for name, H in (("rec", 96), ("rec_world", 32)):
    x = (rng.random((3, H, 160)) < 0.2).astype(np.float32)
    wd = np.array([160, 101, 33])
    e, W = ref.load_pack(pack, name)
    lg, ln = ref.rec(e, W, x, wd)
    o = ref.ctc_outputs(lg)
    y = np.concatenate([o["ids"][..., None], o["conf"][..., None], o["blank"][..., None], o["topv"], o["topi"]], -1)
    cases[name] = {"x": save(f"{name}_x", x), "widths": wd.tolist(), "lengths": ln.tolist(), "want": save(f"{name}_y", y)}
f = rng.random((9, 16)).astype(np.float32)
e, W = ref.load_pack(pack, "order")
a, cls, same, tab = ref.order(e, W, f)
cases["order"] = {"x": save("order_x", f), "a": save("order_a", a), "cls": save("order_cls", cls),
                  "same": save("order_same", same), "tab": save("order_tab", tab)}
(out / "cases.json").write_text(json.dumps(cases, indent=1))
print("ok", list(cases))
