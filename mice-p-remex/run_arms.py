"""Apply each codec arm to one dataset's stored vectors and write decoded arrays.

    python3 run_arms.py DATASET ARM [ARM ...]

Arms: plaid-{2,4,8}, plaid-n-{2,4}, remex-{2,4}, remex-c-{1,2,3,4},
remex-wup-{1,2}, remex-wkv-{1,2}. Fits (k-means, means, buckets) use the
dataset's own candidate tokens, as an index build would. Writes
cache/decoded/DS/ARM.npy and appends bytes/token and proxy errors to
results/arms_DS.json.
"""
import json
import sys
import time
from pathlib import Path

import mcodecs as C
import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_ST = HERE / "cache" / "models" / "Mice-P-130m" / "model.safetensors"


def arm_fn(arm, A):
    kind, bits = arm.rsplit("-", 1)
    bits = int(bits)
    return {
        "plaid": lambda X: C.plaid(X, bits),
        "plaid-n": lambda X: C.plaid(X, bits, normalize_input=True),
        "plaid-rx": lambda X: C.plaid_remex(X, bits),
        "remex": lambda X: C.remex_plain(X, bits),
        "remex-c": lambda X: C.remex_centered(X, bits),
        "remex-wup": lambda X: C.remex_centered(X, bits, A=A["up"]),
        "remex-wkv": lambda X: C.remex_centered(X, bits, A=A["kv"]),
    }[kind]


def main(ds, *arms):
    X = np.load(HERE / "cache" / "vecs" / ds / "m130.vecs.npy")
    A = C.metric_from_weights(MODEL_ST)
    out_dir = HERE / "cache" / "decoded" / ds
    out_dir.mkdir(parents=True, exist_ok=True)
    res_path = HERE / "results" / f"arms_{ds}.json"
    res_path.parent.mkdir(exist_ok=True)
    res = json.loads(res_path.read_text()) if res_path.exists() else {}
    for arm in arms:
        t0 = time.time()
        Xh, b, meta = arm_fn(arm, A)(X)
        np.save(out_dir / f"{arm}.npy", Xh.astype(np.float32))
        res[arm] = {"bytes_per_token": b, "err_up": C.score_err(X, Xh, A["up"]),
                    "err_kv": C.score_err(X, Xh, A["kv"]), **meta, "seconds": round(time.time() - t0, 1)}
        print(arm, res[arm], flush=True)
        res_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
