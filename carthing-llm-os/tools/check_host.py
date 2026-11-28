"""Check the C forward pass (built natively as build/host/host_run) against the
PyTorch reference from tools/convert.py, teacher-forced on two sequences: the
prompt plus its reference greedy continuation, and a 115-token paragraph.

Also measures the noise floor of the device arithmetic: two host builds that
differ only in FMA contraction, compared with each other. Re-quantizing the
activations to Q8_0 before every matrix product turns 1-ulp differences into
logit differences of this size, so it is the tolerance for comparing the
kernel (NEON, different summation order) with the host build.

Writes results/host_check.json.
"""
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent.parent
BLOB = HERE / "work" / "qwen3-0.6b-q4_0.bin"
BIN = HERE / "build" / "host" / "host_run"
BIN_EXACT = HERE / "build" / "host" / "host_run_exact"


def run(ids, binary=BIN):
    with tempfile.TemporaryDirectory() as d:
        ids_path, out_path = Path(d) / "ids.bin", Path(d) / "logits.bin"
        ids_path.write_bytes(np.asarray(ids, dtype=np.uint32).tobytes())
        subprocess.run([str(binary), str(BLOB), "logits", str(ids_path), str(out_path)], check=True)
        return np.fromfile(out_path, dtype=np.float32).reshape(len(ids), -1)


def compare(c, ref):
    d = c.astype(np.float64) - ref.astype(np.float64)
    cos = (c * ref).sum(-1) / (np.linalg.norm(c, axis=-1) * np.linalg.norm(ref, axis=-1))
    p = np.exp(ref - ref.max(-1, keepdims=True)); p /= p.sum(-1, keepdims=True)
    q = np.exp(c - c.max(-1, keepdims=True)); q /= q.sum(-1, keepdims=True)
    kl = (p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum(-1)
    return {
        "positions": int(len(c)),
        "top1_agreement": float((c.argmax(-1) == ref.argmax(-1)).mean()),
        "max_abs_logit_diff": float(np.abs(d).max()),
        "rms_logit_diff": float(np.sqrt((d ** 2).mean())),
        "min_cosine": float(cos.min()),
        "mean_kl_ref_to_c": float(kl.mean()),
        "max_kl_ref_to_c": float(kl.max()),
    }


def build_variant(name, flag):
    out = HERE / "build" / "host" / name
    src = [str(HERE / p) for p in ("tools/host_run.c", "src/qwen3.c", "src/q4.c")]
    subprocess.run(["clang", "-O2", "-march=native", "-fno-math-errno", flag, "-pthread",
                    "-I" + str(HERE / "src"), *src, "-o", str(out)], check=True, stderr=subprocess.DEVNULL)
    return out


def main():
    ref = np.load(HERE / "work" / "ref.npz")
    c_eval = run(ref["eval_ids"])
    out = {
        "generation_sequence": compare(run(ref["gen_ids"]), ref["gen_logits"]),
        "eval_paragraph": compare(c_eval, ref["eval_logits"]),
        "eval_paragraph_c_vs_bf16_top1": float((c_eval.argmax(-1) == ref["eval_logits_orig"].argmax(-1)).mean()),
        # Float activations and f32 KV cache: what is left is the C code itself.
        "exact_build_eval_paragraph": compare(run(ref["eval_ids"], BIN_EXACT), ref["eval_logits"]),
    }
    a = run(ref["gen_ids"], build_variant("host_run_nofma", "-ffp-contract=off"))
    b = run(ref["gen_ids"], build_variant("host_run_fma", "-ffp-contract=fast"))
    out["noise_floor_fma_off_vs_fast_generation_sequence"] = compare(a, b)
    (HERE / "results" / "host_check.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
