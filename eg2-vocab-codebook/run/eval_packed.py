"""Score a shipped engine build (manifest.json + weights + tokenizer.json) on all three benchmarks,
through the engine's own NumPy forward pass (eg2-wgsl-engine/run/ref.py, rhq decode for rhq tables),
so the numbers are those of the file the page loads.
Usage: python3 -I run/eval_packed.py NAME DIR  -> results/packed_<NAME>.json"""
import json
import sys
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vc import HERE, Benches  # noqa: E402
from engine_check import ref, rhq_rows  # noqa: E402

orig_rows = ref.table_rows


class RefModel:
    """Duck-types the torch Student for vc.Benches.run via a patched embed()."""


def main():
    name, d = sys.argv[1], Path(sys.argv[2])
    man = json.loads((d / "manifest.json").read_text())
    data = (d / man["data"]).read_bytes()
    ref.table_rows = rhq_rows if man["table"].get("kind") == "rhq" else orig_rows
    R = ref.Ref(man, data)
    import vc
    vc_embed = lambda model, enc, batch=128: np.stack([R(x) for x in enc]).astype(np.float32)  # noqa: E731
    vc.embed = vc_embed
    import vc as _v
    _v.Benches.run.__globals__["embed"] = vc_embed
    B = Benches()
    tok = (name, Tokenizer.from_file(str(d / "tokenizer.json")))
    out, per = B.run(None, tok, per=True)
    np.savez(HERE / ".work" / f"packed_{name}.npz", **per)
    out["weights_mb"] = round(len(data) / 1e6, 3)
    print(name, out, flush=True)
    (HERE / "results" / f"packed_{name}.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
