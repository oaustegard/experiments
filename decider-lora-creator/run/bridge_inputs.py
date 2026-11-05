"""Inputs for web/bridge_job.js: adapter-library's growth-arm samples for one workflow.

    python3 -I run/bridge_inputs.py SHA [WORKFLOW] > out/bridge_cfg.json

The revealed cases are adapter-library's own `revealed(workflow, seed, n)` (seeded, nested), and the
epochs are its `growth_epochs(n)`, so the page's head-only runs see the same cases and the same number
of steps as the Python LoRA growth arms (results/G-*-<workflow>-n<n>-s<seed>.json there).
"""
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

sys.path.insert(0, str(experiment("adapter-library") / "run"))
from shared import revealed  # noqa: E402

SIZES, SEEDS = [16, 64, 300], [0, 1]


def growth_epochs(n, bs=16):   # adapter-library/run/lora.py, verbatim
    spe = math.ceil(5 * n / bs)
    return min(40, max(10, math.ceil(150 / spe)))


def main():
    sha = sys.argv[1]
    w = sys.argv[2] if len(sys.argv) > 2 else "invoice_processing"
    cfg = {"sha": sha, "workflow": w, "models": "https://austegard.com/experiments/adapter-library/web/models/",
           "reveal": {str(n): {str(s): [c["id"] for c in revealed(w, s, n)] for s in SEEDS} for n in SIZES},
           "epochs": {str(n): growth_epochs(n) for n in SIZES}, "lr": 8e-4, "bs": 16}
    print(json.dumps(cfg))


if __name__ == "__main__":
    main()
