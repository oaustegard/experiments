"""Paths, the browser-decider imports, and the one sampling rule every growth method shares."""
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT.parent))
from _lib.paths import experiment  # noqa: E402

BD = experiment("browser-decider")
sys.path.insert(0, str(BD / "run"))
from common import WORKFLOWS, examples, load_cases, score_examples  # noqa: E402,F401

E17C = BD / "out/base/st-ettin17m-tuned-e17-cap-c-epoch1/epoch1"
E17C_URL = ("https://github.com/oaustegard/experiments/releases/download/ettin-native-20261021/"
            "st-ettin17m-tuned-e17-cap-c-epoch1.zip")
SIZES = [0, 16, 64, 300]
GPU_SEEDS = [0, 1]
KNN_SEEDS = [0, 1, 2, 3, 4]


def revealed(workflow: str, seed: int, n: int) -> list[dict]:
    """The first n of W's train cases in a fixed per-(workflow, seed) order: the same sample for knn,
    scratch, warm and full, and nested (the 16 are inside the 64)."""
    cases = load_cases(workflow, "train")
    order = sorted(cases, key=lambda c: c["id"])
    random.Random(f"{workflow}-{seed}").shuffle(order)
    return order[:n]
