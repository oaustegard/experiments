"""The #102 metric, unchanged: top-10 overlap with EG2 on the committed 2-bit index for
real_test (1,000) and synth_test (1,498), bootstrap 95% CIs, and synth_test source-post
hit@10 with its paired difference against EG2's own hit@10."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import PREV, load  # noqa: E402

sys.path.insert(0, str(PREV / "run"))
from common import Index, boot_ci, overlap  # noqa: E402
from final_eval import paired  # noqa: E402

TESTS = ["real_test", "synth_test"]


class Bench:
    def __init__(self):
        self.ix = Index()
        self.rows, self.E, self.top = {}, {}, {}
        for t in TESTS:
            self.rows[t], self.E[t] = load(t)
            self.top[t] = self.ix.topk(self.E[t])
        self.src = np.array([r["post"] for r in self.rows["synth_test"]])
        self.hit_t = (self.top["synth_test"] == self.src[:, None]).any(1).astype(float)

    def texts(self, t):
        return [r["q"] for r in self.rows[t]]

    def score(self, P):
        """P: {test: (n, 768) unit vectors}. Returns (summary, per-query overlap arrays)."""
        r, per = {}, {}
        for t in TESTS:
            top = self.ix.topk(P[t])
            per[t] = overlap(top, self.top[t])
            r[t] = {"overlap@10": boot_ci(per[t]),
                    "cos_to_eg2": float((P[t] * self.E[t]).sum(1).mean())}
            if t == "synth_test":
                h = (top == self.src[:, None]).any(1).astype(float)
                r[t]["hit@10"] = boot_ci(h)
                r[t]["hit@10_minus_eg2"] = paired(h, self.hit_t)
        return r, per
