"""1/2/3-bit codes for Jev tag vectors: a sign bit (p >= 0.5) plus confidence buckets.

Noul confidence is |2p - 1| (TypeSafe's two-option confidence). The extra bits split each side
of 0.5 into 2 (2-bit) or 4 (3-bit) confidence buckets. Two cut schemes:
  uniform  confidence edges evenly spaced: 2-bit p cuts .25/.5/.75; 3-bit every .125
  logit    edges evenly spaced in |logit p|, which spends levels where Jev's mass is (87% of values
           are <= 0.02): 2-bit p cuts .1/.5/.9; 3-bit .03/.1/.25/.5/.75/.9/.97
Each code dequantizes to the mean p of its bucket on the fit set (the 200-doc mixed corpus, which
is out-of-sample for arXiv and SciFact); an empty bucket falls back to its midpoint.
"""
import numpy as np

CUTS = {
    (1, "sign"): [0.5],
    (2, "uniform"): [0.25, 0.5, 0.75],
    (2, "logit"): [0.1, 0.5, 0.9],
    (3, "uniform"): [0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875],
    (3, "logit"): [0.03, 0.1, 0.25, 0.5, 0.75, 0.9, 0.97],
}


def codes(P: np.ndarray, cuts: list[float]) -> np.ndarray:
    """Bucket index per value; a value equal to a cut goes up (so p = 0.5 is on the yes side)."""
    return np.searchsorted(np.asarray(cuts), P, side="right")


def fit_levels(P_fit: np.ndarray, cuts: list[float]) -> np.ndarray:
    edges = [0.0, *cuts, 1.0]
    c = codes(P_fit.ravel(), cuts)
    v = P_fit.ravel()
    return np.array([v[c == k].mean() if (c == k).any() else (edges[k] + edges[k + 1]) / 2
                     for k in range(len(cuts) + 1)])


class Quantizer:
    def __init__(self, bits: int, scheme: str, P_fit: np.ndarray):
        self.bits, self.scheme = bits, scheme
        self.cuts = CUTS[(bits, scheme)]
        self.levels = fit_levels(P_fit, self.cuts)

    @property
    def name(self) -> str:
        return f"{self.bits}bit-{self.scheme}"

    def __call__(self, P: np.ndarray) -> np.ndarray:
        """Quantize then dequantize; NaN rows stay NaN."""
        out = self.levels[codes(np.nan_to_num(P, nan=0.0), self.cuts)].astype(np.float32)
        out[np.isnan(P)] = np.nan
        return out

    def occupancy(self, P: np.ndarray) -> list[float]:
        c = codes(P[~np.isnan(P).any(1)].ravel(), self.cuts)
        return (np.bincount(c, minlength=len(self.cuts) + 1) / c.size).round(4).tolist()
