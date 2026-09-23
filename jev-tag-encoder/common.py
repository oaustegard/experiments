"""Shared helpers: fixtures, label matrices, bootstrap CIs, metric functions."""
import json
from pathlib import Path

import numpy as np

import jev
from data import DATA, LABELS

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
LABEL_NAMES = list(LABELS)
TAG_IDX = {t: i for i, t in enumerate(jev.TAGS)}
B = 2000  # bootstrap resamples


def jsonl(p):
    return [json.loads(line) for line in open(p)]


def save(name: str, obj) -> None:
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(obj, indent=1, default=float))


def arxiv(split: str | None = None) -> tuple[list[str], np.ndarray, list[dict]]:
    recs = [r for r in jsonl(DATA / "arxiv.jsonl") if split is None or r["split"] == split]
    Y = np.array([[lab in r["labels"] for lab in LABEL_NAMES] for r in recs], dtype=bool)
    return [r["id"] for r in recs], Y, recs


def label_scores(P: np.ndarray) -> np.ndarray:
    """(n, 256) tag probabilities -> (n, 24) label scores, max over each label's mapped tags."""
    return np.stack([P[:, [TAG_IDX[t] for t in LABELS[lab]]].max(axis=1) for lab in LABEL_NAMES], axis=1)


def f1s(Y: np.ndarray, Yhat: np.ndarray) -> tuple[float, float]:
    """(micro F1, macro F1 over labels with >= 1 positive in Y)."""
    tp = (Y & Yhat).sum(0).astype(float)
    fp = (~Y & Yhat).sum(0).astype(float)
    fn = (Y & ~Yhat).sum(0).astype(float)
    micro = 2 * tp.sum() / max(1e-9, 2 * tp.sum() + fp.sum() + fn.sum())
    has = Y.sum(0) > 0
    per = np.where(2 * tp + fp + fn > 0, 2 * tp / np.maximum(1e-9, 2 * tp + fp + fn), 0.0)
    return float(micro), float(per[has].mean())


def ap_scores(Y: np.ndarray, S: np.ndarray) -> tuple[float, float]:
    """(micro-AP, macro-AP) — threshold-free, so arms are not ordered by calibration."""
    from sklearn.metrics import average_precision_score
    has = Y.sum(0) > 0
    micro = average_precision_score(Y[:, has].ravel(), S[:, has].ravel())
    macro = np.mean([average_precision_score(Y[:, j], S[:, j]) for j in np.where(has)[0]])
    return float(micro), float(macro)


def boot(stat, n: int, seed: int = 0) -> np.ndarray:
    """Bootstrap over n units: stat(idx) -> scalar or tuple; returns (B, k) array."""
    rng = np.random.default_rng(seed)
    return np.array([np.atleast_1d(stat(rng.integers(0, n, n))) for _ in range(B)])


def ci(samples: np.ndarray) -> list[list[float]]:
    return np.percentile(samples, [2.5, 97.5], axis=0).T.round(4).tolist()


def fmt(point: float, lohi) -> str:
    return f"{point:.3f} [{lohi[0]:.3f}, {lohi[1]:.3f}]"
