#!/usr/bin/env python3
"""How far up the stack does this routing problem have to go before it is solved?

The lexical arms in `RESULTS.md` all fail the same way: a fitted decision list
can only learn the surface forms it was shown (0.984 fitted -> 0.239 held-out),
and even the hand-written rules top out at 0.546 / 0.486 because a regex
enumerates synonyms only as far as its author's imagination reached. A sentence
encoder has no such limit — it was trained on the synonymy itself. So this arm
is the *ceiling* the regex arms are measured against, not a competitor: if
384-dimensional semantics also lands in the fifties, the catalogue is genuinely
ambiguous and no amount of lexical cleverness was ever going to fix it.

Three label representations, because "embed the label" is underspecified:

  schema    each of the 79 targets embedded as its own schema text (tool name,
            title, description, and the per-method gloss where the `method`
            enum's description carries one). Zero training data — this is what a
            router could do on a catalogue it has never seen traffic for. It is
            also the weakest arm measured here: 0.298 on family B and 0.392 on
            wild, below the hand-written regexes' 0.546 / 0.486 and level with
            the best fitted decision list. The catalogue does not describe
            itself in the words people use to ask for it.
  centroid  each target embedded as the mean of its family-A training queries,
            renormalised. Uses labelled traffic, so its family-A number is
            fitted and meaningless; B and wild are the real ones.
  fusion    alpha * cos(schema) + (1 - alpha) * cos(centroid), alpha=0.5 fixed a
            priori rather than tuned on a test split.

Every arm exposes `score()`, so `cascade_arms.py` can use it as a thresholded
fallback behind the hand-written rules.

    python3 encoder_arms.py            # table + results_encoder.json
    python3 eval.py enc-schema enc-fusion
"""

from __future__ import annotations

import json
import os
import re
import time
from functools import lru_cache
from pathlib import Path

# Guarded so `arms.load_all()` on a container without the encoder stack skips
# this module instead of taking the whole evaluation down.
try:
    import numpy as np
    import onnxruntime as ort
    from tokenizers import Tokenizer
except ImportError as e:  # pragma: no cover - environment-dependent
    raise ImportError(
        f"encoder arm needs numpy, onnxruntime and tokenizers ({e}). "
        "python3 -m pip install --break-system-packages onnxruntime tokenizers"
    ) from e

from arms import ArmBase, labels, register

HERE = Path(__file__).resolve().parent

# bekko-embedding-v1-a8m, the same encoder `repo-index/ask.py` and `xr.py` load,
# so the vectors here live in the space those tools already validated. Not
# fetched by this file: it is a 124 MB download and an experiment arm should not
# quietly pull one. Override with BEKKO_HOME.
MODEL_DIR = Path(os.environ.get("BEKKO_HOME", Path.home() / ".cache" / "repo-index"))
DIM = 384  # config.json hidden_size; mean-pooled, so pooling does not change it
FUSION_ALPHA = 0.5

_ENCODER = None
LOAD_MS = 0.0  # ONNX session init, reported separately from per-query latency


class Encoder:
    """Mean-pooled over the attention mask, then L2-normalised.

    Copied deliberately from `xr.py`/`ask.py` rather than re-derived: METHODS.md
    records that a mismatched pooling or prefix rule produces plausible-looking
    vectors that are silently in a different space, and this model is a
    ModernBERT with `classifier_pooling: mean` and no query prefix.
    """

    def __init__(self) -> None:
        if not (MODEL_DIR / "model.onnx").exists():
            raise FileNotFoundError(
                f"no encoder at {MODEL_DIR}. Set BEKKO_HOME, or fetch it with "
                "`python3 repo-index/ask.py --build` in the experiments repo "
                "(bekko-embedding-v1-a8m, ~124 MB)."
            )
        self.tok = Tokenizer.from_file(str(MODEL_DIR / "tokenizer.json"))
        self.tok.enable_truncation(max_length=512)
        self.tok.enable_padding(pad_id=0, pad_token="<pad>")
        self.sess = ort.InferenceSession(
            str(MODEL_DIR / "model.onnx"), providers=["CPUExecutionProvider"])

    def __call__(self, texts: list[str], batch: int = 32) -> np.ndarray:
        # Length-sorted batching: padding is masked out of the mean, so this is
        # a throughput knob only — it cannot change the vectors.
        order = np.argsort([len(t) for t in texts], kind="stable")
        out = np.empty((len(texts), DIM), dtype=np.float32)
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            e = self.tok.encode_batch([texts[i] for i in idx])
            ids = np.array([x.ids for x in e], dtype=np.int64)
            am = np.array([x.attention_mask for x in e], dtype=np.int64)
            h = self.sess.run(None, {"input_ids": ids, "attention_mask": am})[0]
            m = am.astype(np.float32)[..., None]
            out[idx] = (h * m).sum(1) / np.clip(m.sum(1), 1e-9, None)
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return (out / np.clip(n, 1e-9, None)).astype(np.float32)


def encoder() -> Encoder:
    global _ENCODER, LOAD_MS
    if _ENCODER is None:
        t0 = time.perf_counter()
        _ENCODER = Encoder()
        LOAD_MS = (time.perf_counter() - t0) * 1000
    return _ENCODER


# ── label representations ───────────────────────────────────────────────────

def _method_gloss(method_desc: str, method: str) -> str:
    """The `1. get_diff - Get the diff of a pull request.` clause, if present.

    `catalogue.py` truncates every parameter description at 240 chars, so only
    the first two or three methods of a nine-way dispatcher keep their gloss.
    The rest fall back to the method name in words, which is most of the signal
    anyway (`get_review_comments` -> "get review comments").
    """
    m = re.search(rf"\d+\.\s*{re.escape(method)}\s*[-—:]\s*([^.]+\.)", method_desc or "")
    return m.group(1).strip() if m else ""


def schema_text(label: str, catalogue: dict) -> str:
    tool, _, method = label.partition("::")
    spec = catalogue[tool]
    parts = [tool.replace("_", " ")]
    if method:
        # The method words lead and the gloss follows, because the tool's own
        # description is shared by all 9 `pull_request_read` targets and would
        # otherwise dominate the vector for every one of them.
        parts += [method.replace("_", " "),
                  _method_gloss(spec["params"]["method"]["description"], method)]
    parts += [spec["title"], spec["description"]]
    # Required parameter names are the one structural signal that survives
    # embedding — `pullNumber`, `issue_number`, `sha` name the object in play.
    parts.append(" ".join(re.sub(r"(?<!^)(?=[A-Z])", " ", p).replace("_", " ").lower()
                          for p in spec["required"] if p not in ("owner", "repo", "method")))
    return ". ".join(p for p in parts if p)


@lru_cache(maxsize=None)
def schema_matrix() -> tuple[tuple[str, ...], np.ndarray]:
    from catalogue import load
    cat = load("session")
    labs = tuple(labels(cat))
    return labs, encoder()([schema_text(x, cat) for x in labs])


@lru_cache(maxsize=None)
def centroid_matrix(split: str = "family_a") -> tuple[tuple[str, ...], np.ndarray]:
    """One vector per label: the mean of its training queries, renormalised.

    Off-topic rows have no label and contribute nothing, so nothing here models
    "not a tool call" — abstention has to come from the threshold.
    """
    labs, _ = schema_matrix()
    rows = [json.loads(x) for x in
            (HERE / "data" / f"{split}.jsonl").read_text().splitlines() if x.strip()]
    by = {}
    for r in rows:
        if r.get("label"):
            by.setdefault(r["label"], []).append(r["query"])
    flat = [q for x in labs for q in by.get(x, ())]
    V = encoder()(flat) if flat else np.zeros((0, DIM), np.float32)
    M, i = np.zeros((len(labs), DIM), np.float32), 0
    for j, x in enumerate(labs):
        n = len(by.get(x, ()))
        if n:
            M[j] = V[i:i + n].mean(0)
            i += n
        else:  # a label with no training query falls back to its schema text
            M[j] = schema_matrix()[1][j]
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return labs, (M / np.clip(n, 1e-9, None)).astype(np.float32)


# ── the arm ─────────────────────────────────────────────────────────────────

class EncoderArm(ArmBase):
    """Cosine nearest-label, with a threshold so it can abstain.

    Without the threshold this arm answers every off-topic row — the same
    giveaway the catch-all rule was priced at in `RESULTS.md` (abstention 0.867
    -> 0.000 for +0.014 accuracy).
    """

    def __init__(self, source: str = "schema", threshold: float = 0.0,
                 alpha: float = FUSION_ALPHA) -> None:
        self.source, self.threshold, self.alpha = source, threshold, alpha
        if source == "schema":
            self.labels_, self.M = schema_matrix()
            self.M2 = None
        elif source == "centroid":
            self.labels_, self.M = centroid_matrix()
            self.M2 = None
        elif source == "fusion":
            self.labels_, self.M = schema_matrix()
            self.M2 = centroid_matrix()[1]
        else:
            raise ValueError(f"unknown source {source!r}")
        # Per-instance, never module-level: a shared cache would make the second
        # arm in a process look 100x faster than the first.
        self._cache: dict[str, np.ndarray] = {}

    def embed(self, query: str) -> np.ndarray:
        v = self._cache.get(query)
        if v is None:
            v = self._cache[query] = encoder()([query])[0]
        return v

    def precompute(self, queries: list[str]) -> None:
        """Batch-fill the cache for a threshold sweep; latency is measured cold."""
        todo = [q for q in dict.fromkeys(queries) if q not in self._cache]
        if todo:
            for q, v in zip(todo, encoder()(todo)):
                self._cache[q] = v

    def raw_scores(self, query: str) -> np.ndarray:
        v = self.embed(query)
        s = self.M @ v
        if self.M2 is not None:
            s = self.alpha * s + (1 - self.alpha) * (self.M2 @ v)
        return s

    def score(self, query: str) -> list[tuple[str, float]]:
        s = self.raw_scores(query)
        return [(self.labels_[i], float(s[i])) for i in np.argsort(-s)]

    def route(self, query: str) -> str | None:
        s = self.raw_scores(query)
        i = int(np.argmax(s))
        return self.labels_[i] if s[i] >= self.threshold else None


# Thresholds swept on family A under the cascade's 0.70 abstention floor; these
# are the values that sweep chose, so `eval.py enc-fusion` reproduces the row.
#
# The threshold is the least transferable part of this arm. A cosine to a
# training centroid is calibrated to the phrasing family it was built from:
# 0.53 keeps coverage 0.998 on family A and 0.343 on family B, taking accuracy
# from 0.916 to 0.247 while the same arm at an open threshold scores 0.540.
# Fusion is far less brittle (0.999 -> 0.742) because half its score comes from
# schema text, which no split had a hand in writing.
THRESHOLDS = {"schema": 0.28, "centroid": 0.53, "fusion": 0.35}

for _src in ("schema", "centroid", "fusion"):
    register(f"enc-{_src}", lambda s=_src: EncoderArm(s, THRESHOLDS[s]))
    register(f"enc-{_src}-open", lambda s=_src: EncoderArm(s, -1.0))  # never abstains


# ── driver ──────────────────────────────────────────────────────────────────

SPLITS = ("family A (fitted)", "family B (held-out)", "wild (hand-authored)")


def _paths() -> dict[str, Path]:
    return {SPLITS[0]: HERE / "data" / "family_a.jsonl",
            SPLITS[1]: HERE / "data" / "family_b.jsonl",
            SPLITS[2]: HERE / "wild.jsonl"}


def sweep(arm: EncoderArm, rows: list[dict], grid: np.ndarray) -> list[dict]:
    """Coverage/precision/accuracy/abstention at every threshold, one encode pass."""
    on = [r for r in rows if r.get("label")]
    off = [r for r in rows if not r.get("label")]
    arm.precompute([r["query"] for r in rows])
    top_on = []
    for r in on:
        sc = arm.raw_scores(r["query"])
        i = int(np.argmax(sc))
        top_on.append((float(sc[i]), arm.labels_[i]))
    top_off = [float(arm.raw_scores(r["query"]).max()) for r in off]
    out = []
    for t in grid:
        ans = [(s, lab, r) for (s, lab), r in zip(top_on, on) if s >= t]
        hits = sum(lab == r["label"] for _, lab, r in ans)
        out.append({
            "threshold": round(float(t), 3),
            "coverage": round(len(ans) / max(len(on), 1), 4),
            "precision": round(hits / len(ans), 4) if ans else 0.0,
            "label_acc": round(hits / max(len(on), 1), 4),
            "abstain_acc": round(sum(s < t for s in top_off) / len(off), 4) if off else None,
        })
    return out


def main() -> int:
    from eval import load_split, score

    paths = _paths()
    rows = {k: load_split(v) for k, v in paths.items()}
    t0 = time.perf_counter()
    encoder()
    labs, _ = schema_matrix()
    setup_ms = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    centroid_matrix()
    centroid_ms = (time.perf_counter() - t0) * 1000

    out: dict = {"encoder_load_ms": round(LOAD_MS, 1),
                 "schema_setup_ms": round(setup_ms, 1),
                 "centroid_setup_ms": round(centroid_ms, 1),
                 "n_labels": len(labs), "model_dir": str(MODEL_DIR)}

    # Threshold picked on family A only — the split the fitted arms were fitted
    # on — under the same abstention floor the cascade is held to.
    # Wide grid: schema-text cosines run 0.1-0.4 (label text and a request are
    # different genres of English) while centroid cosines run 0.3-0.8, so one
    # hardcoded band would silently pin an arm at coverage 0 or 1.
    grid = np.round(np.arange(0.00, 0.90, 0.01), 3)
    print("threshold sweep on family A (selection split), abstention floor 0.70\n")
    hdr = f"{'source':<10}{'thr':>7}{'cov':>7}{'prec':>7}{'acc':>7}{'abst':>7}"
    print(hdr)
    print("-" * len(hdr))
    chosen = {}
    for src in ("schema", "centroid", "fusion"):
        arm = EncoderArm(src, -1.0)
        s = sweep(arm, rows[SPLITS[0]], grid)
        ok = [r for r in s if (r["abstain_acc"] or 0) >= 0.70]
        best = max(ok or s, key=lambda r: (r["label_acc"], r["threshold"]))
        chosen[src] = best["threshold"]
        out.setdefault("family_a_sweep", {})[src] = s
        print(f"{src:<10}{best['threshold']:>7.2f}{best['coverage']:>7.3f}"
              f"{best['precision']:>7.3f}{best['label_acc']:>7.3f}{best['abstain_acc']:>7.3f}")
    out["chosen_thresholds"] = chosen
    print()

    # Full sweep on the honest splits too, for the writeup's curve.
    for src in ("schema", "centroid", "fusion"):
        arm = EncoderArm(src, -1.0)
        for sname in SPLITS[1:]:
            out.setdefault("sweeps", {}).setdefault(sname, {})[src] = \
                sweep(arm, rows[sname], grid)

    # Alpha diagnostic: is 0.5 anywhere near right, and does it matter?
    print("fusion alpha (1.0 = schema only, 0.0 = centroid only), open threshold\n")
    ahdr = f"{'alpha':>7}" + "".join(f"{s.split(' ')[0] + s.split(' ')[1][:1]:>10}" for s in SPLITS)
    print(ahdr)
    print("-" * len(ahdr))
    for a in (0.0, 0.25, 0.5, 0.75, 1.0):
        arm = EncoderArm("fusion", -1.0, alpha=a)
        accs = []
        for sname in SPLITS:
            on = [r for r in rows[sname] if r.get("label")]
            arm.precompute([r["query"] for r in on])
            hits = sum(arm.route(r["query"]) == r["label"] for r in on)
            accs.append(hits / len(on))
        out.setdefault("alpha", {})[str(a)] = [round(x, 4) for x in accs]
        print(f"{a:>7.2f}" + "".join(f"{x:>10.3f}" for x in accs))
    print()

    # The comparable table: same `score()` eval.py runs for every other arm,
    # on fresh instances so per-query latency is a cold encode.
    hdr = (f"{'arm':<20}{'split':<22}{'cov':>7}{'prec':>7}{'acc':>7}"
           f"{'tool':>7}{'meth':>7}{'abst':>7}{'args':>7}{'ms':>9}")
    print(hdr)
    print("-" * len(hdr))
    for src in ("schema", "centroid", "fusion"):
        for tag, thr in ((f"enc-{src}", chosen[src]), (f"enc-{src}-open", -1.0)):
            for sname in SPLITS:
                s = score(EncoderArm(src, thr), rows[sname])
                s.pop("errors")
                out.setdefault("arms", {}).setdefault(tag, {})[sname] = s
                f = lambda k: "  -  " if s[k] is None else f"{s[k]:.3f}"
                print(f"{tag:<20}{sname:<22}{f('coverage'):>7}{f('precision'):>7}"
                      f"{f('label_acc'):>7}{f('tool_acc'):>7}"
                      f"{f('method_acc_given_tool'):>7}{f('abstain_acc'):>7}"
                      f"{f('args_acc'):>7}{s['median_latency_ms']:>9.4f}")
            print()

    print(f"encoder load {out['encoder_load_ms']:.0f} ms, "
          f"79 label texts {out['schema_setup_ms']:.0f} ms, "
          f"948 training queries {out['centroid_setup_ms']:.0f} ms")
    (HERE / "results_encoder.json").write_text(json.dumps(out, indent=1) + "\n")
    print("wrote results_encoder.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
