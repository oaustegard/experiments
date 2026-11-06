"""Paired test of the ±31 refits against the shipped spikes on ±63, decision by decision.

    python3 -I run/refit31_paired.py shipped     # per-decision correctness of the shipped spikes on ±63 (cached)
    python3 -I run/refit31_paired.py compare     # the same for the refits in out/base31/spikes on ±31, then the test

Registered 2026-11-06, before the round-3 refits were scored (Oskar chose option 3). Each spike's test
decisions are scored by both systems, so the difference is paired: d_i = ok_refit_i - ok_shipped_i. Its
mean is the accuracy change; its 95% interval comes from 10,000 bootstrap resamples of the decisions
(seed 0), resampling whole cases so a case's questions stay together.

Decision rule: ship the ±31 base with the seven refit spikes if
  (a) no spike is detectably worse: every spike's 95% upper bound is at or above 0, and
  (b) the change pooled over all spikes' decisions is at or above 0.
Otherwise keep ±63. This replaces the fixed 0.005 threshold, which the test sets (500 to 987 decisions per
spike, standard error about 0.015) could not resolve.

Writes out/base31/paired_shipped.json and results/refit31_paired.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import stock_path  # noqa: E402
from pack_system import WEB, LoraRef, everyday_cases  # noqa: E402
from shared import BD, ROOT, examples, load_cases  # noqa: E402
sys.path.insert(0, str(BD / "run"))
from train_enc import Enc  # noqa: E402

SPIKES = {"workflows": ["agent_trace_observability", "customer_service", "invoice_processing", "security_incidents"],
          "everyday": ["transactions", "emails", "reviews"]}
B31 = ROOT / "out" / "base31"


def per_decision(ref, tok, sm, sb, cases):
    """[(case_id, ok)] per test decision, ok = top option matches gold's top option (the dataset's scorer)."""
    ref.use(sm, sb)
    out = []
    for e in examples(cases):
        ids, pos = tok.build(e)
        start = ids.index(tok.sep, pos[-1]) + 1
        z, _ = ref.run(ids, pos, (start, len(ids) - 1))
        out.append((e["case"], int(int(np.argmax(z)) == int(np.argmax(e["gold"])))))   # temperature does not move the argmax
    return out


def cases_of(set_, w):
    return load_cases(w, "test") if set_ == "workflows" else everyday_cases(w, "test")


def boot(cases, d, n=10000, seed=0):
    """95% interval of mean(d) resampling whole cases."""
    ids = sorted(set(cases))
    idx = {c: [i for i, x in enumerate(cases) if x == c] for c in ids}
    sums = np.array([d[idx[c]].sum() for c in ids]); cnts = np.array([len(idx[c]) for c in ids])
    r = np.random.default_rng(seed)
    pick = r.integers(0, len(ids), size=(n, len(ids)))
    means = sums[pick].sum(1) / cnts[pick].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main():
    mode = sys.argv[1]
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    if mode == "shipped":
        ref = LoraRef(json.loads((WEB / "base" / "manifest.json").read_text()), (WEB / "base" / "weights.bin").read_bytes())
        res = {}
        for set_, ws in SPIKES.items():
            for w in ws:
                sm = json.loads((WEB / set_ / "spikes" / f"{w}.json").read_text()); sb = (WEB / set_ / "spikes" / f"{w}.bin").read_bytes()
                res[w] = per_decision(ref, tok, sm, sb, cases_of(set_, w))
                print(f"shipped ±63 {w}: {np.mean([o for _, o in res[w]]):.4f} ({len(res[w])})", flush=True)
        (B31 / "paired_shipped.json").write_text(json.dumps(res))
        return
    shipped = json.loads((B31 / "paired_shipped.json").read_text())
    ref = LoraRef(json.loads((B31 / "manifest.json").read_text()), (B31 / "weights.bin").read_bytes())
    out, allc, alld = {"rule": __doc__.split("Decision rule: ")[1].split("\n\nWrites")[0], "spikes": {}}, [], []
    for set_, ws in SPIKES.items():
        for w in ws:
            sm = json.loads((B31 / "spikes" / set_ / f"{w}.json").read_text()); sb = (B31 / "spikes" / set_ / f"{w}.bin").read_bytes()
            new = per_decision(ref, tok, sm, sb, cases_of(set_, w))
            old = shipped[w]
            assert [c for c, _ in new] == [c for c, _ in old], w
            cs = [c for c, _ in new]
            d = np.array([n - o for (_, n), (_, o) in zip(new, old)], dtype=float)
            lo, hi = boot(cs, d)
            out["spikes"][w] = {"n": len(d), "shipped_L63": float(np.mean([o for _, o in old])), "refit_L31": float(np.mean([n for _, n in new])),
                                "change": float(d.mean()), "ci95": [lo, hi], "better": int((d > 0).sum()), "worse": int((d < 0).sum()),
                                "refit": sm.get("refit")}
            allc += [f"{w}/{c}" for c in cs]; alld += list(d)
            print(f"{w}: {out['spikes'][w]['shipped_L63']:.4f} -> {out['spikes'][w]['refit_L31']:.4f}  change {d.mean():+.4f}  95% [{lo:+.4f}, {hi:+.4f}]"
                  f"  ({out['spikes'][w]['better']} better, {out['spikes'][w]['worse']} worse)", flush=True)
    alld = np.array(alld); lo, hi = boot(allc, alld)
    out["pooled"] = {"n": len(alld), "change": float(alld.mean()), "ci95": [lo, hi]}
    out["ship"] = all(v["ci95"][1] >= 0 for v in out["spikes"].values()) and out["pooled"]["change"] >= 0
    (ROOT / "results" / "refit31_paired.json").write_text(json.dumps(out, indent=1) + "\n")
    print(f"pooled change {alld.mean():+.4f} 95% [{lo:+.4f}, {hi:+.4f}]; ship ±31: {out['ship']}")


if __name__ == "__main__":
    main()
