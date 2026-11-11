"""Held-out comparison of a tagged decide.py run against the baseline angles.

    python3 run/compare.py v030stack [--split test]   # -> results/prompts/compare_<tag>_<split>.json

Arms, each scored on the answers every arm covers:
  plain           the old single call (one plain sample, k=1 run)
  old stack       plain + literal + sceptic single runs, pooled (no option shuffle)
  new single      sample 0 of the tagged run: one plain sample under the current system prompt
  new stack       the tagged run's own pool: decide.py's k samples with shuffled options
Accuracy deltas are against `plain`, with a paired bootstrap 90% interval.
A second block adds the 0.1.x and 0.2.0 arms from results/v020__<set>__{old,stack}.jsonl on the answers they
share (sections there are a 150-item sample): `0.1 default` is three plain samples at the CLI's default effort,
`0.1 single` its first sample, `0.2.0 stack` the cue-less three-angle stack run through decide.py.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import prompts as P  # noqa: E402


def load_v020(s, arm, split):
    f = P.HERE / "results" / f"v020__{s}__{arm}.jsonl"
    if not f.exists():
        return {}
    rows = [json.loads(x) for x in f.read_text().splitlines()]
    return {(r["id"], r["name"]): r for r in rows if "raw" in r and (split == "all" or P.split_of(r["id"]) == split)}


def block(title, keys, arms, lab, out):
    m0 = P.metrics([(p, lab[k]) for p, k in zip(arms["plain"], keys)])
    print(title)
    for name, dists in arms.items():
        m = P.metrics([(p, lab[k]) for p, k in zip(dists, keys)])
        lo, hi = P.boot_diff(m["right"], m0["right"]) if name != "plain" else (0.0, 0.0)
        out[name] = {k: v for k, v in m.items() if k != "right"} | {"d_acc": m["acc"] - m0["acc"], "ci90": [lo, hi]}
        print(f"  {name:12s} acc {m['acc']:.3f} ({m['acc'] - m0['acc']:+.3f} [{lo:+.3f},{hi:+.3f}]) "
              f"ece {m['ece']:.3f} auroc {m['auroc'] or 0:.3f} brier {m['brier']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    out = {}
    for s in P.SETS:
        base = {n: P.load(s, n, a.split) for n in ("plain", "literal", "sceptic")}
        new = P.load(s, a.tag, a.split)
        if not new or not all(base.values()):
            continue
        keys = sorted(set(new).intersection(*map(set, base.values())))
        lab = {k: new[k]["label"] for k in keys}
        arms = {
            "plain": [base["plain"][k]["raw"] for k in keys],
            "old stack": [P.pool([base[n][k]["raw"] for n in base]) for k in keys],
            "new single": [new[k]["per_sample"][0] for k in keys],
            "new stack": [new[k]["raw"] for k in keys],
        }
        ids = {k[0]: k for k in keys}
        lat = sorted(new[k]["usage"]["latency_ms"] for k in ids.values())
        out[s] = {}
        block(f"== {s} ({a.split}, {len(keys)} answers, {len(ids)} items; new stack p50 {lat[len(lat) // 2]} ms)",
              keys, arms, lab, out[s])
        old, st = load_v020(s, "old", a.split), load_v020(s, "stack", a.split)
        shared = [i for i, k in enumerate(keys) if k in old and k in st]
        if shared:
            ks = [keys[i] for i in shared]
            arms2 = {n: [d[i] for i in shared] for n, d in arms.items()}
            arms2["0.1 default"] = [old[k]["raw"] for k in ks]
            arms2["0.1 single"] = [old[k]["per_sample"][0] for k in ks]
            arms2["0.2.0 stack"] = [st[k]["raw"] for k in ks]
            out[s + " vs v020"] = {}
            block(f"-- {s} with the v020 arms ({len(ks)} answers)", ks, arms2, lab, out[s + " vs v020"])
    print("== mean over sets")
    for name in ("plain", "old stack", "new single", "new stack"):
        v = [out[s][name] for s in out if "v020" not in s]
        print(f"  {name:12s} " + " ".join(f"{k} {sum(x[k] or 0 for x in v) / len(v):.3f}" for k in ("acc", "ece", "auroc", "brier")))
    (P.OUT / f"compare_{a.tag}_{a.split}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
