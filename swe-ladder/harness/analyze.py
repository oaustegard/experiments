"""Tables for RESULTS.md from the graded runs and their cost files.

    python harness/analyze.py   -> stdout (markdown) and data/analysis.json

Runs: r1-final (rung 1, Haiku; r1-haiku with the 30 redone tasks replaced),
r2-haiku and r2-sonnet (rung 2 on r1-final's failures), r1-sonnet-cold (Sonnet
with issue text only, on the rung-1 failures plus 50 rung-1 successes drawn
with random.Random(0)). Cost is input-side dollars per spawn (cost.py). Rung-1
cost per task is the attempt that was graded: a redone task counts its redo,
and the discarded first attempts are reported separately.
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import DATA  # noqa: E402

RUNS = DATA / "runs"


def rows(run: str, name: str) -> dict[str, dict]:
    f = RUNS / run / name
    return {r["instance_id"]: r for r in map(json.loads, f.open())} if f.exists() else {}


def repo(iid: str) -> str:
    return iid.split("__")[0]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (math.nan, math.nan)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def pct(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.1f}%, 95% CI {100 * lo:.0f}–{100 * hi:.0f})" if n else "0/0"


def usd(cost: dict[str, dict], ids) -> float:
    return sum(cost[i]["usd_input_side"] for i in ids if i in cost)


def main() -> None:
    r1 = rows("r1-final", "results.jsonl")
    c1 = {**rows("r1-haiku", "costs.jsonl"), **rows("r1-haiku-redo", "costs.jsonl")}
    c1_discarded = {i: r for i, r in rows("r1-haiku", "costs.jsonl").items()
                    if i in rows("r1-haiku-redo", "costs.jsonl")}
    rh, rs = rows("r2-haiku", "results.jsonl"), rows("r2-sonnet", "results.jsonl")
    ch, cs = rows("r2-haiku", "costs.jsonl"), rows("r2-sonnet", "costs.jsonl")
    cold, cc = rows("r1-sonnet-cold", "results.jsonl"), rows("r1-sonnet-cold", "costs.jsonl")
    cold_list = json.loads((DATA / "cold-ids.json").read_text())

    ok = lambda r, i: r.get(i, {}).get("resolved") is True  # noqa: E731
    out, A = [], {}
    n1 = len(r1)
    s1 = [i for i in r1 if ok(r1, i)]
    fail1 = sorted(i for i in r1 if not ok(r1, i))
    A["r1"] = {"n": n1, "resolved": len(s1), "errors": sum(r1[i].get("resolved") is None for i in r1),
               "usd": usd(c1, r1), "usd_discarded": usd(c1_discarded, c1_discarded),
               "by_repo": {p: [sum(ok(r1, i) for i in r1 if repo(i) == p), sum(repo(i) == p for i in r1)]
                           for p in sorted({repo(i) for i in r1})}}
    out.append("## Rung 1 (Haiku, issue text only)\n")
    out.append(f"- resolved {pct(len(s1), n1)}; grading errors {A['r1']['errors']}")
    for p, (k, n) in A["r1"]["by_repo"].items():
        out.append(f"- {p}: {pct(k, n)}")
    out.append(f"- input-side ${A['r1']['usd']:.2f} over {len(c1)} graded spawns "
               f"(mean ${A['r1']['usd'] / max(1, n1):.4f}); discarded first attempts of redone tasks "
               f"${A['r1']['usd_discarded']:.2f}")

    # Rung 2: paired comparison on the same failures.
    both = [i for i in fail1 if ok(rh, i) and ok(rs, i)]
    h_only = [i for i in fail1 if ok(rh, i) and not ok(rs, i)]
    s_only = [i for i in fail1 if ok(rs, i) and not ok(rh, i)]
    neither = [i for i in fail1 if not ok(rh, i) and not ok(rs, i)]
    kh, ks = len(both) + len(h_only), len(both) + len(s_only)
    A["r2"] = {"n": len(fail1), "haiku": kh, "sonnet": ks, "both": both, "haiku_only": h_only,
               "sonnet_only": s_only, "neither": neither, "mcnemar_p": mcnemar_exact(len(h_only), len(s_only)),
               "usd_haiku": usd(ch, fail1), "usd_sonnet": usd(cs, fail1),
               "turns_haiku": sum(ch[i]["turns"] for i in fail1 if i in ch) / max(1, len(ch)),
               "turns_sonnet": sum(cs[i]["turns"] for i in fail1 if i in cs) / max(1, len(cs))}
    out.append("\n## Rung 2 (on the rung-1 failures, with failing tests and tracebacks)\n")
    out.append(f"- Haiku retry {pct(kh, len(fail1))}; Sonnet escalation {pct(ks, len(fail1))}")
    out.append(f"- both {len(both)}, Haiku only {len(h_only)} {h_only}, Sonnet only {len(s_only)} {s_only}, "
               f"neither {len(neither)} {neither}; exact McNemar p = {A['r2']['mcnemar_p']:.3f}")
    out.append(f"- cost: Haiku ${A['r2']['usd_haiku']:.2f} (mean ${A['r2']['usd_haiku'] / len(fail1):.4f}, "
               f"{A['r2']['turns_haiku']:.1f} turns), Sonnet ${A['r2']['usd_sonnet']:.2f} "
               f"(mean ${A['r2']['usd_sonnet'] / len(fail1):.4f}, {A['r2']['turns_sonnet']:.1f} turns), "
               f"ratio {A['r2']['usd_sonnet'] / max(1e-9, A['r2']['usd_haiku']):.1f}×")
    for p in sorted({repo(i) for i in fail1}):
        ids = [i for i in fail1 if repo(i) == p]
        out.append(f"- {p}: Haiku {sum(ok(rh, i) for i in ids)}/{len(ids)}, "
                   f"Sonnet {sum(ok(rs, i) for i in ids)}/{len(ids)}")

    # Whole ladder over all rung-1 tasks.
    out.append("\n## Ladder totals (all rung-1 tasks)\n")
    for arm, k, c in (("Haiku → Haiku", kh, A["r2"]["usd_haiku"]), ("Haiku → Sonnet", ks, A["r2"]["usd_sonnet"])):
        tot = len(s1) + k
        spend = A["r1"]["usd"] + c
        A.setdefault("ladder", {})[arm] = {"resolved": tot, "usd": spend}
        out.append(f"- {arm}: {pct(tot, n1)}; ${spend:.2f}; ${spend / tot:.4f} per resolved task; "
                   f"rung 2 ${c / max(1, k):.4f} per extra task resolved")
    dk = ks - kh
    dc = A["r2"]["usd_sonnet"] - A["r2"]["usd_haiku"]
    out.append(f"- marginal: Sonnet buys {dk} more resolved tasks for ${dc:.2f} "
               f"(${dc / dk:.2f} each)" if dk else f"- marginal: no difference in resolved tasks for ${dc:.2f}")

    # Sonnet cold. cold-ids.json is one list; its strata come from the rung-1 grade.
    cold_ids = {"failures": [i for i in cold_list if not ok(r1, i)],
                "successes": [i for i in cold_list if ok(r1, i)]}
    cf = [i for i in cold_ids["failures"] if i in cold]
    csu = [i for i in cold_ids["successes"] if i in cold]
    kf, ksu = sum(ok(cold, i) for i in cf), sum(ok(cold, i) for i in csu)
    A["cold"] = {"failures": [kf, len(cf), len(cold_ids["failures"])],
                 "successes": [ksu, len(csu), len(cold_ids["successes"])],
                 "regressions": [i for i in csu if not ok(cold, i)],
                 "usd": usd(cc, cc), "spawns": len(cc)}
    out.append("\n## Sonnet cold (issue text only)\n")
    out.append(f"- graded {len(cf)}/{len(cold_ids['failures'])} failures, "
               f"{len(csu)}/{len(cold_ids['successes'])} successes")
    out.append(f"- on rung-1 failures: {pct(kf, len(cf)) if cf else 'n/a'}; Sonnet escalation on the same "
               f"tasks {sum(ok(rs, i) for i in cf)}/{len(cf)}")
    if cf:
        e_only = [i for i in cf if ok(rs, i) and not ok(cold, i)]
        c_only = [i for i in cf if ok(cold, i) and not ok(rs, i)]
        A["cold"]["vs_escalation"] = {"escalation_only": e_only, "cold_only": c_only,
                                      "mcnemar_p": mcnemar_exact(len(e_only), len(c_only))}
        out.append(f"  - escalation only {e_only}, cold only {c_only}, "
                   f"p = {A['cold']['vs_escalation']['mcnemar_p']:.3f}")
        h_only = [i for i in cf if ok(rh, i) and not ok(cold, i)]
        c_only = [i for i in cf if ok(cold, i) and not ok(rh, i)]
        A["cold"]["vs_haiku_retry"] = {"haiku_retry": sum(ok(rh, i) for i in cf), "haiku_only": h_only,
                                       "cold_only": c_only, "mcnemar_p": mcnemar_exact(len(h_only), len(c_only))}
        out.append(f"- Haiku retry on the same tasks {A['cold']['vs_haiku_retry']['haiku_retry']}/{len(cf)}: "
                   f"Haiku only {h_only}, cold only {c_only}, "
                   f"p = {A['cold']['vs_haiku_retry']['mcnemar_p']:.3f}")
    out.append(f"- on rung-1 successes: {pct(ksu, len(csu)) if csu else 'n/a'}; regressions "
               f"{A['cold']['regressions']}")
    if cc:
        m = A["cold"]["usd"] / len(cc)
        out.append(f"- cost ${A['cold']['usd']:.2f} over {len(cc)} spawns (mean ${m:.4f}, "
                   f"{sum(r['turns'] for r in cc.values()) / len(cc):.1f} turns)")
        if cf and csu:
            # Stratified estimate of Sonnet on every task: successes stratum scaled from its sample.
            est = len(s1) * ksu / len(csu) + len(fail1) * kf / len(cf)
            A["cold"]["sonnet_everywhere"] = {"resolved_est": est, "usd_est": m * n1}
            out.append(f"- Sonnet on all {n1} tasks, stratified estimate: {est:.0f} resolved "
                       f"({100 * est / n1:.1f}%), ~${m * n1:.2f}")

    # Haiku re-roll: a fresh rung-1 attempt on the failures, no feedback. Separates
    # "a second attempt" from "the failing tests' output" in the rung-2 Haiku arm.
    rr, cr = rows("r1-haiku-reroll", "results.jsonl"), rows("r1-haiku-reroll", "costs.jsonl")
    if rr:
        ids = [i for i in fail1 if i in rr]
        k = sum(ok(rr, i) for i in ids)
        fb_only = [i for i in ids if ok(rh, i) and not ok(rr, i)]
        rr_only = [i for i in ids if ok(rr, i) and not ok(rh, i)]
        A["reroll"] = {"resolved": k, "n": len(ids), "feedback_only": fb_only, "reroll_only": rr_only,
                       "mcnemar_p": mcnemar_exact(len(fb_only), len(rr_only)), "usd": usd(cr, cr),
                       "cold_sonnet_same": sum(ok(cold, i) for i in ids if i in cold)}
        out.append("\n## Haiku re-roll (issue text only, on the rung-1 failures)\n")
        out.append(f"- {pct(k, len(ids))}; Haiku retry with feedback on the same tasks "
                   f"{sum(ok(rh, i) for i in ids)}/{len(ids)}; Sonnet cold "
                   f"{A['reroll']['cold_sonnet_same']}/{sum(i in cold for i in ids)}")
        out.append(f"- feedback only {fb_only}, re-roll only {rr_only}, p = {A['reroll']['mcnemar_p']:.3f}")
        if cr:
            out.append(f"- cost ${A['reroll']['usd']:.2f} over {len(cr)} spawns")

    (DATA / "analysis.json").write_text(json.dumps(A, indent=1, default=str))
    print("\n".join(out))


if __name__ == "__main__":
    main()
