"""Perform the piece bar by bar: one Jev call per bar, five Choice questions (one per layer).

Modes: sampled (draw from Jev's probabilities, options under 0.05 dropped), argmax, random (no Jev).
Each bar's state carries the arc position and intent plus the last four bars' choices, so a run is
sequential. Logs go to runs/<mode>-<seed>.json; metrics to results/metrics.json.
"""
import json, random, statistics as st, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jev-tag-encoder"))
import jev  # noqa: E402
from hymnal import LAYERS, N_BARS, section, energy, PROGRESSIONS, PHRASE  # noqa: E402

VARIANT = "v2"  # v1: a chord per bar; v2: a progression per 4-bar phrase

QUESTION = {
    "harmony": "Which chord should the next bar use?",
    "drums": "Which drum pattern should play in the next bar?",
    "bass": "Which bass pattern should play in the next bar?",
    "keys": "Which keys pattern should play in the next bar?",
    "lead": "Which melody pattern should play in the next bar?",
}
QS = {l: {"type": "choice", "instructions": QUESTION[l], "criteria": {k: v[0] for k, v in opts.items()}}
      for l, opts in LAYERS.items()}
QS_PROG = {"type": "choice", "instructions": "Which chord progression should the next four bars follow?",
           "criteria": {k: v[0] for k, v in PROGRESSIONS.items()}}


def questions(bar):
    if VARIANT == "v1":
        return QS
    qs = {l: q for l, q in QS.items() if l != "harmony"}
    if (bar - 1) % PHRASE == 0:
        qs["progression"] = QS_PROG
    return qs


def line(bar, c):
    return f"bar {bar}: " + ", ".join(f"{l} {c[l]}" for l in LAYERS)


def state(bar, history):
    name, intent, _, i, n = section(bar)
    if VARIANT == "v2" and (bar - 1) % PHRASE == 0:
        phrase_end = min(bar + PHRASE - 1, N_BARS)
        spans = sorted({section(b)[0] for b in range(bar, phrase_end + 1)}, key=lambda x: [a[2] for a in __import__("hymnal").ARC].index(x))
        extra = f" The next four bars (bars {bar}-{phrase_end}) cover: {', '.join(spans)}."
    else:
        extra = ""
    nxt = section(bar + 1)[0] if bar < N_BARS else "end of the piece"
    return {
        "piece": f"A {N_BARS}-bar instrumental in A minor at 120 bpm, played live one bar at a time.",
        "position": f"Next is bar {bar} of {N_BARS}: bar {i} of {n} of the {name}. After this section comes: {nxt}.{extra}",
        "section_intent": intent,
        "recent_bars": "\n".join(line(b, c) for b, c in history[-4:]) or "Nothing has played yet.",
    }


def pick(probs, mode, rng):
    if mode == "argmax":
        return max(probs, key=probs.get)
    items = [(k, p) for k, p in probs.items() if p >= 0.05] or list(probs.items())
    r, acc = rng.random() * sum(p for _, p in items), 0.0
    for k, p in items:
        acc += p
        if r <= acc:
            return k
    return items[-1][0]


def run(mode, seed):
    rng, hist, log, prog = random.Random(seed), [], [], None
    for bar in range(1, N_BARS + 1):
        rec = {"bar": bar, "section": section(bar)[0]}
        qs = questions(bar)
        if mode == "random":
            choice = {l: rng.choice(list(q["criteria"])) for l, q in qs.items()}
        else:
            res, dt, _ = jev.post(state(bar, hist), qs)
            ans = res["answers"]
            rec.update(latency=dt, in_tok=res.get("usage", {}).get("input_tokens"),
                       probs={l: ans[l]["probabilities"] for l in qs},
                       conf={l: ans[l].get("confidence") for l in qs})
            choice = {l: pick(ans[l]["probabilities"], mode, rng) for l in qs}
        if "progression" in choice:
            prog = choice.pop("progression"); rec["progression"] = prog
        if VARIANT == "v2":
            choice["harmony"] = PROGRESSIONS[prog][1][(bar - 1) % PHRASE]
            rec["phrase_progression"] = prog
        rec["choice"] = choice
        hist.append((bar, choice)); log.append(rec)
    return log


def metrics(log):
    e = [energy(r["choice"]) for r in log]
    t = [section(r["bar"])[2] for r in log]
    h = [r["choice"]["harmony"] for r in log]
    after_e = [h[i + 1] for i in range(len(h) - 1) if h[i] == "E"]
    hook = [r["section"] for r in log if r["choice"]["lead"] == "hook"]
    same = sum(log[i]["choice"] == log[i - 1]["choice"] for i in range(1, len(log)))
    lat = [r["latency"] for r in log if "latency" in r]
    return {
        "arc_r": round(st.correlation(e, t), 3) if len(set(e)) > 1 else None,
        "arc_mae": round(sum(abs(a - b) for a, b in zip(e, t)) / len(e), 3),
        "ends_on_Am": h[-1] == "Am",
        "E_resolves_to_Am": f"{after_e.count('Am')}/{len(after_e)}",
        "chords_used": len(set(h)),
        "progressions": [r["progression"] for r in log if "progression" in r],
        "hook_in_drop_or_finale": f"{sum(s in ('drop', 'finale') for s in hook)}/{len(hook)}",
        "bars_identical_to_previous": same,
        "latency_p50": round(st.median(lat), 3) if lat else None,
        "latency_max": round(max(lat), 3) if lat else None,
    }


if __name__ == "__main__":
    VARIANT = sys.argv[1] if len(sys.argv) > 1 else VARIANT
    (HERE / "runs").mkdir(exist_ok=True); (HERE / "results").mkdir(exist_ok=True)
    plan = [("sampled", s) for s in (1, 2, 3)] + [("argmax", 0)] + [("random", s) for s in range(1, 21)]
    out = {}
    for mode, seed in plan:
        path = HERE / "runs" / f"{VARIANT}-{mode}-{seed}.json"
        log = json.loads(path.read_text()) if path.exists() else run(mode, seed)
        path.write_text(json.dumps(log, indent=0))
        out[f"{VARIANT}-{mode}-{seed}"] = m = metrics(log)
        if mode != "random":
            print(mode, seed, m, flush=True)
    rnd = [v for k, v in out.items() if "-random-" in k]
    out["random_summary"] = {"arc_r_mean": round(st.mean(v["arc_r"] for v in rnd), 3),
                             "arc_mae_mean": round(st.mean(v["arc_mae"] for v in rnd), 3),
                             "ends_on_Am": sum(v["ends_on_Am"] for v in rnd), "n": len(rnd)}
    print("random", out["random_summary"])
    (HERE / "results" / f"metrics-{VARIANT}.json").write_text(json.dumps(out, indent=1))
