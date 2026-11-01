#!/usr/bin/env python3
"""Summarise one force-touch run: which touch and motion channels the phone actually reports, and how well each one
tells a light press from a firm one.

    python3 force-touch/analyze.py force-touch/results/raw-ROOM.jsonl

Each press becomes one number per channel (the median over the hold, ignoring the first and last 150 ms, which are
landing and lifting). A channel that never varies across presses is reported as unsupported. Separation is the AUC of
light vs firm (0.5 = no information, 1.0 = every firm press above every light one, 0.0 = the reverse); the posture
check repeats it for flat-light vs tip-firm, where contact area and force pull in opposite directions.
Writes results/summary-ROOM.json next to the raw file.
"""
import json, sys, statistics as st
from pathlib import Path

def auc(lo, hi):
    """P(random hi > random lo), ties counted half: the Mann-Whitney U over n1*n2."""
    if not lo or not hi: return None
    wins = sum((h > l) + 0.5 * (h == l) for h in hi for l in lo)
    return round(wins / (len(lo) * len(hi)), 3)

def hold(samples, t0, t1, col):
    # lift-off events (pointerup/cancel, touchend) report pressure 0 and default sizes: leave them out
    samples = [s for s in samples if s[-1] not in ("u", "c", "e")]
    vals = [s[col] for s in samples if s[col] is not None and t0 + 150 <= s[0] <= t1 - 150]
    if not vals: vals = [s[col] for s in samples if s[col] is not None]
    return st.median(vals) if vals else None

def motion_features(m, t0, t1):
    pre = [r for r in m if r[0] < t0 and r[3] is not None]
    dur = [r for r in m if t0 + 150 <= r[0] <= t1 - 150 and r[3] is not None]
    onset = [r for r in m if t0 <= r[0] <= t0 + 300]
    out = {}
    if pre and dur:   # how far the phone sits back (z, gravity removed) while held, against just before the press
        out["motion_dz"] = st.mean(r[3] for r in dur) - st.mean(r[3] for r in pre)
        out["gravity_dz"] = (st.mean(r[6] for r in dur if r[6] is not None) - st.mean(r[6] for r in pre if r[6] is not None)) if all(r[6] is not None for r in pre + dur) else None
    if onset:
        out["onset_jolt"] = max((r[1] ** 2 + r[2] ** 2 + r[3] ** 2) ** 0.5 for r in onset if None not in r[1:4]) if any(None not in r[1:4] for r in onset) else None
        rr = [r for r in onset if None not in r[7:10]]
        out["onset_rotation"] = max((r[7] ** 2 + r[8] ** 2 + r[9] ** 2) ** 0.5 for r in rr) if rr else None
    return out

def main(raw):
    raw = Path(raw)
    recs = [json.loads(l) for l in raw.read_text().splitlines() if l.strip()]
    presses = {r["i"]: r for r in recs if r.get("t") == "ft-press"}          # a resend replaces the earlier copy
    dev = next((r["dev"] for r in reversed(recs) if r.get("t") in ("ft-hello", "ft-done") and r.get("dev")), {})
    rows = []
    for i in sorted(presses):
        p = presses[i]; t0, t1 = p["t0"], p["t1"]
        f = {"i": i, "cond": p["cond"], "ms": t1 - t0, "pointerType": p.get("pointerType")}
        for name, col in (("pressure", 1), ("width", 2), ("height", 3)): f[name] = hold(p["ptr"], t0, t1, col)
        for name, col in (("force", 1), ("radiusX", 2), ("radiusY", 3)): f[name] = hold(p["tch"], t0, t1, col)
        f["ptr_area"] = f["width"] * f["height"] if f["width"] is not None and f["height"] is not None else None
        f["touch_area"] = f["radiusX"] * f["radiusY"] if f["radiusX"] is not None and f["radiusY"] is not None else None
        f.update(motion_features(p["mot"], t0, t1))
        rows.append(f)
    channels = ["pressure", "width", "height", "ptr_area", "force", "radiusX", "radiusY", "touch_area", "motion_dz", "gravity_dz", "onset_jolt", "onset_rotation"]
    by = lambda cond, ch: [r[ch] for r in rows if r["cond"] == cond and r.get(ch) is not None]
    summary = {"device": dev, "presses": len(rows), "channels": {}}
    print(f"{len(rows)} presses from {dev.get('ua', 'unknown device')}\n")
    print(f"{'channel':15} {'varies':7} {'light':>18} {'medium':>18} {'firm':>18} {'AUC firm>light':>15} {'AUC tip>flat':>13}")
    for ch in channels:
        vals = [r[ch] for r in rows if r.get(ch) is not None]
        varies = len(vals) > 1 and max(vals) - min(vals) > 1e-6
        fmt = lambda xs: f"{st.mean(xs):.3f}±{st.pstdev(xs):.3f}" if xs else "–"
        c = {"reported": bool(vals), "varies": varies, "values": sorted(set(round(v, 4) for v in vals))[:6] if not varies else None,
             **{k: {"n": len(by(k, ch)), "mean": st.mean(by(k, ch)) if by(k, ch) else None} for k in ("light", "medium", "firm", "flat-light", "tip-firm")},
             "auc_firm_vs_light": auc(by("light", ch), by("firm", ch)) if varies else None,
             "auc_tip_firm_vs_flat_light": auc(by("flat-light", ch), by("tip-firm", ch)) if varies else None}
        summary["channels"][ch] = c
        print(f"{ch:15} {'yes' if varies else 'NO':7} {fmt(by('light', ch)):>18} {fmt(by('medium', ch)):>18} {fmt(by('firm', ch)):>18} "
              f"{str(c['auc_firm_vs_light']):>15} {str(c['auc_tip_firm_vs_flat_light']):>13}" + ("" if varies else f"   constant {c['values']}"))
    summary["rows"] = rows
    out = raw.with_name(raw.name.replace("raw-", "summary-").replace(".jsonl", ".json"))
    out.write_text(json.dumps(summary, indent=1))
    print(f"\nwrote {out}")

if __name__ == "__main__":
    main(sys.argv[1])
