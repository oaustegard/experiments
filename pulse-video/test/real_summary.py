#!/usr/bin/env python3
"""Headline numbers from a real-camera Copy JSON (results/real-*.json, compacted by hand) into results/real-summary.json."""
import json, statistics as S
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"
d = json.loads((R / "real-mac-1.json").read_text())
rows = [dict(zip(d["columns"], r)) for r in d["log"]]
# phases by Oskar's account and the face's brightness range: seated, jumping jacks (face moves), seated recovery
phase = lambda t: "rest" if t < 43 else "exercise" if t < 69 else "recovery"
still = lambda r: r["faceDrift"] < 0.05
out = {"windows": len(rows)}
for p in ("rest", "exercise", "recovery"):
    w = [r for r in rows if phase(r["t"]) == p]
    st = [r for r in w if still(r)]
    agree = [abs(r["posBpm"] - r["evmBpm"]) for r in st if r["posSnr"] >= 0 and r["evmSnr"] >= 0]
    out[p] = {
        "windows": len(w), "stillWindows": len(st),
        "posPassFrac": round(sum(r["posSnr"] >= 0 for r in w) / len(w), 2),
        "posSnrMedian": round(S.median(r["posSnr"] for r in w), 2),
        "evmSnrMedian": round(S.median(r["evmSnr"] for r in w), 2),
        "posBpmMedianPassing": round(S.median([r["posBpm"] for r in w if r["posSnr"] >= 0] or [0]), 1) or None,
        "bothPassWindows": len(agree), "posEvmAgreeMedian": round(S.median(agree), 2) if agree else None,
        "faceDriftMedian": round(S.median(r["faceDrift"] for r in w), 3),
        "bgBpmMedian": round(S.median(r["bgBpm"] for r in w), 1),
        "bgSnrMedian": round(S.median(r["bgSnr"] for r in w), 2),
        "bgPassFrac": round(sum(r["bgSnr"] >= 0 for r in w) / len(w), 2),
    }
ex = [r for r in rows if 49 <= r["t"] <= 56]
out["exercise"]["bgBpmJacks"] = round(S.median(r["bgBpm"] for r in ex), 1)
out["exercise"]["posBpmJacks"] = round(S.median(r["posBpm"] for r in ex), 1)
rec = [r for r in rows if r["t"] >= 99 and r["posSnr"] >= 0]
out["recovery"]["posFirst"], out["recovery"]["posLast"] = rec[0]["posBpm"], rec[-1]["posBpm"]
out["recovery"]["recoverySeconds"] = round(rec[-1]["t"] - rec[0]["t"], 1)
out["rest"]["posBpmRange"] = [min(r["posBpm"] for r in rows if phase(r["t"]) == "rest" and r["posSnr"] >= 0),
                              max(r["posBpm"] for r in rows if phase(r["t"]) == "rest" and r["posSnr"] >= 0)]
# the displayed median at t 54.25 mixed a jumping-jack window into four rest windows
out["staleShown"] = next({"t": r["t"], "shown": r["shown"], "pos": r["posBpm"]} for r in rows if 43 < r["t"] < 69 and r["shown"])
# the page's display rule since version 2, replayed: SNR >= 0, not moving (faceDrift <= 0.08), peak not on the
# band's lower edge, median of the passing windows in the last 5 s
hist, shown2 = [], []
for r in rows:
    good = r["posSnr"] >= 0 and not r["suspect"] and r["faceDrift"] <= 0.08 and r["posBpm"] >= 43
    if good: hist.append((r["t"], r["posBpm"]))
    hist = [h for h in hist if h[0] > r["t"] - 5]
    shown2.append(S.median(b for _, b in hist) if good else None)
for p in ("rest", "exercise", "recovery"):
    v = [x for r, x in zip(rows, shown2) if phase(r["t"]) == p]
    out[p]["v2ShownFrac"] = round(sum(x is not None for x in v) / len(v), 2)
    out[p]["v2ShownRange"] = [round(min(x for x in v if x), 1), round(max(x for x in v if x), 1)] if any(v) else None
out["frameMs"] = round(d["frameMs"], 2); out["detectMs"] = round(d["detectMs"], 1)
(R / "real-summary.json").write_text(json.dumps(out, indent=1) + "\n")
print(json.dumps(out, indent=1))
