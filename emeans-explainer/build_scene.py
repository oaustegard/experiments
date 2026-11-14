#!/usr/bin/env python3
"""Build scene.json: a narrated Kinescript explainer of ../emeans-pooled-residual, in the Muninn brand.

Nine voice lines in Muninn's voice; every section's layers start on words of its own line ("voK:mark") and
leave as the next line starts, so re-voicing at another rate re-times the whole video. Every number shown or
spoken comes from ../emeans-pooled-residual/summary.json (see README.md for the mapping).
"""
from __future__ import annotations

import json
import math
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent
W, H = 1920, 1080
TEXT, MUTED, ACC, ACC2, POS, SURF = "$text", "$muted", "$accent", "$accent2", "$positive", "$surface"
RATE = 0.9
COLBERT = "[ColBERT](/kQlbˈɛə/)"

VO = [
    # 1 title
    "Most code that clusters vectors runs Lloyd's k-means. [[emeans]]Meta's new e-means promises the same "
    "result in a tenth of the passes. We measured [[where]]where that holds, and where it doesn't.",
    # 2 Lloyd
    "Lloyd's algorithm alternates two steps. [[assign]]Assign every point to its nearest centre. [[move]]Move "
    "each centre to the mean of its points. [[repeat]]Repeat, often twenty-five times, and every round reads "
    "all the data.",
    # 3 e-means
    "e-means reads the data in [[batches]]small batches. Each batch [[nudges]]nudges the centres with a moving "
    "average, and [[revive]]empty clusters are split off busy ones. Two passes usually land where twenty-five "
    "rounds of Lloyd do.",
    # 4 grid
    "We timed both on sixty-one clustering jobs, up to a million points and sixteen thousand clusters. "
    "[[high]]With thirty-two or more dimensions and at least twenty-five points per cluster, two passes of "
    "e-means finished one and a half to twelve times sooner than twenty-five rounds of Lloyd, and ended within "
    "one percent of Lloyd's error, or below it. [[torch]]Its PyTorch version, which also swaps in a faster "
    "search, was up to fifty-two times sooner.",
    # 5 limits
    "[[few]]On our synthetic data, below about twenty-five points per cluster, e-means stopped at a higher "
    "error than Lloyd. [[three]]And on three-dimensional data like pixel colours, with a thousand clusters or more, it was "
    "slower. [[faiss]]The faiss search inside it costs up to fifty times more per distance on small batches "
    "there.",
    # 6 ColBERT
    f"We built this for a {COLBERT} search index, which stores a vector for every token, and we shrink "
    f"each one to a few bits, as an offset from [[mean]]one shared average. Offset each token instead from its "
    f"[[nearest]]nearest of sixteen thousand centroids, and at one bit the ranking scores [[gain]]up to two "
    f"points higher. That's about what a second bit buys.",
    # 7 table
    "The centroids have to be stored, though. [[table]]On a five-thousand-document collection, that table "
    "costs as much as the bit it saves, so the two tie. [[large]]A collection large enough to spread it thin "
    "would get the same quality in about a third fewer bytes. That last number is arithmetic, not a "
    "measurement.",
    # 8 trainer
    "And the trainer hardly mattered for search. [[lloyd2]]Two rounds of Lloyd, twenty-five rounds, or "
    "e-means: [[same]]the rankings mostly came out within half a point. [[faster]]e-means fit the centroids "
    "about twelve times faster than twenty-five rounds of Lloyd.",
    # 9 takeaway
    "[[r1]]Reach for e-means on high-dimensional data with plenty of points per cluster. [[r2]]Keep Lloyd "
    "for small clusters and low-dimensional faiss jobs. [[r3]]And check whether your downstream metric even "
    "notices.",
]
N = len(VO)
layers: list[dict] = []
audio: list[dict] = []


def nxt(k: int) -> str:
    return f"vo{k + 1}.in+0.05" if k < N else "end"


def add(layer: dict, k: int, exit_=True) -> dict:
    if "dur" not in layer:
        layer.setdefault("until", nxt(k))
    if exit_ and "exit" not in layer:
        layer["exit"] = {"name": "fade", "dur": 0.3}
    layers.append(layer)
    return layer


def text(id_, s, x, y, k, at, style="body", fill=TEXT, **kw):
    return add({"id": id_, "type": "text", "text": s, "x": x, "y": y, "textStyle": style, "fill": fill,
                "at": at, **kw}, k)


for k, line in enumerate(VO, 1):
    audio.append({"id": f"vo{k}", "role": "voice", "text": line, "rate": RATE,
                  "at": 0.6 if k == 1 else f"vo{k - 1}.out+0.35"})
    layers.append({"id": f"cc{k}", "type": "captions", "voice": f"vo{k}", "mode": "karaoke", "active": ACC,
                   "fill": TEXT, "box": {"fill": SURF, "pad": [10, 22], "radius": 10}, "size": 38, "y": "91%", "maxWidth": 1500, "at": f"vo{k}.in", "until": nxt(k)})

# ------------------------------------------------------------------ shared point cloud (sections 2 and 3)
rng = random.Random(11)
BLOBS = [(700, 430), (1180, 360), (960, 700)]
PTS = [(b, round(cx + rng.gauss(0, 90)), round(cy + rng.gauss(0, 70))) for b, (cx, cy) in enumerate(BLOBS)
       for _ in range(16)]
START = [(560, 640), (1300, 640), (980, 300)]   # deliberately poor initial centres
COLS = [ACC, ACC2, POS]

# ------------------------------------------------------------------ 1 title
k = 1
text("kicker", "K-MEANS  ·  E-MEANS", "50%", 330, k, "vo1.in+0.2", style="label", fill=ACC,
     letterSpacing=6, enter={"name": "fade", "dur": 0.6})
text("title", "When e-means beats k-means", "50%", 450, k, "vo1.in+0.4", style="title", size=100,
     split="words", enter={"name": "rise", "dur": 0.6, "stagger": 0.15})
text("sub", "61 clustering jobs and one search index", "50%", 590, k, "vo1:emeans", fill=MUTED, enter="fade")
add({"id": "tbar", "type": "rect", "x": "50%", "y": 660, "w": 360, "h": 6, "fill": ACC, "radius": 3,
     "at": "vo1:where", "enter": "wipe"}, k)

# ------------------------------------------------------------------ 2 Lloyd
k = 2
text("s2h", "Lloyd's k-means", "50%", 110, k, "vo2.in+0.2", style="heading", enter="rise")
for n, (b, px, py) in enumerate(PTS):
    add({"id": f"s2g{n}", "type": "circle", "x": px, "y": py, "r": 11, "fill": MUTED,
         "at": f"vo2.in+{0.3 + 0.015 * n:.2f}", "until": "vo2:assign+0.3", "enter": "pop"}, k, exit_=False)
    add({"id": f"s2c{n}", "type": "circle", "x": px, "y": py, "r": 11, "fill": COLS[b],
         "at": f"vo2:assign+{0.01 * n:.2f}", "enter": {"name": "fade", "dur": 0.25}}, k)
CENTRE = {"type": "rect", "w": 34, "h": 34, "radius": 4, "rotate": 45, "outline": {"width": 4, "color": TEXT}}
for c, ((sx, sy), (ex, ey)) in enumerate(zip(START, BLOBS)):
    add({**CENTRE, "id": f"s2m{c}", "fill": COLS[c], "x": sx, "y": sy, "at": "vo2.in+0.8", "until": "vo2:move",
         "enter": "pop"}, k, exit_=False)
    add({**CENTRE, "id": f"s2M{c}", "fill": COLS[c], "at": "vo2:move",
         "x": [[0, sx], [0.9, ex, "inOutCubic"]], "y": [[0, sy], [0.9, ey, "inOutCubic"]]}, k)
text("s2a", "1  assign: nearest centre", 150, 930 - 140, k, "vo2:assign", anchor="left",
     enter="fade", until="vo2:move")
text("s2b", "2  move: centre to the mean", 150, 930 - 140, k, "vo2:move", anchor="left",
     enter="fade", until="vo2:repeat")
text("s2r", "× 25 rounds  ·  every round reads all the data", "50%", 930 - 140, k, "vo2:repeat",
     style="label", fill=ACC, enter="fade")

# ------------------------------------------------------------------ 3 e-means
k = 3
text("s3h", "e-means", "50%", 110, k, "vo3.in+0.2", style="heading", enter="rise")
for n, (b, px, py) in enumerate(PTS):
    batch = n % 4
    add({"id": f"s3g{n}", "type": "circle", "x": px, "y": py, "r": 11, "fill": MUTED, "opacity": 0.45,
         "at": "vo3.in+0.2", "enter": "fade"}, k)
    add({"id": f"s3b{n}", "type": "circle", "x": px, "y": py, "r": 13, "fill": COLS[b],
         "at": f"vo3:batches+{0.55 * batch:.2f}", "dur": 0.6, "enter": "pop",
         "exit": {"name": "fade", "dur": 0.2}}, k, exit_=False)
for c, ((sx, sy), (ex, ey)) in enumerate(zip(START, BLOBS)):
    add({**CENTRE, "id": f"s3m{c}", "fill": COLS[c], "x": sx, "y": sy, "at": "vo3.in+0.2",
         "until": "vo3:batches", "enter": "pop"}, k, exit_=False)
    xs = [[0, sx]] + [[round(0.55 * j + 0.3, 2), round(sx + (ex - sx) * f), "outCubic"]
                      for j, f in enumerate((0.45, 0.75, 0.9, 1.0))]
    ys = [[0, sy]] + [[round(0.55 * j + 0.3, 2), round(sy + (ey - sy) * f), "outCubic"]
                      for j, f in enumerate((0.45, 0.75, 0.9, 1.0))]
    add({**CENTRE, "id": f"s3M{c}", "fill": COLS[c], "x": xs, "y": ys, "at": "vo3:batches"}, k)
text("s3a", "small batches  ·  moving-average updates", "50%", 185, k, "vo3:nudges", style="label",
     fill=MUTED, enter="fade", until="vo3:revive")
text("s3b", "empty clusters split off busy ones", "50%", 185, k, "vo3:revive", style="label", fill=MUTED,
     enter="fade")
text("s3c", "2 passes ≈ 25 Lloyd rounds", "50%", 880, k, "vo3:revive+1.2", style="body", fill=ACC,
     weight=700, enter="pop")

# ------------------------------------------------------------------ 4 grid
k = 4
text("s4h", "61 clustering jobs: n to 1M, k to 16,384", "50%", 100, k, "vo4.in+0.2", style="heading",
     size=60, enter="rise")
text("s4m", "time for 25 Lloyd rounds ÷ time for 2 e-means passes (median, NumPy, ≥ 25 points per cluster)",
     "50%", 175, k, "vo4:high", style="label", fill=MUTED, enter="fade")
text("s4lb", "at 32+ dimensions e-means ended within 1% of Lloyd's error or below it", "50%", 222, k,
     "vo4:high+0.6", style="label", fill=MUTED, size=24, enter="fade")
add({"use": "barChart", "id": "s4c", "data": [["d = 3", 0.96], ["d = 32", 2.65], ["d = 128", 4.91],
                                              ["d = 768", 4.42]],
     "format": "{:.2f}×", "x": "50%", "y": 500, "w": 1100, "h": 470, "max": 6, "highlight": [2, 3, 4],
     "at": "vo4:high", "until": nxt(4)}, k, exit_=False)
add({"id": "s4ref", "type": "line", "points": [[410, 657], [1510, 657]], "stroke": TEXT, "strokeWidth": 3,
     "dash": [14, 10], "at": "vo4:high+1.5", "enter": "draw"}, k)
text("s4one", "1× = same time", 1530, 657, k, "vo4:high+1.7", style="label", fill=TEXT, anchor="left", enter="fade")
text("s4t", "PyTorch on CPU, with its own search: up to 52×", "50%", 850, k, "vo4:torch", style="body", fill=ACC, weight=700,
     enter="pop")

# ------------------------------------------------------------------ 5 limits
k = 5
text("s5h", "Where it falls short", "50%", 130, k, "vo5.in+0.1", style="heading", enter="rise")
for i, (mark, l1, l2, l3) in enumerate([
        ("few", "< 25 points per cluster", "ends at up to 1.38× Lloyd's error", "synthetic data, 128+ dimensions"),
        ("three", "3 dimensions, k ≥ 1,024", "0.1–0.5× Lloyd's speed", "faiss search: up to 52× the cost per distance")]):
    cx = 530 if i == 0 else 1390
    add({"id": f"s5c{i}", "type": "rect", "x": cx, "y": 470, "w": 800, "h": 440, "radius": 24, "fill": SURF,
         "at": f"vo5:{mark}", "enter": "pop"}, k)
    text(f"s5a{i}", l1, cx, 340, k, f"vo5:{mark}+0.1", style="body", weight=700, enter="fade")
    text(f"s5b{i}", l2, cx, 450, k, f"vo5:{mark}+0.3", style="body", fill=ACC, weight=700, size=46, enter="fade")
    text(f"s5d{i}", l3, cx, 560, k, f"vo5:{'faiss' if i else mark}+0.4", style="label", fill=MUTED,
         maxWidth=600, enter="fade")

# ------------------------------------------------------------------ 6 ColBERT
k = 6
text("s6h", "A ColBERT index, coded to 1 bit per number", "50%", 100, k, "vo6.in+0.2", style="heading",
     size=60, enter="rise")
OX, OY = 520, 480
rng6 = random.Random(5)
toks = [(OX + 170 + rng6.gauss(0, 120), OY - 60 + rng6.gauss(0, 110)) for _ in range(18)]
for n, (tx, ty) in enumerate(toks):
    add({"id": f"s6t{n}", "type": "circle", "x": round(tx), "y": round(ty), "r": 10, "fill": ACC2,
         "at": f"vo6.in+{0.4 + 0.03 * n:.2f}", "enter": "pop"}, k)
mx, my = sum(t[0] for t in toks) / 18, sum(t[1] for t in toks) / 18
add({"id": "s6mu", "type": "rect", "x": round(mx), "y": round(my), "w": 30, "h": 30, "rotate": 45, "radius": 4,
     "fill": MUTED, "at": "vo6:mean", "enter": "pop", "until": "vo6:nearest+0.2"}, k)
for n, (tx, ty) in enumerate(toks[:8]):
    add({"id": f"s6l{n}", "type": "line", "points": [[round(mx), round(my)], [round(tx), round(ty)]],
         "stroke": MUTED, "strokeWidth": 3, "at": f"vo6:mean+{0.05 * n:.2f}", "until": "vo6:nearest",
         "enter": "draw"}, k)
cents = [(OX + 60, OY - 150), (OX + 300, OY - 120), (OX + 120, OY + 60), (OX + 330, OY + 40)]
for c, (cx, cy) in enumerate(cents):
    add({"id": f"s6k{c}", "type": "rect", "x": cx, "y": cy, "w": 26, "h": 26, "rotate": 45, "radius": 4,
         "fill": ACC, "at": f"vo6:nearest+{0.08 * c:.2f}", "enter": "pop"}, k)
for n, (tx, ty) in enumerate(toks[:10]):
    cx, cy = min(cents, key=lambda c: (c[0] - tx) ** 2 + (c[1] - ty) ** 2)
    add({"id": f"s6r{n}", "type": "line", "points": [[cx, cy], [round(tx), round(ty)]], "stroke": ACC,
         "strokeWidth": 3, "at": f"vo6:nearest+{0.4 + 0.04 * n:.2f}", "enter": "draw"}, k)
text("s6lab1", "offset from one average", OX + 170, 790, k, "vo6:mean", style="label", fill=MUTED, enter="fade",
     until="vo6:nearest")
text("s6lab2", "offset from the nearest of 16,384 centroids", OX + 170, 790, k, "vo6:nearest", style="label",
     fill=ACC, enter="fade")
add({"use": "barChart", "id": "s6b", "data": [["one average, 1 bit", 0.730], ["centroids, 1 bit", 0.751],
                                              ["one average, 2 bits", 0.746]],
     "format": "{:.3f}", "x": 1380, "y": 500, "w": 760, "h": 440, "max": 0.78, "highlight": 2, "size": 26,
     "at": "vo6:gain", "until": nxt(6)}, k, exit_=False)
text("s6m", "SciFact nDCG@10, a quarter of the tokens kept", 1380, 210, k, "vo6:gain", style="label",
     fill=MUTED, enter="fade")

# ------------------------------------------------------------------ 7 table
k = 7
text("s7h", "The centroid table has to be stored too", "50%", 110, k, "vo7.in+0.2", style="heading", size=60,
     enter="rise")
S = 0.32   # px per byte
BARS = [("one average, 2 bits", [("codes + scalars", 2314, ACC2)], "vo7:table"),
        ("centroids, 1 bit, 5k documents", [("codes + scalars + cell", 1490, ACC), ("table", 809, MUTED)], "vo7:table+0.6"),
        ("centroids, 1 bit, large corpus", [("codes + scalars + cell", 1490, ACC)], "vo7:large")]
for i, (label, segs, at) in enumerate(BARS):
    y = 330 + 190 * i
    text(f"s7l{i}", label, 200, y - 62, k, at, style="label", fill=TEXT, anchor="left", enter="fade")
    x = 200
    for j, (nm, b, col) in enumerate(segs):
        add({"id": f"s7b{i}_{j}", "type": "rect", "x": x, "y": y, "w": round(b * S), "h": 64, "anchor": "left",
             "fill": col, "radius": 6, "at": f"{at}+{0.3 * j:.1f}", "enter": "wipe"}, k)
        x += round(b * S) + 6
    total = sum(b for _, b, _ in segs)
    text(f"s7t{i}", f"{total / 1000:.1f} KB per document", x + 30, y, k, f"{at}+0.5", style="label",
         fill=ACC if i == 2 else TEXT, anchor="left", enter="fade")
text("s7n", "same quality as 2 bits on one average (within noise)", "50%", 830 - 20, k, "vo7:large+1.0",
     style="label", fill=MUTED, enter="fade")

# ------------------------------------------------------------------ 8 trainer
k = 8
text("s8h", "The trainer hardly mattered for search", "50%", 110, k, "vo8.in+0.2", style="heading", size=60,
     enter="rise")
TR = [("Lloyd, 2 rounds", 0.754, 25.3), ("Lloyd, 25 rounds", 0.751, 321.3), ("e-means, 2 passes", 0.751, 26.1)]
for i, (nm, nd, secs) in enumerate(TR):
    y = 320 + 170 * i
    text(f"s8n{i}", nm, 200, y, k, f"vo8:lloyd2+{0.25 * i:.2f}", style="body", anchor="left", enter="fade")
    text(f"s8d{i}", f"{nd:.3f}", 760, y, k, f"vo8:same+{0.1 * i:.1f}", style="body", fill=ACC2, weight=700,
         font="DM Mono", enter="fade")
    add({"id": f"s8f{i}", "type": "rect", "x": 950, "y": y, "w": max(8, round(secs * 2.0)), "h": 48,
         "anchor": "left", "fill": ACC if i == 1 else POS, "radius": 6, "at": f"vo8:faster+{0.15 * i:.2f}",
         "enter": "wipe"}, k)
    text(f"s8s{i}", f"{secs:.0f} s", 950 + max(8, round(secs * 2.0)) + 24, y, k, f"vo8:faster+{0.15 * i + 0.3:.2f}",
         style="label", anchor="left", enter="fade")
text("s8c1", "nDCG@10", 760, 220, k, "vo8:same", style="label", fill=MUTED, enter="fade")
text("s8c2", "fit time, 16,384 centroids", 1200, 220, k, "vo8:faster", style="label", fill=MUTED, enter="fade")

# ------------------------------------------------------------------ 9 takeaway
k = 9
text("s9h", "Where e-means pays", "50%", 200, k, "vo9.in+0.1", style="heading", fill=MUTED, enter="rise")
for i, (mark, s) in enumerate([("r1", "1   ≥ 32 dimensions, ≥ 25 points per cluster"),
                               ("r2", "2   Keep Lloyd for small clusters and 3-d faiss jobs"),
                               ("r3", "3   Check whether your metric notices")]):
    text(f"s9l{i}", s, 300, 370 + 130 * i, k, f"vo9:{mark}", style="body", size=56, weight=700,
         fill=[ACC, ACC2, POS][i], anchor="left", enter={"name": "rise", "dur": 0.4})
end = add({"id": "s9f", "type": "text", "text": "oaustegard/experiments  ·  emeans-pooled-residual", "x": "50%",
           "y": 860, "textStyle": "label", "fill": MUTED, "at": "vo9.out+0.2", "dur": 3.0, "enter": "fade"}, k)
end.pop("until", None)

layers.append({"use": "bug"})
scene = {"brand": "brand", "canvas": {"w": W, "h": H, "fps": 30, "duration": "auto", "poster": "title.in+1.2"},
         "layers": layers, "audio": audio}
(HERE / "scene.json").write_text(json.dumps(scene, indent=1, ensure_ascii=False))
print(f"{len(layers)} layers, {len(audio)} voice lines -> scene.json")
